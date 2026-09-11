from __future__ import annotations

from pathlib import Path
from typing import Any

from .capture_reconcile import reconcile_capture
from .io import read_jsonl
from .metrics import logits_metrics, near_tie
from .raw_logits import load_raw_logits, raw_bytes


def _prompt_id(request_id: str) -> str:
    parts = request_id.split(":")
    if len(parts) < 3 or parts[0] != "parity":
        raise ValueError(f"unexpected parity request ID: {request_id}")
    return parts[-1]


def _request_run_id(request_id: str) -> str:
    parts = request_id.split(":")
    if len(parts) < 3 or parts[0] != "parity":
        raise ValueError(f"unexpected parity request ID: {request_id}")
    return parts[1]


def load_capture(directory: str | Path) -> list[dict[str, Any]]:
    capture_dir = Path(directory)
    rows = read_jsonl(capture_dir / "capture.jsonl")
    acceptance_path = capture_dir / "acceptance.jsonl"
    acceptance = (
        {row["batch_id"]: row for row in read_jsonl(acceptance_path)}
        if acceptance_path.exists()
        else {}
    )
    result = []
    for original in rows:
        row = dict(original)
        row["prompt_id"] = _prompt_id(row["request_id"])
        if row["branch_status"] == "pending_acceptance":
            mapping = acceptance.get(row["batch_id"])
            if mapping is None or mapping.get("num_sampled") is None:
                row["branch_status"] = "acceptance_unknown"
            elif row["logits_row_within_request"] < int(mapping["num_sampled"]):
                row["branch_status"] = "committed_spec"
            else:
                row["branch_status"] = "rejected"
            row["acceptance"] = mapping
        result.append(row)
    return result


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    result = {}
    for row in rows:
        if row["branch_status"] in {
            "rejected",
            "acceptance_unknown",
            "discarded_after_termination",
        }:
            continue
        key = (row["prompt_id"], int(row["prediction_position"]), row["history_sha256"])
        if key in result:
            raise ValueError(f"duplicate comparable capture row: {key}")
        result[key] = row
    return result


def compare_capture_pair(
    left_run_id: str,
    left_dir: str | Path,
    right_run_id: str,
    right_dir: str | Path,
    left_generation: dict[str, Any] | None = None,
    right_generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (left_generation is None) != (right_generation is None):
        raise ValueError("both generation artifacts must be supplied together")
    if left_generation is not None:
        left_rows, left_validation = reconcile_capture(left_dir, left_generation, left_run_id)
        right_rows, right_validation = reconcile_capture(right_dir, right_generation, right_run_id)
        alignment_source = "reconstructed_committed_token_stream"
    else:
        left_rows = [
            row for row in load_capture(left_dir) if _request_run_id(row["request_id"]) == left_run_id
        ]
        right_rows = [
            row for row in load_capture(right_dir) if _request_run_id(row["request_id"]) == right_run_id
        ]
        left_validation = right_validation = None
        alignment_source = "captured_history_sha256"
    left_index = _index(left_rows)
    right_index = _index(right_rows)
    keys = sorted(left_index.keys() & right_index.keys())
    comparisons = []
    for key in keys:
        left = left_index[key]
        right = right_index[key]
        left_logits = load_raw_logits(left_dir, left)
        right_logits = load_raw_logits(right_dir, right)
        metrics = logits_metrics(left_logits, right_logits)
        metrics["bitwise_equal"] = raw_bytes(left_dir, left) == raw_bytes(right_dir, right)
        comparisons.append(
            {
                "prompt_id": key[0],
                "prediction_position": key[1],
                "history_sha256": key[2],
                "left_graph_mode": left["graph_mode"],
                "right_graph_mode": right["graph_mode"],
                "left_query_length": left["target_query_length"],
                "right_query_length": right["target_query_length"],
                "left_branch_status": left["branch_status"],
                "right_branch_status": right["branch_status"],
                "near_tie_1e-4": near_tie(metrics["reference_top1_top2_gap"], 1e-4),
                "near_tie_1e-3": near_tie(metrics["reference_top1_top2_gap"], 1e-3),
                "near_tie_1e-2": near_tie(metrics["reference_top1_top2_gap"], 1e-2),
                **metrics,
            }
        )
    top1_flips = sum(not row["top1_match"] for row in comparisons)
    return {
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "left_capture_dir": str(left_dir),
        "right_capture_dir": str(right_dir),
        "alignment_source": alignment_source,
        "left_capture_validation": left_validation,
        "right_capture_validation": right_validation,
        "left_rows_total": len(left_rows),
        "right_rows_total": len(right_rows),
        "shared_history_rows": len(comparisons),
        "top1_flip_count": top1_flips,
        "top1_flip_rate": top1_flips / len(comparisons) if comparisons else None,
        "near_tie_flip_counts": {
            threshold: sum((not row["top1_match"]) and row[field] for row in comparisons)
            for threshold, field in (("1e-4", "near_tie_1e-4"), ("1e-3", "near_tie_1e-3"), ("1e-2", "near_tie_1e-2"))
        },
        "rows": comparisons,
    }
