from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, read_json, sha256_json
from .metrics import logits_metrics, near_tie
from .raw_logits import load_raw_logits, raw_bytes


PAIR_SPECS = (
    ("self_ar", "AR_1", "AR_2", 8),
    ("self_q7", "Q7_1", "Q7_2", 7),
    ("self_q8", "Q8_1", "Q8_2", 8),
    ("shape_ar_q7", "AR_1", "Q7_1", 7),
    ("shape_ar_q7", "AR_2", "Q7_2", 7),
    ("shape_ar_q8", "AR_1", "Q8_1", 8),
    ("shape_ar_q8", "AR_2", "Q8_2", 8),
    ("shape_q7_q8", "Q7_1", "Q8_1", 7),
    ("shape_q7_q8", "Q7_2", "Q8_2", 7),
)


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    max_abs = [float(row["max_abs_error"]) for row in rows]
    mean_abs = [float(row["mean_abs_error"]) for row in rows]
    rel_l2 = [float(row["relative_l2_error"]) for row in rows]
    cosine = [float(row["cosine_similarity"]) for row in rows if row["cosine_similarity"] is not None]
    return {
        "rows": len(rows),
        "bitwise_equal_count": sum(bool(row["bitwise_equal"]) for row in rows),
        "bitwise_equal_rate": sum(bool(row["bitwise_equal"]) for row in rows) / len(rows),
        "top1_flip_count": sum(not row["top1_match"] for row in rows),
        "top1_flip_rate": sum(not row["top1_match"] for row in rows) / len(rows),
        "top5_overlap_mean": float(np.mean([row["top5_overlap"] for row in rows])),
        "top20_overlap_mean": float(np.mean([row["top20_overlap"] for row in rows])),
        "max_abs_error": {
            "median": float(np.median(max_abs)),
            "p95": _percentile(max_abs, 95),
            "p99": _percentile(max_abs, 99),
            "max": max(max_abs),
        },
        "mean_abs_error": {
            "median": float(np.median(mean_abs)),
            "p95": _percentile(mean_abs, 95),
            "max": max(mean_abs),
        },
        "relative_l2_error": {
            "median": float(np.median(rel_l2)),
            "p95": _percentile(rel_l2, 95),
            "max": max(rel_l2),
        },
        "cosine_similarity": {
            "median": float(np.median(cosine)) if cosine else None,
            "minimum": min(cosine) if cosine else None,
        },
        "near_tie_flip_counts": {
            threshold: sum((not row["top1_match"]) and row[field] for row in rows)
            for threshold, field in (
                ("1e-4", "near_tie_1e-4"),
                ("1e-3", "near_tie_1e-3"),
                ("1e-2", "near_tie_1e-2"),
            )
        },
    }


def _compare_rows(
    directory: Path,
    window: dict[str, Any],
    kind: str,
    left_path: str,
    right_path: str,
    width: int,
) -> list[dict[str, Any]]:
    by_key = {(row["path"], int(row["row"])): row for row in window["rows"]}
    result = []
    for row_index in range(width):
        left = by_key[(left_path, row_index)]
        right = by_key[(right_path, row_index)]
        if (
            int(left["input_position"]) != int(right["input_position"])
            or int(left["prediction_position"]) != int(right["prediction_position"])
        ):
            raise ValueError(
                f"position mismatch {window['window_id']} {left_path}/{right_path} row {row_index}"
            )
        metrics = logits_metrics(
            load_raw_logits(directory, left), load_raw_logits(directory, right)
        )
        metrics["bitwise_equal"] = raw_bytes(directory, left) == raw_bytes(directory, right)
        result.append(
            {
                "comparison_kind": kind,
                "window_id": window["window_id"],
                "prompt_id": window["prompt_id"],
                "window_kind": window["window_kind"],
                "output_start": window["output_start"],
                "row": row_index,
                "prediction_position": left["prediction_position"],
                "left_path": left_path,
                "right_path": right_path,
                "reference_top20_token_ids": left["top20_token_ids"],
                "reference_top20_logits": left["top20_logits"],
                "candidate_top20_token_ids": right["top20_token_ids"],
                "candidate_top20_logits": right["top20_logits"],
                "near_tie_1e-4": near_tie(metrics["reference_top1_top2_gap"], 1e-4),
                "near_tie_1e-3": near_tie(metrics["reference_top1_top2_gap"], 1e-3),
                "near_tie_1e-2": near_tie(metrics["reference_top1_top2_gap"], 1e-2),
                **metrics,
            }
        )
    return result


def compare_replay(replay_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    directory = Path(replay_dir)
    window_paths = sorted(
        path
        for path in directory.glob("*.json")
        if not path.name.endswith("status.json")
        and not path.name.startswith("request_")
        and "comparison" not in path.stem
        and path.name != Path(output_path).name
    )
    windows = [read_json(path) for path in window_paths]
    if not windows:
        raise ValueError(f"no replay windows found in {directory}")
    all_rows: list[dict[str, Any]] = []
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validation_rows = []
    for window in windows:
        fixed = [int(value) for value in window["fixed_input_token_ids"]]
        mapping_ok = True
        lookup = {(row["path"], int(row["row"])): row for row in window["rows"]}
        for path_name, width in (
            ("AR_1", 8),
            ("Q7_1", 7),
            ("Q8_1", 8),
            ("Q8_2", 8),
            ("Q7_2", 7),
            ("AR_2", 8),
        ):
            path_rows = [lookup[(path_name, index)] for index in range(width)]
            mapping_ok = mapping_ok and [row["input_token_id"] for row in path_rows] == fixed[:width]
            positions = [int(row["input_position"]) for row in path_rows]
            mapping_ok = mapping_ok and positions == list(range(positions[0], positions[0] + width))
        for kind, left, right, width in PAIR_SPECS:
            compared = _compare_rows(directory, window, kind, left, right, width)
            all_rows.extend(compared)
            by_kind[kind].extend(compared)

        causal = _compare_rows(
            directory,
            window,
            "causal_mask",
            "Q8_2",
            "CAUSAL_Q8_MUTATE_LAST",
            7,
        )
        all_rows.extend(causal)
        by_kind["causal_mask"].extend(causal)
        q8_noise = [
            row["max_abs_error"]
            for row in by_kind["self_q8"]
            if row["window_id"] == window["window_id"] and row["row"] < 7
        ]
        causal_error = [row["max_abs_error"] for row in causal]
        tolerance = max(q8_noise, default=0.0) * 2.0 + 1e-7
        causal_passed = (
            all(row["top1_match"] for row in causal)
            and max(causal_error, default=0.0) <= tolerance
        )
        validation_rows.append(
            {
                "window_id": window["window_id"],
                "state_restore_passed": bool(window["state_restore_passed"]),
                "input_and_position_mapping_passed": mapping_ok,
                "causal_mask_passed": causal_passed,
                "causal_max_abs_error": max(causal_error, default=0.0),
                "same_q8_repeat_max_abs_error_rows_0_6": max(q8_noise, default=0.0),
                "causal_tolerance": tolerance,
            }
        )

    raw_path = directory / "raw_logits.bin"
    max_end = max(
        int(row["raw_offset_bytes"]) + int(row["raw_length_bytes"])
        for window in windows
        for row in window["rows"]
    )
    aggregates = {kind: _aggregate(rows) for kind, rows in sorted(by_kind.items())}
    result = {
        "schema_version": 1,
        "kind": "fixed_state_replay_comparison",
        "replay_dir": str(directory),
        "num_windows": len(windows),
        "raw_size_bytes": raw_path.stat().st_size,
        "raw_offsets_complete": raw_path.stat().st_size == max_end,
        "validations": validation_rows,
        "all_state_restores_passed": all(row["state_restore_passed"] for row in validation_rows),
        "all_input_and_position_mappings_passed": all(
            row["input_and_position_mapping_passed"] for row in validation_rows
        ),
        "all_causal_mask_checks_passed": all(row["causal_mask_passed"] for row in validation_rows),
        "aggregates": aggregates,
        "rows": all_rows,
    }
    result["artifact_sha256"] = sha256_json(result)
    atomic_write_json(output_path, result)
    return result
