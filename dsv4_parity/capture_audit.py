from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .capture_reconcile import reconcile_capture
from .io import read_json, read_jsonl
from .raw_logits import NUMPY_DTYPES, bfloat16_to_float32


def _decode(payload: bytes, dtype_name: str) -> np.ndarray:
    if dtype_name == "bfloat16":
        return bfloat16_to_float32(np.frombuffer(payload, dtype="<u2"))
    try:
        return np.frombuffer(payload, dtype=NUMPY_DTYPES[dtype_name]).astype(np.float32)
    except KeyError as exc:
        raise ValueError(f"unsupported raw dtype: {dtype_name}") from exc


def audit_capture(
    capture_dir: str | Path,
    generation: dict[str, Any],
    run_id: str,
    *,
    max_error_examples: int = 100,
) -> dict[str, Any]:
    directory = Path(capture_dir)
    all_rows = read_jsonl(directory / "capture.jsonl")
    formal_rows, reconciliation = reconcile_capture(directory, generation, run_id)
    formal_by_offset = {
        int(row["raw_offset_bytes"]): row
        for row in formal_rows
        if row["branch_status"] in {"committed_ar", "committed_spec"}
    }
    errors: list[dict[str, Any]] = []

    raw_files = {str(row["raw_file"]) for row in all_rows}
    if len(raw_files) != 1:
        raise ValueError(f"expected one raw file, got {sorted(raw_files)}")
    raw_path = directory / next(iter(raw_files))
    raw_size = raw_path.stat().st_size
    status_path = directory / "capture_status.json"
    status = read_json(status_path) if status_path.exists() else None

    offsets = [int(row["raw_offset_bytes"]) for row in all_rows]
    duplicate_offsets = len(offsets) - len(set(offsets))
    out_of_bounds_rows = 0
    metadata_top1_value_mismatches = 0
    metadata_max_value_mismatches = 0
    metadata_nonfinite_mismatches = 0
    committed_expected_is_max = 0
    committed_expected_exact_argmax = 0
    committed_rows_checked = 0
    max_tie_rows = 0

    def record(kind: str, row: dict[str, Any], **details: Any) -> None:
        if len(errors) < max_error_examples:
            errors.append(
                {
                    "kind": kind,
                    "request_id": row.get("request_id"),
                    "request_step": row.get("request_step"),
                    "logits_row_within_request": row.get("logits_row_within_request"),
                    "raw_offset_bytes": row.get("raw_offset_bytes"),
                    **details,
                }
            )

    with raw_path.open("rb") as handle:
        for row in sorted(all_rows, key=lambda item: int(item["raw_offset_bytes"])):
            offset = int(row["raw_offset_bytes"])
            length = int(row["raw_length_bytes"])
            expected_length = int(row["vocab_size"]) * (
                2 if row["raw_dtype"] in {"bfloat16", "float16"} else NUMPY_DTYPES[row["raw_dtype"]].itemsize
            )
            if length != expected_length or offset < 0 or offset + length > raw_size:
                out_of_bounds_rows += 1
                record(
                    "raw_range_invalid",
                    row,
                    raw_size=raw_size,
                    raw_length=length,
                    expected_length=expected_length,
                )
                continue
            handle.seek(offset)
            payload = handle.read(length)
            vector = _decode(payload, str(row["raw_dtype"]))
            finite = np.isfinite(vector)
            nonfinite = int((~finite).sum())
            if nonfinite != int(row["nonfinite_count"]):
                metadata_nonfinite_mismatches += 1
                record("nonfinite_count_mismatch", row, decoded=nonfinite, stored=row["nonfinite_count"])
            if not finite.any():
                continue
            maximum = float(np.max(vector[finite]))
            stored_token = int(row["top20_token_ids"][0])
            stored_value = float(row["top20_logits"][0])
            if float(vector[stored_token]) != stored_value:
                metadata_top1_value_mismatches += 1
                record(
                    "stored_top1_value_mismatch",
                    row,
                    decoded=float(vector[stored_token]),
                    stored=stored_value,
                )
            if stored_value != maximum:
                metadata_max_value_mismatches += 1
                record("stored_top1_not_maximum", row, stored=stored_value, maximum=maximum)

            formal = formal_by_offset.get(offset)
            if formal is None:
                continue
            committed_rows_checked += 1
            expected_token = int(formal["expected_output_token_id"])
            expected_value = float(vector[expected_token])
            if expected_value == maximum:
                committed_expected_is_max += 1
            else:
                record(
                    "committed_token_not_raw_maximum",
                    row,
                    expected_token_id=expected_token,
                    expected_logit=expected_value,
                    maximum=maximum,
                    raw_argmax=int(np.argmax(vector)),
                )
            argmax = int(np.argmax(vector))
            if argmax == expected_token:
                committed_expected_exact_argmax += 1
            if int(np.count_nonzero(vector == maximum)) > 1:
                max_tie_rows += 1

    status_consistent = bool(
        status is not None
        and status.get("complete") is True
        and int(status.get("raw_bytes", -1)) == raw_size
    )
    return {
        "schema_version": 1,
        "kind": "raw_capture_audit",
        "run_id": run_id,
        "capture_dir": str(directory),
        "raw_file": str(raw_path),
        "raw_size_bytes": raw_size,
        "capture_status": status,
        "capture_status_consistent": status_consistent,
        "metadata_rows_checked": len(all_rows),
        "duplicate_raw_offsets": duplicate_offsets,
        "out_of_bounds_rows": out_of_bounds_rows,
        "metadata_top1_value_mismatch_rows": metadata_top1_value_mismatches,
        "metadata_max_value_mismatch_rows": metadata_max_value_mismatches,
        "metadata_nonfinite_mismatch_rows": metadata_nonfinite_mismatches,
        "committed_rows_checked": committed_rows_checked,
        "committed_expected_is_raw_max_count": committed_expected_is_max,
        "committed_expected_is_raw_max_rate": (
            committed_expected_is_max / committed_rows_checked if committed_rows_checked else None
        ),
        "committed_expected_exact_argmax_count": committed_expected_exact_argmax,
        "committed_expected_exact_argmax_rate": (
            committed_expected_exact_argmax / committed_rows_checked if committed_rows_checked else None
        ),
        "raw_max_tie_rows": max_tie_rows,
        "reconciliation": reconciliation,
        "raw_structure_passed": (
            status_consistent
            and duplicate_offsets == 0
            and out_of_bounds_rows == 0
            and metadata_top1_value_mismatches == 0
            and metadata_max_value_mismatches == 0
            and metadata_nonfinite_mismatches == 0
        ),
        "errors": errors,
    }
