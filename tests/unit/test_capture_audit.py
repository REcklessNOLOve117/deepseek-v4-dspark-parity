import json
from pathlib import Path

import numpy as np

from dsv4_parity.capture_audit import audit_capture


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_raw_capture_audit_validates_structure_and_argmax(tmp_path: Path):
    logits = np.asarray([1.0, 3.0, 2.0], dtype="<f2")
    (tmp_path / "raw_logits.bin").write_bytes(logits.tobytes())
    row = {
        "request_id": "parity:A1:p1",
        "batch_id": "b0",
        "request_step": 0,
        "logits_row_within_request": 0,
        "input_token_id": 10,
        "input_position": 0,
        "prediction_position": 1,
        "history_length": 1,
        "history_sha256": "stale",
        "num_draft_tokens": 0,
        "branch_status": "committed_ar",
        "raw_file": "raw_logits.bin",
        "raw_offset_bytes": 0,
        "raw_length_bytes": 6,
        "raw_dtype": "float16",
        "vocab_size": 3,
        "top20_token_ids": [1, 2, 0],
        "top20_logits": [3.0, 2.0, 1.0],
        "nonfinite_count": 0,
    }
    _write_jsonl(tmp_path / "capture.jsonl", [row])
    _write_jsonl(
        tmp_path / "acceptance.jsonl",
        [
            {
                "request_id": "parity:A1:p1",
                "batch_id": "b0",
                "num_sampled": None,
                "sampled_token_ids": [1],
            }
        ],
    )
    (tmp_path / "capture_status.json").write_text(
        json.dumps({"complete": True, "raw_bytes": 6}), encoding="utf-8"
    )
    generation = {
        "prompts": [{"prompt_id": "p1", "prompt_token_ids": [10], "token_ids": [1]}]
    }

    result = audit_capture(tmp_path, generation, "A1")
    assert result["raw_structure_passed"]
    assert result["committed_expected_is_raw_max_rate"] == 1.0
    assert result["committed_expected_exact_argmax_rate"] == 1.0
