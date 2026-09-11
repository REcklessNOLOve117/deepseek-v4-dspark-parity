import json
from pathlib import Path

from dsv4_parity.capture_reconcile import reconcile_capture, token_history_sha256


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_reconcile_uses_committed_stream_and_marks_rejection(tmp_path: Path):
    generation = {
        "prompts": [
            {
                "prompt_id": "p1",
                "prompt_token_ids": [10, 11],
                "token_ids": [20, 21],
            }
        ]
    }
    rows = []
    for step, values in enumerate(((11, 1), (20, 2), (999, 3))):
        input_token, position = values
        rows.append(
            {
                "request_id": "parity:S1:p1",
                "batch_id": "b0" if step == 0 else "b1",
                "request_step": 0 if step == 0 else 1,
                "logits_row_within_request": 0 if step == 0 else step - 1,
                "input_token_id": input_token,
                "input_position": position,
                "prediction_position": position + 1,
                "history_length": position + 1,
                "history_sha256": "stale",
                "num_draft_tokens": 0 if step == 0 else 1,
                "branch_status": "committed_ar" if step == 0 else "pending_acceptance",
            }
        )
    _write_jsonl(tmp_path / "capture.jsonl", rows)
    _write_jsonl(
        tmp_path / "acceptance.jsonl",
        [
            {
                "request_id": "parity:S1:p1",
                "batch_id": "b0",
                "num_sampled": 1,
                "sampled_token_ids": [20],
            },
            {
                "request_id": "parity:S1:p1",
                "batch_id": "b1",
                "num_sampled": 1,
                "sampled_token_ids": [21],
            },
        ],
    )

    reconciled, validation = reconcile_capture(tmp_path, generation, "S1")
    assert [row["branch_status"] for row in reconciled] == [
        "committed_ar",
        "committed_spec",
        "rejected",
    ]
    assert reconciled[1]["history_sha256"] == token_history_sha256([10, 11, 20])
    assert reconciled[1]["committed_output_index"] == 1
    assert validation["control_mapping_passed"]
    assert validation["captured_history_hash_mismatch_rows"] == 2


def test_reconcile_detects_position_mapping_failure(tmp_path: Path):
    generation = {
        "prompts": [{"prompt_id": "p1", "prompt_token_ids": [10], "token_ids": [20]}]
    }
    _write_jsonl(
        tmp_path / "capture.jsonl",
        [
            {
                "request_id": "parity:A1:p1",
                "batch_id": "b0",
                "request_step": 0,
                "logits_row_within_request": 0,
                "input_token_id": 999,
                "input_position": 0,
                "prediction_position": 1,
                "history_length": 1,
                "history_sha256": "stale",
                "num_draft_tokens": 0,
                "branch_status": "committed_ar",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "acceptance.jsonl",
        [
            {
                "request_id": "parity:A1:p1",
                "batch_id": "b0",
                "num_sampled": None,
                "sampled_token_ids": [20],
            }
        ],
    )

    _, validation = reconcile_capture(tmp_path, generation, "A1")
    assert not validation["control_mapping_passed"]
    assert validation["input_mapping_mismatch_rows"] == 1
