from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl


def token_history_sha256(token_ids: list[int]) -> str:
    """Hash token IDs using the exact representation used by the capture hook."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


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


def _accepted_count(mapping: dict[str, Any] | None, rows: list[dict[str, Any]]) -> int:
    if mapping is not None and mapping.get("num_sampled") is not None:
        return int(mapping["num_sampled"])
    if mapping is not None and mapping.get("sampled_token_ids") is not None:
        return len(mapping["sampled_token_ids"])
    if len(rows) == 1 and int(rows[0].get("num_draft_tokens", 0)) == 0:
        return 1
    raise ValueError(f"cannot determine accepted row count for batch {rows[0]['batch_id']}")


def reconcile_capture(
    capture_dir: str | Path,
    generation: dict[str, Any],
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild committed histories from the externally observed token stream.

    The legacy vLLM runner can leave its CPU token table behind the GPU state
    under asynchronous scheduling.  Consequently, its in-hook history hash is
    retained for diagnostics but is not trusted for cross-run alignment.  This
    function reconstructs every committed row from the fixed prompt, sampler
    acceptance mapping and final output token stream.
    """

    directory = Path(capture_dir)
    capture_rows = [
        dict(row)
        for row in read_jsonl(directory / "capture.jsonl")
        if _request_run_id(row["request_id"]) == run_id
    ]
    acceptance_rows = read_jsonl(directory / "acceptance.jsonl")
    acceptance = {
        row["batch_id"]: row
        for row in acceptance_rows
        if _request_run_id(row["request_id"]) == run_id
    }
    prompts = {row["prompt_id"]: row for row in generation["prompts"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in capture_rows:
        prompt_id = _prompt_id(row["request_id"])
        row["prompt_id"] = prompt_id
        grouped[prompt_id].append(row)

    reconciled: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    sampled_token_mismatches = 0
    captured_history_hash_mismatches = 0
    input_mapping_mismatches = 0
    rejected_rows = 0
    discarded_rows = 0
    accepted_rows = 0

    if set(grouped) != set(prompts):
        raise ValueError(
            f"captured prompts differ from generation prompts: "
            f"capture_only={sorted(set(grouped) - set(prompts))}, "
            f"generation_only={sorted(set(prompts) - set(grouped))}"
        )

    for prompt_id, prompt in prompts.items():
        prompt_tokens = [int(value) for value in prompt["prompt_token_ids"]]
        output_tokens = [int(value) for value in prompt["token_ids"]]
        rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in grouped[prompt_id]:
            rows_by_step[int(row["request_step"])].append(row)
        cursor = 0
        expected_steps = list(range(len(rows_by_step)))
        if sorted(rows_by_step) != expected_steps:
            raise ValueError(f"non-contiguous request steps for {run_id}/{prompt_id}")

        for step in expected_steps:
            step_rows = sorted(
                rows_by_step[step], key=lambda row: int(row["logits_row_within_request"])
            )
            batch_ids = {row["batch_id"] for row in step_rows}
            if len(batch_ids) != 1:
                raise ValueError(f"multiple batch IDs in {run_id}/{prompt_id}/step {step}")
            batch_id = next(iter(batch_ids))
            mapping = acceptance.get(batch_id)
            accepted_count = _accepted_count(mapping, step_rows)
            if accepted_count < 1 or accepted_count > len(step_rows):
                raise ValueError(
                    f"invalid accepted count {accepted_count}/{len(step_rows)} "
                    f"for {run_id}/{prompt_id}/step {step}"
                )

            committed_count = min(accepted_count, max(0, len(output_tokens) - cursor))
            expected_slice = output_tokens[cursor : cursor + committed_count]
            sampled = None if mapping is None else mapping.get("sampled_token_ids")
            if sampled is not None:
                sampled = [int(value) for value in sampled[:committed_count]]
                if sampled != expected_slice:
                    sampled_token_mismatches += 1
                    errors.append(
                        {
                            "kind": "sampled_token_mismatch",
                            "prompt_id": prompt_id,
                            "request_step": step,
                            "output_cursor": cursor,
                            "sampled": sampled,
                            "generation": expected_slice,
                        }
                    )

            for local_index, row in enumerate(step_rows):
                item = dict(row)
                item["acceptance"] = mapping
                item["captured_history_sha256"] = row["history_sha256"]
                if local_index >= accepted_count:
                    item["branch_status"] = "rejected"
                    item["committed_output_index"] = None
                    item["reconstructed_history_sha256"] = None
                    rejected_rows += 1
                    reconciled.append(item)
                    continue

                if local_index >= committed_count:
                    item["branch_status"] = "discarded_after_termination"
                    item["committed_output_index"] = None
                    item["reconstructed_history_sha256"] = None
                    discarded_rows += 1
                    reconciled.append(item)
                    continue

                output_index = cursor + local_index
                history = prompt_tokens + output_tokens[:output_index]
                reconstructed_hash = token_history_sha256(history)
                expected_input_position = len(history) - 1
                expected_prediction_position = len(history)
                expected_input_token = history[-1]
                mapping_ok = (
                    int(row["input_position"]) == expected_input_position
                    and int(row["prediction_position"]) == expected_prediction_position
                    and int(row["input_token_id"]) == expected_input_token
                    and int(row["history_length"]) == len(history)
                )
                if not mapping_ok:
                    input_mapping_mismatches += 1
                    errors.append(
                        {
                            "kind": "input_mapping_mismatch",
                            "prompt_id": prompt_id,
                            "request_step": step,
                            "row": local_index,
                            "observed": {
                                "input_position": row["input_position"],
                                "prediction_position": row["prediction_position"],
                                "input_token_id": row["input_token_id"],
                                "history_length": row["history_length"],
                            },
                            "expected": {
                                "input_position": expected_input_position,
                                "prediction_position": expected_prediction_position,
                                "input_token_id": expected_input_token,
                                "history_length": len(history),
                            },
                        }
                    )
                if row["history_sha256"] != reconstructed_hash:
                    captured_history_hash_mismatches += 1

                item["branch_status"] = (
                    "committed_spec" if int(row.get("num_draft_tokens", 0)) else "committed_ar"
                )
                item["committed_output_index"] = output_index
                item["reconstructed_history_sha256"] = reconstructed_hash
                item["history_sha256"] = reconstructed_hash
                item["reconstructed_mapping_ok"] = mapping_ok
                item["expected_output_token_id"] = output_tokens[output_index]
                accepted_rows += 1
                reconciled.append(item)
            cursor += committed_count

        if cursor != len(output_tokens):
            errors.append(
                {
                    "kind": "output_length_mismatch",
                    "prompt_id": prompt_id,
                    "captured_committed_tokens": cursor,
                    "generation_tokens": len(output_tokens),
                }
            )

    validation = {
        "run_id": run_id,
        "capture_rows": len(capture_rows),
        "acceptance_rows": len(acceptance),
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "discarded_after_termination_rows": discarded_rows,
        "sampled_token_mismatch_batches": sampled_token_mismatches,
        "input_mapping_mismatch_rows": input_mapping_mismatches,
        "captured_history_hash_mismatch_rows": captured_history_hash_mismatches,
        "captured_history_hash_mismatch_rate": (
            captured_history_hash_mismatches / accepted_rows if accepted_rows else None
        ),
        "control_mapping_passed": not any(
            error["kind"] in {"sampled_token_mismatch", "input_mapping_mismatch", "output_length_mismatch"}
            for error in errors
        ),
        "errors": errors,
    }
    return reconciled, validation
