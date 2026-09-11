"""Minimal target-logits instrumentation imported by the patched vLLM runner.

The hook is deliberately outside vLLM. The runner patch contains only two
calls and one graph-mode assignment, making backup/restore verification simple.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()
_REQUEST_STEPS: dict[str, int] = {}


def _enabled() -> bool:
    return os.environ.get("DSV4_PARITY_CAPTURE") == "1"


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def normalize_parity_request_id(engine_request_id: str) -> str | None:
    marker = engine_request_id.find("parity:")
    if marker < 0:
        return None
    value = engine_request_id[marker:]
    value = re.sub(r"-\d+-[0-9a-fA-F]{8}$", "", value)
    head, separator, tail = value.rpartition("-")
    if separator and tail.isdigit():
        value = head
    return value


def _to_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    try:
        return [int(item) for item in value]
    except TypeError:
        return [int(value)]


def _history_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


def _modular_draft_count(input_batch: Any, request_index: int) -> int:
    values = input_batch.num_draft_tokens_per_req
    return 0 if values is None else int(values[request_index])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def _write_status(directory: Path, *, complete: bool, reason: str | None = None) -> None:
    temporary = directory / ".capture_status.tmp"
    destination = directory / "capture_status.json"
    temporary.write_text(
        json.dumps(
            {
                "complete": complete,
                "reason": reason,
                "raw_bytes": (directory / "raw_logits.bin").stat().st_size
                if (directory / "raw_logits.bin").exists()
                else 0,
                "updated_unix_ns": time.time_ns(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def maybe_capture_target_logits(runner: Any, logits: torch.Tensor, input_batch: Any) -> None:
    if not _enabled() or _rank() != 0 or logits is None:
        return
    req_ids = list(input_batch.req_ids[: input_batch.num_reqs])
    if not any(normalize_parity_request_id(str(req_id)) is not None for req_id in req_ids):
        return

    directory = Path(os.environ["DSV4_PARITY_CAPTURE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("DSV4_PARITY_CAPTURE_LIMIT_BYTES", str(32 << 30)))
    cu = [int(value) for value in input_batch.cu_num_logits_np[: input_batch.num_reqs + 1]]
    selected_indices = input_batch.logits_indices[: cu[-1]].long()
    positions = _to_list(input_batch.positions[selected_indices])
    input_tokens = _to_list(input_batch.input_ids[selected_indices])
    graph_mode = str(getattr(runner, "_dsv4_parity_graph_mode", "unknown"))
    batch_id = uuid.uuid4().hex
    setattr(runner, "_dsv4_parity_capture_batch_id", batch_id)

    with _LOCK:
        raw_path = directory / "raw_logits.bin"
        metadata_path = directory / "capture.jsonl"
        current_size = raw_path.stat().st_size if raw_path.exists() else 0
        raw_bytes = logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        if current_size + len(raw_bytes) > limit:
            _write_status(directory, complete=False, reason="capture_limit_exceeded")
            os.environ["DSV4_PARITY_CAPTURE"] = "0"
            return
        with raw_path.open("ab") as raw_handle:
            raw_handle.write(raw_bytes)
            raw_handle.flush()
        element_size = logits.element_size()
        row_size = logits.shape[-1] * element_size

        for request_index, req_id_value in enumerate(req_ids):
            engine_req_id = str(req_id_value)
            req_id = normalize_parity_request_id(engine_req_id)
            if req_id is None:
                continue
            row_start, row_end = cu[request_index], cu[request_index + 1]
            if row_start == row_end:
                continue
            step = _REQUEST_STEPS.get(engine_req_id, 0)
            _REQUEST_STEPS[engine_req_id] = step + 1
            request_state_index = runner.req_states.req_id_to_index[engine_req_id]
            max_position = max(positions[row_start:row_end])
            history_buffer = runner.req_states.all_token_ids.gpu[
                request_state_index, : max_position + 1
            ]
            history_tokens = _to_list(history_buffer)
            num_scheduled = int(input_batch.num_scheduled_tokens[request_index])
            num_draft = _modular_draft_count(input_batch, request_index)
            for local_row, global_row in enumerate(range(row_start, row_end)):
                input_position = positions[global_row]
                history = history_tokens[: input_position + 1]
                vector = logits[global_row].detach().float()
                k = min(20, int(vector.numel()))
                top_values, top_indices = torch.topk(vector, k=k, dim=-1)
                row = {
                    "schema_version": 1,
                    "run_id": os.environ.get("DSV4_PARITY_RUN_ID"),
                    "mode": os.environ.get("DSV4_SPEC_MODE"),
                    "batch_id": batch_id,
                    "request_id": req_id,
                    "engine_request_id": engine_req_id,
                    "request_step": step,
                    "request_index": request_index,
                    "logits_row_within_request": local_row,
                    "global_logits_row": global_row,
                    "input_token_id": input_tokens[global_row],
                    "input_position": input_position,
                    "prediction_position": input_position + 1,
                    "history_length": len(history),
                    "history_sha256": _history_sha256(history),
                    "num_scheduled_tokens": num_scheduled,
                    "num_draft_tokens": num_draft,
                    "target_query_length": row_end - row_start,
                    "num_tokens": int(input_batch.num_tokens),
                    "num_tokens_after_padding": int(input_batch.num_tokens_after_padding),
                    "padding_tokens": int(input_batch.num_tokens_after_padding - input_batch.num_tokens),
                    "graph_mode": graph_mode,
                    "raw_file": raw_path.name,
                    "raw_offset_bytes": current_size + global_row * row_size,
                    "raw_length_bytes": row_size,
                    "raw_dtype": str(logits.dtype).removeprefix("torch."),
                    "vocab_size": int(logits.shape[-1]),
                    "top20_token_ids": [int(value) for value in top_indices.cpu().tolist()],
                    "top20_logits": [float(value) for value in top_values.cpu().tolist()],
                    "nonfinite_count": int((~torch.isfinite(vector)).sum().item()),
                    "branch_status": "pending_acceptance" if num_draft else "committed_ar",
                    "captured_unix_ns": time.time_ns(),
                }
                _append_jsonl(metadata_path, row)
        _write_status(directory, complete=True)


def maybe_capture_target_logits_legacy(
    runner: Any,
    logits: torch.Tensor | None,
    scheduler_output: Any,
    logits_indices: torch.Tensor,
    positions_tensor: torch.Tensor,
    cudagraph_mode: Any,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    spec_decode_metadata: Any,
) -> None:
    """Capture adapter for `vllm.v1.worker.gpu_model_runner.GPUModelRunner`.

    This image contains both the newer modular runner and this legacy-layout
    runner. The deployment uses the latter; its token source of truth is the
    persistent CPU `InputBatch.token_ids_cpu` table.
    """

    if not _enabled() or _rank() != 0 or logits is None:
        return
    input_batch = runner.input_batch
    num_reqs = int(input_batch.num_reqs)
    engine_req_ids = list(input_batch.req_ids[:num_reqs])
    normalized_ids = [normalize_parity_request_id(str(value)) for value in engine_req_ids]
    if not any(value is not None for value in normalized_ids):
        return

    if spec_decode_metadata is None:
        row_counts = [1] * num_reqs
        draft_counts = [0] * num_reqs
    else:
        draft_counts = [int(value) for value in spec_decode_metadata.num_draft_tokens]
        row_counts = [value + 1 for value in draft_counts]
    cu = [0]
    for count in row_counts:
        cu.append(cu[-1] + count)
    if cu[-1] != int(logits.shape[0]):
        raise RuntimeError(f"logits row mapping mismatch: {cu[-1]} != {logits.shape[0]}")

    selected_indices = logits_indices[: cu[-1]].long()
    positions = _to_list(positions_tensor[selected_indices])
    input_tokens = _to_list(runner.input_ids.gpu[selected_indices])
    directory = Path(os.environ["DSV4_PARITY_CAPTURE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get("DSV4_PARITY_CAPTURE_LIMIT_BYTES", str(32 << 30)))
    batch_id = uuid.uuid4().hex
    setattr(runner, "_dsv4_parity_capture_batch_id", batch_id)

    with _LOCK:
        raw_path = directory / "raw_logits.bin"
        metadata_path = directory / "capture.jsonl"
        current_size = raw_path.stat().st_size if raw_path.exists() else 0
        raw_bytes_value = logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        if current_size + len(raw_bytes_value) > limit:
            _write_status(directory, complete=False, reason="capture_limit_exceeded")
            os.environ["DSV4_PARITY_CAPTURE"] = "0"
            return
        with raw_path.open("ab") as raw_handle:
            raw_handle.write(raw_bytes_value)
            raw_handle.flush()
        row_size = int(logits.shape[-1]) * logits.element_size()

        for request_index, (engine_req_id_value, req_id) in enumerate(
            zip(engine_req_ids, normalized_ids)
        ):
            if req_id is None:
                continue
            engine_req_id = str(engine_req_id_value)
            row_start, row_end = cu[request_index], cu[request_index + 1]
            step = _REQUEST_STEPS.get(engine_req_id, 0)
            _REQUEST_STEPS[engine_req_id] = step + 1
            req_index = input_batch.req_id_to_index[engine_req_id]
            max_position = max(positions[row_start:row_end])
            history_tokens = [
                int(value)
                for value in input_batch.token_ids_cpu[req_index, : max_position + 1].tolist()
            ]
            num_scheduled = int(scheduler_output.num_scheduled_tokens[engine_req_id])
            num_draft = draft_counts[request_index]
            for local_row, global_row in enumerate(range(row_start, row_end)):
                input_position = positions[global_row]
                history = history_tokens[: input_position + 1]
                vector = logits[global_row].detach().float()
                k = min(20, int(vector.numel()))
                top_values, top_indices = torch.topk(vector, k=k, dim=-1)
                _append_jsonl(
                    metadata_path,
                    {
                        "schema_version": 1,
                        "runner_layout": "v1.worker.gpu_model_runner",
                        "run_id": os.environ.get("DSV4_PARITY_RUN_ID"),
                        "mode": os.environ.get("DSV4_SPEC_MODE"),
                        "batch_id": batch_id,
                        "request_id": req_id,
                        "engine_request_id": engine_req_id,
                        "request_step": step,
                        "request_index": request_index,
                        "logits_row_within_request": local_row,
                        "global_logits_row": global_row,
                        "input_token_id": input_tokens[global_row],
                        "input_position": input_position,
                        "prediction_position": input_position + 1,
                        "history_length": len(history),
                        "history_sha256": _history_sha256(history),
                        "num_scheduled_tokens": num_scheduled,
                        "num_draft_tokens": num_draft,
                        "target_query_length": row_counts[request_index],
                        "num_tokens": int(num_tokens_unpadded),
                        "num_tokens_after_padding": int(num_tokens_padded),
                        "padding_tokens": int(num_tokens_padded - num_tokens_unpadded),
                        "graph_mode": str(cudagraph_mode),
                        "raw_file": raw_path.name,
                        "raw_offset_bytes": current_size + global_row * row_size,
                        "raw_length_bytes": row_size,
                        "raw_dtype": str(logits.dtype).removeprefix("torch."),
                        "vocab_size": int(logits.shape[-1]),
                        "top20_token_ids": [int(value) for value in top_indices.cpu().tolist()],
                        "top20_logits": [float(value) for value in top_values.cpu().tolist()],
                        "nonfinite_count": int((~torch.isfinite(vector)).sum().item()),
                        "branch_status": "pending_acceptance" if num_draft else "committed_ar",
                        "captured_unix_ns": time.time_ns(),
                    },
                )
        _write_status(directory, complete=True)


def maybe_capture_acceptance_legacy(runner: Any, sampler_output: Any) -> None:
    maybe_capture_acceptance(runner, sampler_output, runner.input_batch)


def maybe_capture_acceptance(runner: Any, sampler_output: Any, input_batch: Any) -> None:
    if not _enabled() or _rank() != 0:
        return
    batch_id = getattr(runner, "_dsv4_parity_capture_batch_id", None)
    if batch_id is None:
        return
    req_ids = list(input_batch.req_ids[: input_batch.num_reqs])
    if not any(normalize_parity_request_id(str(req_id)) is not None for req_id in req_ids):
        return
    directory = Path(os.environ["DSV4_PARITY_CAPTURE_DIR"])
    num_sampled = _to_list(getattr(sampler_output, "num_sampled", []))
    num_rejected = _to_list(getattr(sampler_output, "num_rejected", []))
    sampled = getattr(sampler_output, "sampled_token_ids", None)
    sampled_rows = sampled.detach().cpu().tolist() if isinstance(sampled, torch.Tensor) else None
    with _LOCK:
        for index, req_id_value in enumerate(req_ids):
            engine_req_id = str(req_id_value)
            req_id = normalize_parity_request_id(engine_req_id)
            if req_id is None:
                continue
            valid_count = num_sampled[index] if index < len(num_sampled) else None
            sampled_tokens = None
            if sampled_rows is not None and index < len(sampled_rows):
                sampled_tokens = sampled_rows[index]
                if valid_count is not None:
                    sampled_tokens = sampled_tokens[:valid_count]
            _append_jsonl(
                directory / "acceptance.jsonl",
                {
                    "schema_version": 1,
                    "run_id": os.environ.get("DSV4_PARITY_RUN_ID"),
                    "batch_id": batch_id,
                    "request_id": req_id,
                    "engine_request_id": engine_req_id,
                    "num_sampled": valid_count,
                    "num_rejected": num_rejected[index] if index < len(num_rejected) else None,
                    "sampled_token_ids": sampled_tokens,
                    "captured_unix_ns": time.time_ns(),
                },
            )
