"""In-process fixed-state q_len=1/7/8 replay for the modular vLLM runner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_LOCK = threading.Lock()
_MANIFEST: dict[str, dict[str, Any]] | None = None
_FORCED_PREFILL: set[str] = set()
_COMPLETED: set[str] = set()


def _normalize_request_id(engine_request_id: str) -> str | None:
    marker = engine_request_id.find("parity:REPLAY:")
    if marker < 0:
        return None
    value = engine_request_id[marker:]
    return re.sub(r"-\d+-[0-9a-fA-F]{8}$", "", value)


def _manifest() -> dict[str, dict[str, Any]]:
    global _MANIFEST
    if _MANIFEST is None:
        path = os.environ.get("DSV4_REPLAY_MANIFEST")
        if not path:
            _MANIFEST = {}
        else:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            _MANIFEST = {row["request_id"]: row for row in document["windows"]}
    return _MANIFEST


def _plan_for_engine_id(engine_request_id: str) -> dict[str, Any] | None:
    normalized = _normalize_request_id(engine_request_id)
    return None if normalized is None else _manifest().get(normalized)


def _only_replay_request(input_batch: Any) -> tuple[str, dict[str, Any]] | None:
    matches = []
    for req_id in input_batch.req_ids[: input_batch.num_reqs]:
        plan = _plan_for_engine_id(str(req_id))
        if plan is not None:
            matches.append((str(req_id), plan))
    if not matches:
        return None
    if len(matches) != 1 or int(input_batch.num_reqs) != 1:
        raise RuntimeError("fixed replay requires exactly one active request")
    return matches[0]


def maybe_force_replay_sample(runner: Any, sampler_output: Any, input_batch: Any) -> None:
    """Force the first post-prefill token so the next target step has fixed input."""

    matched = _only_replay_request(input_batch)
    if matched is None:
        return
    engine_req_id, plan = matched
    if engine_req_id in _FORCED_PREFILL or engine_req_id in _COMPLETED:
        return
    if int(input_batch.num_draft_tokens) != 0:
        return
    sampler_output.sampled_token_ids[0, 0] = int(plan["fixed_input_token_ids"][0])
    _FORCED_PREFILL.add(engine_req_id)


def maybe_force_replay_drafts(
    runner: Any, input_batch: Any, draft_tokens: torch.Tensor
) -> torch.Tensor:
    """Replace the setup proposal with seven fixed A1 trajectory tokens."""

    matched = _only_replay_request(input_batch)
    if matched is None:
        return draft_tokens
    engine_req_id, plan = matched
    if engine_req_id in _COMPLETED:
        return draft_tokens
    fixed = torch.as_tensor(
        plan["fixed_input_token_ids"][1:8],
        dtype=draft_tokens.dtype,
        device=draft_tokens.device,
    )
    if draft_tokens.ndim != 2 or draft_tokens.shape[1] < 7:
        raise RuntimeError(f"unexpected draft tensor shape: {tuple(draft_tokens.shape)}")
    draft_tokens[0, :7] = fixed
    return draft_tokens


@dataclass
class _CacheSlice:
    tensor: torch.Tensor
    axis: int
    indices: torch.Tensor
    value: torch.Tensor
    identity: dict[str, Any]


@dataclass
class _StateSnapshot:
    req_index: int
    token_end: int
    request_tensors: dict[str, torch.Tensor]
    request_scalars: dict[str, Any]
    rng_cpu: torch.Tensor
    rng_cuda: torch.Tensor
    cache_slices: list[_CacheSlice]


def _cache_axis_and_ratio(tensor: torch.Tensor, num_blocks: int) -> tuple[int, int]:
    exact = [axis for axis, size in enumerate(tensor.shape) if int(size) == num_blocks]
    if exact:
        return exact[0], 1
    candidates = [
        (int(size) // num_blocks, axis)
        for axis, size in enumerate(tensor.shape)
        if int(size) > num_blocks and int(size) % num_blocks == 0
    ]
    if not candidates:
        raise RuntimeError(
            f"cannot identify block axis for cache shape {tuple(tensor.shape)} "
            f"and num_blocks={num_blocks}"
        )
    ratio, axis = min(candidates)
    return axis, ratio


def _snapshot_cache_blocks(
    runner: Any, block_tables: tuple[torch.Tensor, ...], base_tokens: int, width: int
) -> list[_CacheSlice]:
    logical_ids: set[int] = set()
    for group_index, table in enumerate(block_tables):
        spec = runner.kv_cache_config.kv_cache_groups[group_index].kv_cache_spec
        global_block_size = int(spec.block_size) * int(runner.dcp_size)
        count = (base_tokens + width + global_block_size - 1) // global_block_size
        values = table[0, :count].detach().cpu().tolist()
        logical_ids.update(int(value) for value in values if int(value) >= 0)
    if not logical_ids:
        raise RuntimeError("no allocated cache blocks found for replay request")

    result: list[_CacheSlice] = []
    seen: set[tuple[Any, ...]] = set()
    num_blocks = int(runner.kv_cache_config.num_blocks)
    for ordinal, value in enumerate(runner.kv_caches):
        tensors = value if isinstance(value, (list, tuple)) else [value]
        for component, tensor in enumerate(tensors):
            axis, ratio = _cache_axis_and_ratio(tensor, num_blocks)
            expanded = [
                logical_id * ratio + offset
                for logical_id in sorted(logical_ids)
                for offset in range(ratio)
            ]
            indices = torch.as_tensor(expanded, dtype=torch.int64, device=tensor.device)
            identity_key = (
                int(tensor.data_ptr()),
                int(tensor.storage_offset()),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            if identity_key in seen:
                continue
            seen.add(identity_key)
            result.append(
                _CacheSlice(
                    tensor=tensor,
                    axis=axis,
                    indices=indices,
                    value=torch.index_select(tensor, axis, indices).clone(),
                    identity={
                        "ordinal": ordinal,
                        "component": component,
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "data_ptr": int(tensor.data_ptr()),
                        "storage_offset": int(tensor.storage_offset()),
                        "block_axis": axis,
                        "kernel_blocks_per_logical_block": ratio,
                        "logical_block_ids": sorted(logical_ids),
                    },
                )
            )
    return result


def _snapshot_state(
    runner: Any,
    engine_req_id: str,
    block_tables: tuple[torch.Tensor, ...],
    width: int,
) -> _StateSnapshot:
    req_index = int(runner.req_states.req_id_to_index[engine_req_id])
    base_tokens = int(runner.req_states.num_computed_tokens.gpu[req_index].item())
    token_end = base_tokens + width + 2
    request_tensors = {
        "num_computed_tokens": runner.req_states.num_computed_tokens.gpu[req_index].clone(),
        "total_len": runner.req_states.total_len.gpu[req_index].clone(),
        "last_sampled_tokens": runner.req_states.last_sampled_tokens[req_index].clone(),
        "draft_tokens": runner.req_states.draft_tokens[req_index].clone(),
        "next_prefill_tokens": runner.req_states.next_prefill_tokens[req_index].clone(),
        "all_token_ids": runner.req_states.all_token_ids.gpu[req_index, :token_end].clone(),
    }
    request_scalars = {
        "num_computed_tokens_np": int(runner.req_states.num_computed_tokens_np[req_index]),
        "num_computed_prefill_tokens": int(
            runner.req_states.num_computed_prefill_tokens[req_index]
        ),
        "max_seq_len": int(runner.req_states.max_seq_len[req_index]),
        "prompt_len": int(runner.req_states.prompt_len.np[req_index]),
        "prefill_len": int(runner.req_states.prefill_len.np[req_index]),
    }
    return _StateSnapshot(
        req_index=req_index,
        token_end=token_end,
        request_tensors=request_tensors,
        request_scalars=request_scalars,
        rng_cpu=torch.random.get_rng_state().clone(),
        rng_cuda=torch.cuda.get_rng_state(runner.device).clone(),
        cache_slices=_snapshot_cache_blocks(runner, block_tables, base_tokens, width),
    )


def _restore_state(runner: Any, snapshot: _StateSnapshot) -> None:
    index = snapshot.req_index
    state = runner.req_states
    state.num_computed_tokens.gpu[index].copy_(snapshot.request_tensors["num_computed_tokens"])
    state.total_len.gpu[index].copy_(snapshot.request_tensors["total_len"])
    state.last_sampled_tokens[index].copy_(snapshot.request_tensors["last_sampled_tokens"])
    state.draft_tokens[index].copy_(snapshot.request_tensors["draft_tokens"])
    state.next_prefill_tokens[index].copy_(snapshot.request_tensors["next_prefill_tokens"])
    state.all_token_ids.gpu[index, : snapshot.token_end].copy_(
        snapshot.request_tensors["all_token_ids"]
    )
    state.num_computed_tokens_np[index] = snapshot.request_scalars["num_computed_tokens_np"]
    state.num_computed_prefill_tokens[index] = snapshot.request_scalars[
        "num_computed_prefill_tokens"
    ]
    state.max_seq_len[index] = snapshot.request_scalars["max_seq_len"]
    state.prompt_len.np[index] = snapshot.request_scalars["prompt_len"]
    state.prefill_len.np[index] = snapshot.request_scalars["prefill_len"]
    for cache in snapshot.cache_slices:
        cache.tensor.index_copy_(cache.axis, cache.indices, cache.value)
    torch.random.set_rng_state(snapshot.rng_cpu)
    torch.cuda.set_rng_state(snapshot.rng_cuda, runner.device)
    torch.cuda.synchronize(runner.device)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )


def _state_fingerprint(runner: Any, snapshot: _StateSnapshot) -> str:
    digest = hashlib.sha256()
    state = runner.req_states
    index = snapshot.req_index
    for tensor in (
        state.num_computed_tokens.gpu[index],
        state.total_len.gpu[index],
        state.last_sampled_tokens[index],
        state.draft_tokens[index],
        state.next_prefill_tokens[index],
        state.all_token_ids.gpu[index, : snapshot.token_end],
    ):
        digest.update(_tensor_bytes(tensor))
    digest.update(struct.pack("<q", int(state.num_computed_tokens_np[index])))
    digest.update(struct.pack("<q", int(state.num_computed_prefill_tokens[index])))
    for cache in snapshot.cache_slices:
        current = torch.index_select(cache.tensor, cache.axis, cache.indices)
        digest.update(_tensor_bytes(current))
    return digest.hexdigest()


def _rank_fingerprints(local: str) -> list[str]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        result: list[str | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(result, local)
        return [str(value) for value in result]
    return [local]


def _synthetic_scheduler(
    scheduler_output: Any, engine_req_id: str, fixed: list[int], query_len: int
) -> Any:
    result = copy.copy(scheduler_output)
    result.num_scheduled_tokens = {engine_req_id: query_len}
    result.total_num_scheduled_tokens = query_len
    result.scheduled_spec_decode_tokens = (
        {} if query_len == 1 else {engine_req_id: list(fixed[1:query_len])}
    )
    result.scheduled_encoder_inputs = {}
    result.finished_req_ids = set()
    result.new_block_ids_to_zero = []
    return result


def _forward_variant(
    runner: Any,
    scheduler_output: Any,
    engine_req_id: str,
    fixed: list[int],
    query_len: int,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    req_index = int(runner.req_states.req_id_to_index[engine_req_id])
    runner.req_states.draft_tokens[req_index].zero_()
    if query_len > 1:
        runner.req_states.draft_tokens[req_index, : query_len - 1] = torch.as_tensor(
            fixed[1:query_len],
            dtype=runner.req_states.draft_tokens.dtype,
            device=runner.device,
        )
    synthetic = _synthetic_scheduler(scheduler_output, engine_req_id, fixed, query_len)
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=query_len,
        num_reqs=None,
        uniform_token_count=query_len,
        num_active_loras=0,
    )
    input_batch = runner.prepare_inputs(synthetic, descriptor)
    input_ids = [int(value) for value in input_batch.input_ids[:query_len].detach().cpu()]
    positions = [int(value) for value in input_batch.positions[:query_len].detach().cpu()]
    if input_ids != fixed[:query_len]:
        raise RuntimeError(
            f"fixed replay input mismatch q{query_len}: {input_ids} != {fixed[:query_len]}"
        )
    block_tables, slot_mappings = runner.prepare_attn(input_batch)
    runner.model_state.preprocess_state(
        input_batch,
        block_tables,
        runner.kv_cache_config,
        runner.req_states.num_computed_tokens.gpu,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, runner.kv_cache_config
    )
    attn_metadata = runner.model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables,
        slot_mappings,
        runner.attn_groups,
        runner.kv_cache_config,
    )
    model_inputs = {
        "input_ids": input_batch.input_ids,
        "positions": input_batch.positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
        **runner.model_state.prepare_inputs(input_batch, runner.req_states),
    }
    runner.eplb.prepare_forward(runner.model_config, input_batch.num_tokens)
    batch_descriptor = BatchDescriptor(
        num_tokens=input_batch.num_tokens_after_padding,
        has_lora=False,
        num_active_loras=0,
    )
    with set_forward_context(
        attn_metadata,
        runner.vllm_config,
        num_tokens=input_batch.num_tokens_after_padding,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        num_tokens_across_dp=None,
        batch_descriptor=batch_descriptor,
        slot_mapping=slot_mappings_by_layer,
        skip_compiled=True,
        is_padding=input_batch.is_padding,
    ):
        output = runner.model(**model_inputs)
    if runner.use_aux_hidden_state_outputs:
        hidden_states, _ = output
    else:
        hidden_states = output
    logits = runner.model.compute_logits(hidden_states[input_batch.logits_indices])
    torch.cuda.synchronize(runner.device)
    metadata = {
        "query_length": query_len,
        "input_token_ids": input_ids,
        "input_positions": positions,
        "prediction_positions": [value + 1 for value in positions],
        "num_tokens": int(input_batch.num_tokens),
        "num_tokens_after_padding": int(input_batch.num_tokens_after_padding),
        "padding_tokens": int(
            input_batch.num_tokens_after_padding - input_batch.num_tokens
        ),
        "graph_mode": "CUDAGraphMode.NONE",
        "skip_compiled": True,
        "draft_execution": False,
        "slot_mapping_shapes": {
            name: list(value.shape) for name, value in slot_mappings_by_layer.items()
        },
    }
    return logits, input_batch, metadata


def _advance_ar_state(runner: Any, input_batch: Any, next_token: int) -> None:
    sampled = torch.as_tensor([[next_token]], dtype=torch.int64, device=runner.device)
    num_sampled = torch.ones(1, dtype=torch.int32, device=runner.device)
    num_rejected = torch.zeros(1, dtype=torch.int32, device=runner.device)
    runner.postprocess_sampled(
        input_batch.idx_mapping,
        sampled,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
    )
    req_index = int(input_batch.idx_mapping_np[0])
    runner.req_states.num_computed_tokens_np[req_index] += 1
    torch.cuda.synchronize(runner.device)


def _append_logits(
    output_dir: Path,
    window_id: str,
    path_name: str,
    logits: torch.Tensor,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_logits.bin"
    payload = logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    limit = int(os.environ.get("DSV4_REPLAY_LIMIT_BYTES", str(32 << 30)))
    with _LOCK:
        offset = raw_path.stat().st_size if raw_path.exists() else 0
        if offset + len(payload) > limit:
            raise RuntimeError("fixed replay raw-logits limit exceeded")
        with raw_path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
    row_bytes = int(logits.shape[-1]) * logits.element_size()
    vector = logits.detach().float()
    records = []
    for row_index in range(int(logits.shape[0])):
        values, indices = torch.topk(vector[row_index], k=20)
        records.append(
            {
                "window_id": window_id,
                "path": path_name,
                "row": row_index,
                "input_token_id": metadata["input_token_ids"][row_index],
                "input_position": metadata["input_positions"][row_index],
                "prediction_position": metadata["prediction_positions"][row_index],
                "raw_file": raw_path.name,
                "raw_offset_bytes": offset + row_index * row_bytes,
                "raw_length_bytes": row_bytes,
                "raw_dtype": str(logits.dtype).removeprefix("torch."),
                "vocab_size": int(logits.shape[-1]),
                "top20_token_ids": [int(value) for value in indices.cpu()],
                "top20_logits": [float(value) for value in values.cpu()],
                "nonfinite_count": int((~torch.isfinite(vector[row_index])).sum().item()),
                **{
                    key: value
                    for key, value in metadata.items()
                    if key not in {"input_token_ids", "input_positions", "prediction_positions"}
                },
            }
        )
    return records


def _write_window_result(output_dir: Path, document: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{document['window_id']}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def maybe_run_fixed_replay(
    runner: Any,
    scheduler_output: Any,
    batch_desc: Any,
    input_batch: Any,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor,
) -> None:
    """Run AR/Q7/Q8 branches before the real q8 verification forward."""

    matched = _only_replay_request(input_batch)
    if matched is None:
        return
    engine_req_id, plan = matched
    if engine_req_id in _COMPLETED or int(input_batch.num_draft_tokens) != 7:
        return
    from vllm.config.compilation import CUDAGraphMode

    if batch_desc.cg_mode != CUDAGraphMode.NONE:
        raise RuntimeError(f"fixed replay requires eager mode, got {batch_desc.cg_mode}")
    fixed = [int(value) for value in plan["fixed_input_token_ids"]]
    if len(fixed) != 8:
        raise RuntimeError(f"fixed replay needs exactly 8 input tokens, got {len(fixed)}")
    actual_inputs = [int(value) for value in input_batch.input_ids[:8].detach().cpu()]
    if actual_inputs != fixed:
        raise RuntimeError(f"actual verification input mismatch: {actual_inputs} != {fixed}")

    snapshot = _snapshot_state(runner, engine_req_id, block_tables, width=8)
    _restore_state(runner, snapshot)
    initial_fingerprints = _rank_fingerprints(_state_fingerprint(runner, snapshot))
    output_dir = Path(os.environ["DSV4_REPLAY_OUTPUT_DIR"])
    records: list[dict[str, Any]] = []
    restores: list[dict[str, Any]] = []

    sequence = ["AR_1", "Q7_1", "Q8_1", "Q8_2", "Q7_2", "AR_2"]
    for path_name in sequence:
        _restore_state(runner, snapshot)
        before = _rank_fingerprints(_state_fingerprint(runner, snapshot))
        if before != initial_fingerprints:
            raise RuntimeError(f"state fingerprint mismatch before {path_name}")
        if path_name.startswith("AR"):
            for row_index in range(8):
                step_fixed = fixed[row_index:]
                logits, ar_batch, metadata = _forward_variant(
                    runner, scheduler_output, engine_req_id, step_fixed, 1
                )
                path_records = _append_logits(
                    output_dir, plan["window_id"], path_name, logits, metadata
                )
                for record in path_records:
                    record["row"] = row_index
                records.extend(path_records)
                if row_index < 7:
                    _advance_ar_state(runner, ar_batch, fixed[row_index + 1])
        else:
            query_len = 7 if path_name.startswith("Q7") else 8
            logits, _, metadata = _forward_variant(
                runner, scheduler_output, engine_req_id, fixed, query_len
            )
            records.extend(
                _append_logits(output_dir, plan["window_id"], path_name, logits, metadata)
            )
        _restore_state(runner, snapshot)
        after = _rank_fingerprints(_state_fingerprint(runner, snapshot))
        restores.append(
            {
                "path": path_name,
                "before_rank_fingerprints": before,
                "after_restore_rank_fingerprints": after,
                "restore_passed": after == initial_fingerprints,
            }
        )
        if after != initial_fingerprints:
            raise RuntimeError(f"state restore failed after {path_name}")

    mutated = list(fixed)
    vocab_size = int(getattr(runner.model_config, "vocab_size", 0) or runner.model_config.get_vocab_size())
    mutated[-1] = (mutated[-1] + 1) % vocab_size
    _restore_state(runner, snapshot)
    logits, _, metadata = _forward_variant(runner, scheduler_output, engine_req_id, mutated, 8)
    records.extend(
        _append_logits(output_dir, plan["window_id"], "CAUSAL_Q8_MUTATE_LAST", logits, metadata)
    )
    _restore_state(runner, snapshot)
    causal_restore = _rank_fingerprints(_state_fingerprint(runner, snapshot))
    if causal_restore != initial_fingerprints:
        raise RuntimeError("state restore failed after causal-mask control")

    runner.req_states.draft_tokens[snapshot.req_index, :7] = torch.as_tensor(
        fixed[1:8], dtype=runner.req_states.draft_tokens.dtype, device=runner.device
    )
    restored_batch = runner.prepare_inputs(scheduler_output, batch_desc)
    runner.prepare_attn(restored_batch)
    restored_inputs = [int(value) for value in restored_batch.input_ids[:8].detach().cpu()]
    if restored_inputs != fixed:
        raise RuntimeError("failed to restore outer q8 input buffers")

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        _write_window_result(
            output_dir,
            {
                "schema_version": 1,
                "kind": "fixed_state_replay_window",
                "window_id": plan["window_id"],
                "request_id": plan["request_id"],
                "prompt_id": plan["prompt_id"],
                "window_kind": plan["kind"],
                "output_start": plan["output_start"],
                "fixed_input_token_ids": fixed,
                "prefill_token_count": len(plan["prefill_token_ids"]),
                "sequence": sequence,
                "causal_control": {
                    "path": "CAUSAL_Q8_MUTATE_LAST",
                    "mutated_row": 7,
                    "original_token_id": fixed[-1],
                    "mutated_token_id": mutated[-1],
                },
                "initial_rank_fingerprints": initial_fingerprints,
                "restores": restores,
                "causal_restore_rank_fingerprints": causal_restore,
                "state_restore_passed": all(item["restore_passed"] for item in restores)
                and causal_restore == initial_fingerprints,
                "cache_slices": [item.identity for item in snapshot.cache_slices],
                "cache_addresses_stable": True,
                "control": {
                    "target_model_class": type(runner.model).__name__,
                    "model_state_class": type(runner.model_state).__name__,
                    "use_aux_hidden_state_outputs": bool(runner.use_aux_hidden_state_outputs),
                    "graph_mode": "CUDAGraphMode.NONE",
                    "torch_compile_skipped": True,
                    "draft_execution_inside_paths": False,
                    "actual_target_query_length": 8,
                    "runtime_draft_length": 7,
                },
                "rows": records,
                "completed_unix_ns": time.time_ns(),
            },
        )
    _COMPLETED.add(engine_req_id)
