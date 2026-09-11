from __future__ import annotations

from pathlib import Path
from typing import Any

from .http_client import VllmClient
from .io import atomic_write_json


def build_execution_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    windows = []
    skipped = []
    for prompt in plan["prompts"]:
        output = [int(value) for value in prompt["fixed_output_token_ids"]]
        prompt_tokens = [int(value) for value in prompt["prompt_token_ids"]]
        for window in prompt["windows"]:
            if window["status"] != "selected":
                skipped.append({"prompt_id": prompt["prompt_id"], "window": window})
                continue
            start = int(window["output_start"])
            window_id = f"{prompt['prompt_id']}__{window['kind']}__{start}"
            fixed = output[start : start + 8]
            if len(fixed) != 8:
                skipped.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "window": window,
                        "reason": "fixed_input_shorter_than_8",
                    }
                )
                continue
            windows.append(
                {
                    "window_id": window_id,
                    "request_id": f"parity:REPLAY:{window_id}",
                    "prompt_id": prompt["prompt_id"],
                    "kind": window["kind"],
                    "output_start": start,
                    "absolute_input_start": window["absolute_input_start"],
                    "prefill_token_ids": prompt_tokens + output[:start],
                    "fixed_input_token_ids": fixed,
                }
            )
    return {
        "schema_version": 1,
        "kind": "fixed_replay_execution_manifest",
        "source_run_id": plan["source_run_id"],
        "sequence": plan["sequence"],
        "windows": windows,
        "skipped": skipped,
    }


def collect_replay_requests(
    endpoint: str,
    model: str,
    manifest: dict[str, Any],
    output_path: str | Path,
    limit: int | None = None,
    start: int = 0,
) -> dict[str, Any]:
    client = VllmClient(endpoint)
    client.wait_ready(deadline_seconds=1800)
    results = []
    selected_windows = manifest["windows"][start:]
    if limit is not None:
        selected_windows = selected_windows[:limit]
    for window in selected_windows:
        reset = client.reset_prefix_cache()
        response = client.complete(
            model=model,
            prompt_token_ids=window["prefill_token_ids"],
            request_id=window["request_id"],
            max_tokens=2,
            seed=20260910,
        )
        choice = response["choices"][0]
        token_ids = [int(value) for value in choice["token_ids"]]
        first_token_forced = bool(
            token_ids and token_ids[0] == window["fixed_input_token_ids"][0]
        )
        results.append(
            {
                "window_id": window["window_id"],
                "request_id": window["request_id"],
                "prefix_reset_response": reset,
                "response_token_ids": token_ids,
                "first_token_forced": first_token_forced,
                "finish_reason": choice.get("finish_reason"),
            }
        )
        if not first_token_forced:
            raise RuntimeError(f"failed to force first replay token for {window['window_id']}")
    artifact = {
        "schema_version": 1,
        "kind": "fixed_replay_request_run",
        "windows_requested": len(selected_windows),
        "start_index": start,
        "all_first_tokens_forced": all(row["first_token_forced"] for row in results),
        "results": results,
    }
    atomic_write_json(output_path, artifact)
    return artifact
