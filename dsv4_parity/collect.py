from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path
from typing import Any

from .http_client import VllmClient
from .io import atomic_write_json, read_json, sha256_json
from .prompts import PROMPTS


def _extract_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"expected one completion choice, got: {response}")
    choice = choices[0]
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError("server omitted token_ids despite return_token_ids=true")
    return {
        "text": str(choice.get("text", "")),
        "token_ids": [int(value) for value in token_ids],
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
    }


def collect_run(
    *,
    endpoint: str,
    model: str,
    mode: str,
    run_id: str,
    output_path: str | Path,
    prompt_tokens_path: str | Path | None,
    max_tokens: int = 256,
    seed: int = 20260910,
    api_key: str = "EMPTY",
) -> dict[str, Any]:
    client = VllmClient(endpoint, api_key=api_key)
    model_response = client.wait_ready()
    registry: dict[str, list[int]] = {}
    if prompt_tokens_path is not None:
        stored = read_json(prompt_tokens_path)
        registry = {
            str(row["prompt_id"]): [int(value) for value in row["prompt_token_ids"]]
            for row in stored["prompts"]
        }

    results = []
    for prompt in PROMPTS:
        token_ids = registry.get(prompt.prompt_id)
        if token_ids is None:
            token_ids = client.tokenize(prompt.text)
        reset_result = client.reset_prefix_cache()
        if isinstance(reset_result, bool):
            reset_ok = reset_result
        elif isinstance(reset_result, dict):
            reset_ok = bool(reset_result.get("success", reset_result.get("reset", True)))
        else:
            reset_ok = False
        if not reset_ok:
            raise RuntimeError(f"prefix cache reset failed for {prompt.prompt_id}: {reset_result}")
        request_id = f"parity:{run_id}:{prompt.prompt_id}"
        response = client.complete(
            model=model,
            prompt_token_ids=token_ids,
            request_id=request_id,
            max_tokens=max_tokens,
            seed=seed,
        )
        choice = _extract_choice(response)
        row = prompt.to_dict()
        row.update(
            {
                "request_id": request_id,
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": sha256_json(token_ids),
                **choice,
            }
        )
        results.append(row)
        print(
            f"collected {run_id} {len(results):02d}/32 {prompt.prompt_id} "
            f"tokens={len(choice['token_ids'])} finish={choice['finish_reason']}",
            flush=True,
        )

    artifact = {
        "schema_version": 1,
        "kind": "generation_run",
        "run_id": run_id,
        "mode": mode,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host_platform": platform.platform(),
        "model": model,
        "server_models_response": model_response,
        "parameters": {
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "prefix_cache_reset_per_prompt": True,
        },
        "prompts": results,
    }
    artifact["artifact_sha256"] = sha256_json(artifact)
    atomic_write_json(output_path, artifact)
    return artifact
