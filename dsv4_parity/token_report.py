from __future__ import annotations

from typing import Any

from .http_client import VllmClient


def build_token_divergence_report(
    comparison: dict[str, Any],
    generations: dict[str, dict[str, Any]],
    client: VllmClient,
    *,
    left_run_id: str = "A1",
    right_run_id: str = "S1",
) -> dict[str, Any]:
    pair = next(
        item
        for item in comparison["pairs"]
        if {item["left_run_id"], item["right_run_id"]} == {left_run_id, right_run_id}
    )
    pair_rows = {row["prompt_id"]: row for row in pair["prompts"]}
    left_rows = {
        row["prompt_id"]: row for row in generations[left_run_id]["prompts"]
    }
    right_rows = {
        row["prompt_id"]: row for row in generations[right_run_id]["prompts"]
    }
    rows = []
    for prompt_id, boundary in sorted(pair_rows.items()):
        divergence = boundary["first_divergence"]
        if divergence is None:
            continue
        left_tokens = [int(value) for value in left_rows[prompt_id]["token_ids"]]
        right_tokens = [int(value) for value in right_rows[prompt_id]["token_ids"]]
        start = max(0, int(divergence) - 6)
        end = int(divergence) + 7
        left_slice = left_tokens[start:end]
        right_slice = right_tokens[start:end]
        left_token = left_tokens[divergence] if divergence < len(left_tokens) else None
        right_token = right_tokens[divergence] if divergence < len(right_tokens) else None
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": boundary["category"],
                "first_divergence": divergence,
                "termination": boundary["termination"],
                "left_token_id": left_token,
                "right_token_id": right_token,
                "left_token_text": None
                if left_token is None
                else client.detokenize([left_token]),
                "right_token_text": None
                if right_token is None
                else client.detokenize([right_token]),
                "context_start": start,
                "left_context_token_ids": left_slice,
                "right_context_token_ids": right_slice,
                "left_context_text": client.detokenize(left_slice),
                "right_context_text": client.detokenize(right_slice),
                "left_finish_reason": left_rows[prompt_id].get("finish_reason"),
                "right_finish_reason": right_rows[prompt_id].get("finish_reason"),
            }
        )
    return {
        "schema_version": 1,
        "kind": "token_divergence_report",
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "rows": rows,
    }
