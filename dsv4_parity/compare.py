from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .alignment import shared_prefix
from .capture_compare import compare_capture_pair
from .io import atomic_write_json, read_json, sha256_json


def compare_generation_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {row["prompt_id"]: row for row in left["prompts"]}
    right_rows = {row["prompt_id"]: row for row in right["prompts"]}
    if left_rows.keys() != right_rows.keys():
        raise ValueError("prompt sets differ")
    rows = []
    for prompt_id in sorted(left_rows):
        a = left_rows[prompt_id]
        b = right_rows[prompt_id]
        if a["prompt_token_ids"] != b["prompt_token_ids"]:
            raise ValueError(f"encoded prompt differs for {prompt_id}")
        prefix = shared_prefix(a["token_ids"], b["token_ids"])
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": a["category"],
                "left_length": len(a["token_ids"]),
                "right_length": len(b["token_ids"]),
                "shared_length": prefix.shared_length,
                "first_divergence": prefix.first_divergence,
                "termination": prefix.termination,
                "exact_match": prefix.termination == "identical",
                "left_finish_reason": a.get("finish_reason"),
                "right_finish_reason": b.get("finish_reason"),
            }
        )
    diverged = sum(not row["exact_match"] for row in rows)
    return {
        "left_run_id": left["run_id"],
        "right_run_id": right["run_id"],
        "left_mode": left["mode"],
        "right_mode": right["mode"],
        "num_prompts": len(rows),
        "exact_match_count": len(rows) - diverged,
        "divergence_count": diverged,
        "sequence_exact_match_rate": (len(rows) - diverged) / len(rows) if rows else None,
        "prompts": rows,
    }


def compare_generation_matrix(paths: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    runs = [read_json(path) for path in paths]
    ids = [run["run_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("run IDs must be unique")
    pair_results = [compare_generation_pair(a, b) for a, b in itertools.combinations(runs, 2)]
    result = {
        "schema_version": 1,
        "kind": "generation_comparison",
        "run_ids": ids,
        "pairs": pair_results,
    }
    result["artifact_sha256"] = sha256_json(result)
    atomic_write_json(output_path, result)
    return result


def add_capture_comparisons(
    comparison: dict[str, Any],
    capture_dirs: dict[str, str | Path],
    generation_artifacts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    runs = {run_id: directory for run_id, directory in capture_dirs.items()}
    expected = set(comparison["run_ids"])
    if runs.keys() != expected:
        raise ValueError(f"capture run IDs must be {sorted(expected)}, got {sorted(runs)}")
    if generation_artifacts is not None and set(generation_artifacts) != expected:
        raise ValueError("generation artifact IDs must match comparison run IDs")
    comparison["capture_pairs"] = [
        compare_capture_pair(
            left,
            runs[left],
            right,
            runs[right],
            None if generation_artifacts is None else generation_artifacts[left],
            None if generation_artifacts is None else generation_artifacts[right],
        )
        for left, right in itertools.combinations(comparison["run_ids"], 2)
    ]
    comparison["artifact_sha256"] = sha256_json(
        {key: value for key, value in comparison.items() if key != "artifact_sha256"}
    )
    return comparison
