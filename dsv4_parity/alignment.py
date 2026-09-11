from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SharedPrefix:
    shared_length: int
    first_divergence: int | None
    termination: str


def shared_prefix(left: Sequence[int], right: Sequence[int]) -> SharedPrefix:
    shared = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            return SharedPrefix(shared, shared, "token_divergence")
        shared += 1
    if len(left) == len(right):
        return SharedPrefix(shared, None, "identical")
    return SharedPrefix(shared, shared, "length_or_eos")


def align_capture_rows(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Align only rows that predict the same absolute position and history.

    Rejected speculative branches remain visible in raw data but are not paired
    merely because they have the same output ordinal.
    """

    index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in right:
        if row.get("branch_status") == "rejected":
            continue
        key = (int(row["prediction_position"]), str(row["history_sha256"]))
        index[key] = row
    result = []
    for row in left:
        if row.get("branch_status") == "rejected":
            continue
        key = (int(row["prediction_position"]), str(row["history_sha256"]))
        candidate = index.get(key)
        if candidate is not None:
            result.append((row, candidate))
    return result


def select_replay_positions(
    prompt_length: int,
    output_length: int,
    first_divergence: int | None,
    width: int = 8,
) -> list[dict[str, Any]]:
    """Select start-adjacent and divergence/position-64 replay windows."""

    if width <= 0:
        raise ValueError("width must be positive")
    if output_length < width:
        return [{"status": "skipped", "reason": "output_shorter_than_window"}]
    candidates = [(0, "generation_start")]
    pivot = first_divergence if first_divergence is not None else min(64, output_length - 1)
    start = max(0, min(pivot - 1, output_length - width))
    candidates.append((start, "first_divergence" if first_divergence is not None else "position_64"))
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for output_start, reason in candidates:
        if output_start in seen:
            result.append({"status": "skipped", "reason": "duplicate_window", "kind": reason})
            continue
        seen.add(output_start)
        result.append(
            {
                "status": "selected",
                "kind": reason,
                "output_start": output_start,
                "absolute_input_start": prompt_length + output_start,
                "width": width,
            }
        )
    return result

