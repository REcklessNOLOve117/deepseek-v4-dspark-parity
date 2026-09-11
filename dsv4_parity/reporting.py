from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import atomic_write_json, read_json, sha256_json


COLORS = {
    "self": "#4477AA",
    "cross": "#CC6677",
    "q7": "#228833",
    "q8": "#AA3377",
    "causal": "#EE7733",
    "pass": "#228833",
    "fail": "#CC3311",
}


def _pair_kind(left: str, right: str) -> str:
    return "self" if (left.startswith("A") == right.startswith("A")) else "cross"


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_generation_overview(
    generation: dict[str, Any], capture: dict[str, Any], directory: Path
) -> list[str]:
    capture_lookup = {
        frozenset((pair["left_run_id"], pair["right_run_id"])): pair
        for pair in capture["capture_pairs"]
    }
    labels, sequence_divergence, flips, colors = [], [], [], []
    for pair in generation["pairs"]:
        left, right = pair["left_run_id"], pair["right_run_id"]
        labels.append(f"{left}–{right}")
        sequence_divergence.append(1.0 - pair["sequence_exact_match_rate"])
        flips.append(capture_lookup[frozenset((left, right))]["top1_flip_rate"])
        colors.append(COLORS[_pair_kind(left, right)])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    axes[0].bar(x, np.asarray(sequence_divergence) * 100, color=colors)
    axes[0].set_ylabel("Diverged prompts (%)")
    axes[0].set_title("Generation consistency: repeat baselines and AR/DSpark pairs")
    axes[0].set_ylim(0, 108)
    for i, value in enumerate(sequence_divergence):
        axes[0].text(i, value * 100 + 1.5, f"{value:.0%}", ha="center", fontsize=9)
    axes[1].bar(x, np.asarray(flips) * 100, color=colors)
    axes[1].set_ylabel("Raw target top-1 flips (%)")
    axes[1].set_xlabel("Run pair (blue = self-repeat, red = AR/DSpark)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for i, value in enumerate(flips):
        axes[1].text(i, value * 100 + 0.2, f"{value:.1%}", ha="center", fontsize=9)
    return _save(fig, directory, "01_generation_consistency")


def plot_prompt_token_map(generation: dict[str, Any], directory: Path) -> list[str]:
    selected_ids = (("A1", "A2"), ("S1", "S2"), ("A1", "S1"))
    pairs = []
    for left, right in selected_ids:
        pairs.append(
            next(
                pair
                for pair in generation["pairs"]
                if {pair["left_run_id"], pair["right_run_id"]} == {left, right}
            )
        )
    prompts = sorted(row["prompt_id"] for row in pairs[0]["prompts"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#C7E9C0", "#CB181D", "#D9D9D9"])
    for ax, pair in zip(axes, pairs):
        lookup = {row["prompt_id"]: row for row in pair["prompts"]}
        width = max(max(row["left_length"], row["right_length"]) for row in pair["prompts"])
        matrix = np.full((len(prompts), width), np.nan)
        for y, prompt_id in enumerate(prompts):
            row = lookup[prompt_id]
            matrix[y, : row["shared_length"]] = 0
            if row["first_divergence"] is not None:
                boundary = int(row["first_divergence"])
                matrix[y, boundary] = 1
                matrix[y, boundary + 1 :] = 2
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
        ax.set_yticks(range(len(prompts)), prompts, fontsize=7)
        ax.set_ylabel("Prompt")
        ax.set_title(f"{pair['left_run_id']} vs {pair['right_run_id']}")
    axes[-1].set_xlabel("Generated token index (green shared, red first divergence, grey excluded)")
    return _save(fig, directory, "02_prompt_token_map")


def plot_token_divergences(tokens: dict[str, Any], directory: Path) -> list[str]:
    rows = sorted(tokens["rows"], key=lambda row: (row["first_divergence"], row["prompt_id"]))[:16]
    y = np.arange(len(rows))
    positions = [row["first_divergence"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.scatter(positions, y, color=COLORS["cross"], s=50, zorder=3)
    ax.hlines(y, 0, positions, color="#BBBBBB", linewidth=1)
    ax.set_yticks(y, [row["prompt_id"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("First divergent generated-token index")
    ax.set_ylabel("Prompt")
    ax.set_title("Earliest A1 vs S1 token divergences")
    ax.grid(axis="x", alpha=0.25)
    for yy, row in zip(y, rows):
        left = repr(row["left_token_text"])
        right = repr(row["right_token_text"])
        label = f" {left} [{row['left_token_id']}]  →  {right} [{row['right_token_id']}]"
        ax.annotate(label, (row["first_divergence"], yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    return _save(fig, directory, "03_single_sample_token_comparison")


def plot_replay_errors(replay: dict[str, Any], directory: Path) -> list[str]:
    order = ["self_ar", "self_q7", "self_q8", "shape_ar_q7", "shape_ar_q8", "shape_q7_q8", "causal_mask"]
    labels = ["AR repeat", "Q7 repeat", "Q8 repeat", "AR↔Q7", "AR↔Q8", "Q7↔Q8", "Causal control"]
    grouped = {
        kind: [row for row in replay["rows"] if row["comparison_kind"] == kind]
        for kind in order
    }
    values = [[row["max_abs_error"] for row in grouped[kind]] for kind in order]
    flips = [replay["aggregates"][kind]["top1_flip_rate"] * 100 for kind in order]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    parts = axes[0].violinplot(values, showmedians=True, showextrema=False)
    for index, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS["self"] if index < 3 else COLORS["cross"])
        body.set_alpha(0.75)
    parts["cmedians"].set_color("#222222")
    axes[0].set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    axes[0].set_ylabel("Max absolute logit error")
    axes[0].set_title("Fixed-state eager replay: cross-shape error versus exact repeats")
    axes[0].grid(axis="y", alpha=0.25)
    bar_colors = [COLORS["self"]] * 3 + [COLORS["cross"]] * 3 + [COLORS["causal"]]
    axes[1].bar(np.arange(len(labels)), flips, color=bar_colors)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axes[1].set_ylabel("Top-1 flip rate (%)")
    axes[1].set_xlabel("Comparison")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(flips):
        axes[1].text(index, value + 0.25, f"{value:.1f}%", ha="center", fontsize=9)
    return _save(fig, directory, "04_replay_numerical_error")


def plot_candidate_ranking(replay: dict[str, Any], directory: Path) -> list[str]:
    candidates = [
        row
        for row in replay["rows"]
        if row["comparison_kind"] == "shape_ar_q8"
        and row["left_path"] == "AR_1"
        and not row["top1_match"]
    ]
    row = min(candidates, key=lambda item: (item["reference_top1_top2_gap"], -item["max_abs_error"]))
    token_ids = []
    for token in row["reference_top20_token_ids"][:6] + row["candidate_top20_token_ids"][:6]:
        if token not in token_ids:
            token_ids.append(token)
    token_ids = token_ids[:8]
    ref = dict(zip(row["reference_top20_token_ids"], row["reference_top20_logits"]))
    cand = dict(zip(row["candidate_top20_token_ids"], row["candidate_top20_logits"]))
    floor = min(min(ref.values()), min(cand.values()))
    ref_values = [ref.get(token, floor) for token in token_ids]
    cand_values = [cand.get(token, floor) for token in token_ids]
    x = np.arange(len(token_ids))
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.bar(x - 0.2, ref_values, width=0.4, label=row["left_path"], color=COLORS["self"])
    ax.bar(x + 0.2, cand_values, width=0.4, label=row["right_path"], color=COLORS["cross"])
    ax.set_xticks(x, [str(token) for token in token_ids], rotation=25)
    ax.set_xlabel("Candidate token ID")
    ax.set_ylabel("Raw target logit (BF16)")
    ax.set_title(
        f"Candidate ranking flip: {row['window_id']} row {row['row']} "
        f"(position {row['prediction_position']})"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, directory, "05_candidate_token_ranking")


def plot_validity(audits: list[dict[str, Any]], replay: dict[str, Any], directory: Path) -> list[str]:
    columns = [audit["run_id"] for audit in audits] + ["Replay"]
    rows = [
        "Raw file / offsets",
        "Accepted-token mapping",
        "Input / position mapping",
        "State restore",
        "Same-shape repeat",
        "Causal content control",
    ]
    matrix = np.full((len(rows), len(columns)), np.nan)
    for column, audit in enumerate(audits):
        matrix[0, column] = float(audit["raw_structure_passed"])
        matrix[1, column] = float(audit["reconciliation"]["control_mapping_passed"])
        matrix[2, column] = float(audit["reconciliation"]["input_mapping_mismatch_rows"] == 0)
    matrix[0, -1] = float(replay["raw_offsets_complete"])
    matrix[2, -1] = float(replay["all_input_and_position_mappings_passed"])
    matrix[3, -1] = float(replay["all_state_restores_passed"])
    matrix[4, -1] = float(
        all(replay["aggregates"][kind]["bitwise_equal_rate"] == 1.0 for kind in ("self_ar", "self_q7", "self_q8"))
    )
    matrix[5, -1] = float(replay["all_causal_mask_checks_passed"])
    cmap = matplotlib.colors.ListedColormap([COLORS["fail"], COLORS["pass"]])
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title("Experiment validity matrix (green pass, red fail, blank not applicable)")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if np.isfinite(matrix[y, x]):
                ax.text(x, y, "PASS" if matrix[y, x] else "FAIL", ha="center", va="center", color="white", fontsize=9)
    return _save(fig, directory, "06_experiment_validity")


def build_summary(
    generation: dict[str, Any],
    capture: dict[str, Any],
    replay: dict[str, Any],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_pairs = []
    capture_lookup = {
        frozenset((pair["left_run_id"], pair["right_run_id"])): pair
        for pair in capture["capture_pairs"]
    }
    for pair in generation["pairs"]:
        key = frozenset((pair["left_run_id"], pair["right_run_id"]))
        raw = capture_lookup[key]
        generation_pairs.append(
            {
                "left": pair["left_run_id"],
                "right": pair["right_run_id"],
                "kind": _pair_kind(pair["left_run_id"], pair["right_run_id"]),
                "sequence_exact_match_rate": pair["sequence_exact_match_rate"],
                "shared_history_rows": raw["shared_history_rows"],
                "raw_top1_flip_rate": raw["top1_flip_rate"],
            }
        )
    self_rates = [row["raw_top1_flip_rate"] for row in generation_pairs if row["kind"] == "self"]
    cross_rates = [row["raw_top1_flip_rate"] for row in generation_pairs if row["kind"] == "cross"]
    result = {
        "schema_version": 1,
        "kind": "experiment_summary",
        "generation_pairs": generation_pairs,
        "self_repeat_top1_flip_range": [min(self_rates), max(self_rates)],
        "cross_path_top1_flip_range": [min(cross_rates), max(cross_rates)],
        "cross_exceeds_self_max": min(cross_rates) > max(self_rates),
        "capture_raw_bytes": sum(int(audit["raw_size_bytes"]) for audit in audits),
        "capture_audits_all_passed": all(audit["raw_structure_passed"] for audit in audits),
        "replay_windows": replay["num_windows"],
        "replay_raw_bytes": replay["raw_size_bytes"],
        "replay_aggregates": replay["aggregates"],
        "replay_state_restore_all_passed": replay["all_state_restores_passed"],
        "replay_mapping_all_passed": replay["all_input_and_position_mappings_passed"],
        "replay_causal_control_all_passed": replay["all_causal_mask_checks_passed"],
        "replay_causal_control_pass_count": sum(
            row["causal_mask_passed"] for row in replay["validations"]
        ),
    }
    result["artifact_sha256"] = sha256_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    generation = read_json(args.generation)
    capture = read_json(args.capture)
    replay = read_json(args.replay)
    tokens = read_json(args.tokens)
    audits = [read_json(path) for path in args.audits]
    directory = Path(args.figures)
    outputs = []
    outputs += plot_generation_overview(generation, capture, directory)
    outputs += plot_prompt_token_map(generation, directory)
    outputs += plot_token_divergences(tokens, directory)
    outputs += plot_replay_errors(replay, directory)
    outputs += plot_candidate_ranking(replay, directory)
    outputs += plot_validity(audits, replay, directory)
    summary = build_summary(generation, capture, replay, audits)
    summary["figure_files"] = outputs
    atomic_write_json(args.summary, summary)
    print(json.dumps({"figures": len(outputs), "summary": args.summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
