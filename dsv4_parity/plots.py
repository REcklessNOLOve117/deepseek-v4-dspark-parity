from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_json


PAIR_COLORS = {"self_ar": "#4477AA", "self_dspark": "#66CCEE", "cross": "#CC6677"}


def _pair_kind(pair: dict[str, Any]) -> str:
    if pair["left_mode"] == pair["right_mode"] == "off":
        return "self_ar"
    if pair["left_mode"] == pair["right_mode"] == "dspark":
        return "self_dspark"
    return "cross"


def plot_generation_overview(comparison: dict[str, Any], output_dir: str | Path) -> list[str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pairs = comparison["pairs"]
    labels = [f"{p['left_run_id']}–{p['right_run_id']}" for p in pairs]
    rates = [1.0 - p["sequence_exact_match_rate"] for p in pairs]
    colors = [PAIR_COLORS[_pair_kind(p)] for p in pairs]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars = ax.bar(labels, rates, color=colors, edgecolor="white")
    ax.set_ylabel("Prompt divergence rate")
    ax.set_ylim(0, max(0.05, max(rates, default=0) * 1.2))
    ax.set_title("Generation consistency: self-repeat vs AR/DSpark")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate, f"{rate:.1%}", ha="center", va="bottom")
    stems = []
    for suffix in ("svg", "png"):
        path = directory / f"generation_overview.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        stems.append(str(path))
    plt.close(fig)
    return stems


def plot_prompt_token_map(comparison: dict[str, Any], output_dir: str | Path) -> list[str]:
    directory = Path(output_dir)
    selected = []
    for pair in comparison["pairs"]:
        key = (pair["left_run_id"], pair["right_run_id"])
        if key in {("A1", "A2"), ("S1", "S2"), ("A1", "S1")}:
            selected.append(pair)
    if not selected:
        selected = comparison["pairs"][:3]
    prompts = sorted({row["prompt_id"] for pair in selected for row in pair["prompts"]})
    fig, axes = plt.subplots(
        max(1, len(selected)), 1, figsize=(12, 2.2 + 0.3 * len(prompts) * max(1, len(selected))), squeeze=False, constrained_layout=True
    )
    for ax, pair in zip(axes[:, 0], selected):
        max_len = max(max(row["left_length"], row["right_length"]) for row in pair["prompts"])
        matrix = np.full((len(prompts), max_len), np.nan)
        lookup = {row["prompt_id"]: row for row in pair["prompts"]}
        for y, prompt_id in enumerate(prompts):
            row = lookup[prompt_id]
            shared = row["shared_length"]
            matrix[y, :shared] = 0
            if row["first_divergence"] is not None and row["first_divergence"] < max_len:
                matrix[y, row["first_divergence"]] = 1
                matrix[y, row["first_divergence"] + 1 :] = 2
        cmap = matplotlib.colors.ListedColormap(["#D9F0D3", "#D7301F", "#D9D9D9"])
        cmap.set_bad("white")
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
        ax.set_yticks(range(len(prompts)), prompts, fontsize=7)
        ax.set_ylabel("Prompt")
        ax.set_title(f"{pair['left_run_id']} vs {pair['right_run_id']}: green=same history/token, red=first divergence, grey=post-divergence")
    axes[-1, 0].set_xlabel("Generated token position")
    outputs = []
    for suffix in ("svg", "png"):
        path = directory / f"prompt_token_map.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def write_interactive_report(comparison: dict[str, Any], output_dir: str | Path) -> str:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(comparison, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<meta charset=\"utf-8\">
<title>DeepSeek-V4 / DSpark parity</title>
<style>
body{{font:14px system-ui;margin:2rem;max-width:1200px;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:.5rem;text-align:left}}.cross{{color:#a51c30}}.self{{color:#2455a4}}button{{margin:.2rem;padding:.4rem .7rem}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem}}
</style>
<h1>DeepSeek-V4-0731 + DSpark parity</h1>
<p>选择运行对；表格只显示共同历史边界。红色代表跨 AR/DSpark，对照噪声为蓝色。</p>
<div id=buttons></div><div id=view></div>
<script type=\"application/json\" id=payload>{payload}</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent), buttons=document.getElementById('buttons'), view=document.getElementById('view');
function render(p){{const cross=p.left_mode!==p.right_mode;let h=`<h2 class=\"${{cross?'cross':'self'}}\">${{p.left_run_id}} vs ${{p.right_run_id}}</h2><p>Exact match: ${{p.exact_match_count}}/${{p.num_prompts}} (${{(100*p.sequence_exact_match_rate).toFixed(1)}}%)</p><table><tr><th>Prompt</th><th>Category</th><th>First divergence</th><th>Shared</th><th>Termination</th></tr>`;for(const r of p.prompts)h+=`<tr><td>${{r.prompt_id}}</td><td>${{r.category}}</td><td>${{r.first_divergence??'—'}}</td><td>${{r.shared_length}}</td><td>${{r.termination}}</td></tr>`;view.innerHTML=h+'</table>'}}
for(const p of data.pairs){{const b=document.createElement('button');b.textContent=p.left_run_id+' vs '+p.right_run_id;b.onclick=()=>render(p);buttons.appendChild(b)}} if(data.pairs.length)render(data.pairs[0]);
</script>"""
    path = directory / "interactive_report.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


def build_plots(comparison_path: str | Path, output_dir: str | Path) -> list[str]:
    comparison = read_json(comparison_path)
    outputs = []
    outputs.extend(plot_generation_overview(comparison, output_dir))
    outputs.extend(plot_prompt_token_map(comparison, output_dir))
    outputs.append(write_interactive_report(comparison, output_dir))
    return outputs

