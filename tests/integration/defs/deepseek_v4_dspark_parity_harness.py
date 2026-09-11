#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dsv4_parity.alignment import select_replay_positions
from dsv4_parity.collect import collect_run
from dsv4_parity.compare import add_capture_comparisons, compare_generation_matrix
from dsv4_parity.capture_audit import audit_capture
from dsv4_parity.http_client import VllmClient
from dsv4_parity.io import atomic_write_json, read_json
from dsv4_parity.prompts import PROMPTS
from dsv4_parity.replay_collect import build_execution_manifest, collect_replay_requests
from dsv4_parity.replay_compare import compare_replay
from dsv4_parity.token_report import build_token_divergence_report


def _env_default(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4-0731 / DSpark numerical parity harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect one deterministic generation run")
    collect.add_argument("--endpoint", default="http://127.0.0.1:8002")
    collect.add_argument("--model", default="deepseek-v4-flash")
    collect.add_argument("--mode", choices=("off", "dspark"), default=_env_default("DSV4_SPEC_MODE", "off"))
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--output", default=_env_default("DSV4_OUTPUT_JSON"), required=_env_default("DSV4_OUTPUT_JSON") is None)
    collect.add_argument("--prompt-tokens", default=_env_default("DSV4_BASELINE_JSON"))
    collect.add_argument("--max-tokens", type=int, default=256)
    collect.add_argument("--seed", type=int, default=20260910)

    replay = subparsers.add_parser("replay", help="build fixed-token replay windows from A1 and a comparison")
    replay.add_argument("--baseline", required=True)
    replay.add_argument("--comparison", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--window", type=int, default=8)
    replay.add_argument("--execution-manifest")

    replay_run = subparsers.add_parser(
        "replay-run", help="submit fixed-state replay requests to an instrumented engine"
    )
    replay_run.add_argument("--endpoint", default="http://127.0.0.1:8002")
    replay_run.add_argument("--model", default="deepseek-v4-flash")
    replay_run.add_argument("--manifest", required=True)
    replay_run.add_argument("--output", required=True)
    replay_run.add_argument("--limit", type=int)
    replay_run.add_argument("--start", type=int, default=0)

    replay_compare = subparsers.add_parser(
        "replay-compare", help="compare fixed-state replay raw logits"
    )
    replay_compare.add_argument("--replay-dir", required=True)
    replay_compare.add_argument("--output", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="compare one short request with capture excluded versus enabled"
    )
    smoke.add_argument("--endpoint", default="http://127.0.0.1:8002")
    smoke.add_argument("--model", default="deepseek-v4-flash")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--max-tokens", type=int, default=8)
    smoke.add_argument("--reverse", action="store_true", help="run capture-enabled request first")

    compare = subparsers.add_parser("compare", help="compare A1/S1/A2/S2 and render plots")
    compare.add_argument("runs", nargs="+", help="generation JSON files")
    compare.add_argument("--output", required=True)
    compare.add_argument("--plots")
    compare.add_argument(
        "--capture",
        action="append",
        default=[],
        metavar="RUN_ID=DIR",
        help="raw capture directory; provide once per run",
    )
    audit = subparsers.add_parser("audit", help="validate raw capture structure and token mapping")
    audit.add_argument("--run-id", required=True)
    audit.add_argument("--capture-dir", required=True)
    audit.add_argument("--generation", required=True)
    audit.add_argument("--output", required=True)

    token_report = subparsers.add_parser(
        "token-report", help="decode first-divergence token windows for reporting"
    )
    token_report.add_argument("runs", nargs="+")
    token_report.add_argument("--comparison", required=True)
    token_report.add_argument("--endpoint", default="http://127.0.0.1:8002")
    token_report.add_argument("--left", default="A1")
    token_report.add_argument("--right", default="S1")
    token_report.add_argument("--output", required=True)
    return parser


def command_replay(args: argparse.Namespace) -> None:
    baseline = read_json(args.baseline)
    comparison = read_json(args.comparison)
    cross = next(
        (
            pair
            for pair in comparison["pairs"]
            if {pair["left_run_id"], pair["right_run_id"]} == {"A1", "S1"}
        ),
        None,
    )
    if cross is None:
        raise ValueError("comparison must contain A1/S1")
    baseline_rows = {row["prompt_id"]: row for row in baseline["prompts"]}
    comparison_rows = {row["prompt_id"]: row for row in cross["prompts"]}
    plans = []
    for prompt_id, row in baseline_rows.items():
        plans.append(
            {
                "prompt_id": prompt_id,
                "request_id": row["request_id"],
                "prompt_token_ids": row["prompt_token_ids"],
                "fixed_output_token_ids": row["token_ids"],
                "windows": select_replay_positions(
                    len(row["prompt_token_ids"]),
                    len(row["token_ids"]),
                    comparison_rows[prompt_id]["first_divergence"],
                    args.window,
                ),
            }
        )
    plan = {
            "schema_version": 1,
            "kind": "fixed_token_replay_plan",
            "source_run_id": baseline["run_id"],
            "sequence": ["AR", "Q7", "Q8", "Q8", "Q7", "AR"],
            "requirements": {
                "eager": True,
                "torch_compile": False,
                "draft_execution": False,
                "restore_same_state_before_each_path": True,
                "state_fingerprint_required": True,
                "causal_mask_check_required": True,
            },
            "prompts": plans,
        }
    atomic_write_json(args.output, plan)
    if args.execution_manifest:
        atomic_write_json(args.execution_manifest, build_execution_manifest(plan))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        artifact = collect_run(
            endpoint=args.endpoint,
            model=args.model,
            mode=args.mode,
            run_id=args.run_id,
            output_path=args.output,
            prompt_tokens_path=args.prompt_tokens,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        print(json.dumps({"run_id": artifact["run_id"], "prompts": len(artifact["prompts"])}))
    elif args.command == "compare":
        generation_artifacts = {run["run_id"]: run for run in map(read_json, args.runs)}
        result = compare_generation_matrix(args.runs, args.output)
        if args.capture:
            capture_dirs = {}
            for item in args.capture:
                run_id, separator, directory = item.partition("=")
                if not separator:
                    raise ValueError(f"invalid --capture value: {item}")
                capture_dirs[run_id] = directory
            result = add_capture_comparisons(result, capture_dirs, generation_artifacts)
            atomic_write_json(args.output, result)
        if args.plots:
            from dsv4_parity.plots import build_plots

            build_plots(args.output, args.plots)
        print(json.dumps({"pairs": len(result["pairs"]), "output": args.output}))
    elif args.command == "replay":
        command_replay(args)
        print(json.dumps({"output": args.output}))
    elif args.command == "replay-run":
        result = collect_replay_requests(
            args.endpoint,
            args.model,
            read_json(args.manifest),
            args.output,
            args.limit,
            args.start,
        )
        print(
            json.dumps(
                {
                    "windows_requested": result["windows_requested"],
                    "all_first_tokens_forced": result["all_first_tokens_forced"],
                    "output": args.output,
                }
            )
        )
    elif args.command == "replay-compare":
        result = compare_replay(args.replay_dir, args.output)
        print(
            json.dumps(
                {
                    "num_windows": result["num_windows"],
                    "all_state_restores_passed": result["all_state_restores_passed"],
                    "all_causal_mask_checks_passed": result[
                        "all_causal_mask_checks_passed"
                    ],
                    "output": args.output,
                }
            )
        )
    elif args.command == "audit":
        result = audit_capture(
            args.capture_dir,
            read_json(args.generation),
            args.run_id,
        )
        atomic_write_json(args.output, result)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "raw_structure_passed": result["raw_structure_passed"],
                    "committed_expected_is_raw_max_rate": result[
                        "committed_expected_is_raw_max_rate"
                    ],
                    "output": args.output,
                }
            )
        )
    elif args.command == "token-report":
        generations = {row["run_id"]: row for row in map(read_json, args.runs)}
        report = build_token_divergence_report(
            read_json(args.comparison),
            generations,
            VllmClient(args.endpoint),
            left_run_id=args.left,
            right_run_id=args.right,
        )
        atomic_write_json(args.output, report)
        print(json.dumps({"rows": len(report["rows"]), "output": args.output}))
    elif args.command == "smoke":
        client = VllmClient(args.endpoint)
        client.wait_ready()
        prompt = PROMPTS[0]
        prompt_token_ids = client.tokenize(prompt.text)
        outputs = []
        order = [
            ("smoke:no-capture:code_01", False),
            ("parity:SMOKE:code_01", True),
        ]
        if args.reverse:
            order.reverse()
        for request_id, capture_expected in order:
            reset = client.reset_prefix_cache()
            response = client.complete(
                model=args.model,
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
                max_tokens=args.max_tokens,
                seed=20260910,
            )
            choice = response["choices"][0]
            outputs.append(
                {
                    "request_id": request_id,
                    "capture_expected": capture_expected,
                    "prefix_reset_response": reset,
                    "text": choice["text"],
                    "token_ids": choice["token_ids"],
                    "finish_reason": choice["finish_reason"],
                }
            )
        passed = outputs[0]["text"] == outputs[1]["text"] and outputs[0]["token_ids"] == outputs[1]["token_ids"]
        atomic_write_json(
            args.output,
            {
                "kind": "capture_observer_smoke",
                "passed": passed,
                "prompt_id": prompt.prompt_id,
                "prompt_token_ids": prompt_token_ids,
                "outputs": outputs,
            },
        )
        print(json.dumps({"passed": passed, "output": args.output}))
        if not passed:
            return 2
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
