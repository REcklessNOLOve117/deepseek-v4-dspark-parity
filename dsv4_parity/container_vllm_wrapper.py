#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ACTIVE = Path("/opt/dsv4-parity/active_run.json")


def _remove_option(argv: list[str], name: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == name:
            index += 2
            continue
        if value.startswith(name + "="):
            index += 1
            continue
        output.append(value)
        index += 1
    return output


def _set_option(argv: list[str], name: str, value: str) -> list[str]:
    return _remove_option(argv, name) + [name, value]


def configure(argv: list[str], config: dict[str, object]) -> list[str]:
    result = list(argv)
    if "serve" not in result:
        raise RuntimeError(f"expected vllm serve argv, got {result}")
    serve_index = result.index("serve")
    if serve_index + 1 >= len(result):
        raise RuntimeError("missing model argument")
    result[serve_index + 1] = str(config["model_path"])
    for name, value in (
        ("--host", "127.0.0.1"),
        ("--port", str(config.get("port", 8002))),
        ("--dtype", "bfloat16"),
        ("--max-num-seqs", "1"),
        ("--max-num-batched-tokens", "2048"),
    ):
        result = _set_option(result, name, value)
    result = _remove_option(result, "--speculative-config")
    if config["mode"] == "dspark":
        result += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "num_speculative_tokens": 7,
                    "draft_sample_method": "probabilistic",
                },
                separators=(",", ":"),
            ),
        ]
    elif config["mode"] != "off":
        raise RuntimeError(f"unsupported mode: {config['mode']}")
    if config.get("enforce_eager"):
        result = [value for value in result if value != "--enforce-eager"]
        result.append("--enforce-eager")
    return result


def main() -> int:
    if not ACTIVE.exists():
        raise RuntimeError(f"experimental wrapper requires {ACTIVE}")
    config = json.loads(ACTIVE.read_text(encoding="utf-8"))
    os.environ["PYTHONPATH"] = "/opt/dsv4-parity" + os.pathsep + os.environ.get("PYTHONPATH", "")
    if "/opt/dsv4-parity" not in sys.path:
        sys.path.insert(0, "/opt/dsv4-parity")
    os.environ["DSV4_PARITY_CAPTURE"] = "1" if config.get("capture", True) else "0"
    os.environ["DSV4_PARITY_CAPTURE_DIR"] = str(config["capture_dir"])
    os.environ["DSV4_PARITY_CAPTURE_LIMIT_BYTES"] = str(config.get("capture_limit_bytes", 32 << 30))
    os.environ["DSV4_PARITY_RUN_ID"] = str(config["run_id"])
    os.environ["DSV4_SPEC_MODE"] = str(config["mode"])
    if config.get("replay_manifest"):
        os.environ["DSV4_REPLAY_MANIFEST"] = str(config["replay_manifest"])
        os.environ["DSV4_REPLAY_OUTPUT_DIR"] = str(config["replay_output_dir"])
        os.environ["DSV4_REPLAY_LIMIT_BYTES"] = str(
            config.get("replay_limit_bytes", 32 << 30)
        )
    # The cache reset endpoint is part of this vLLM build's development router.
    # It is safe here because the research server is forced to loopback only.
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    configured = configure(sys.argv, config)
    run_dir = Path(str(config["capture_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "effective_argv.json").write_text(
        json.dumps(configured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sys.argv[:] = configured
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
