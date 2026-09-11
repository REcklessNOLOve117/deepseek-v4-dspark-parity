# DeepSeek-V4-0731 + DSpark numerical parity

This workspace implements the first research milestone for the fixed image and
the existing `dsv4f-vllm-dspark-8dev-8002` container. The container is reused
in place; the experiment backs up and temporarily instruments the installed
vLLM runner, binds the research API to loopback, exports results, and verifies
byte-for-byte restoration.

The entry point is:

```bash
python tests/integration/defs/deepseek_v4_dspark_parity_harness.py --help
```

The intended run order is `A1 -> S1 -> A2 -> S2`. `collect` resets the prefix
cache before every prompt and reuses the token IDs first recorded by A1.
`compare` enforces shared-prefix semantics. `replay` selects the fixed-token
windows; an execution is valid only when the exported replay record confirms
identical state fingerprints, position maps, backend settings and eager mode.

Generated figures are derived from the same summary JSON used for the report:

- generation self-repeat versus cross-mode overview;
- prompt-by-token first-divergence map;
- fixed-state replay error and candidate-rank views once replay raw logits are
  available;
- an interactive local report containing summaries only, never the large raw
  logit vectors.

Raw capture uses an append-only binary file plus JSONL offsets. It stops and
marks the run incomplete before exceeding the configured 32 GiB limit.

## Repository scope

This repository contains the research harness, instrumentation helpers,
replay implementation, manifests, and unit tests. Generated captures,
full-vocabulary logits, reports, and figures are deliberately excluded: they
can be large and may contain environment-specific metadata. Keep those files
in an access-controlled results directory and verify them with SHA-256 before
analysis.
