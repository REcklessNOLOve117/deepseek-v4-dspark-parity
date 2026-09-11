from dsv4_parity.replay_collect import build_execution_manifest


def test_execution_manifest_uses_a1_prefix_and_eight_fixed_inputs():
    plan = {
        "source_run_id": "A1",
        "sequence": ["AR", "Q7", "Q8", "Q8", "Q7", "AR"],
        "prompts": [
            {
                "prompt_id": "p1",
                "prompt_token_ids": [1, 2],
                "fixed_output_token_ids": list(range(10, 30)),
                "windows": [
                    {
                        "status": "selected",
                        "kind": "first_divergence",
                        "output_start": 3,
                        "absolute_input_start": 5,
                        "width": 8,
                    },
                    {"status": "skipped", "reason": "duplicate_window"},
                ],
            }
        ],
    }
    result = build_execution_manifest(plan)
    assert len(result["windows"]) == 1
    window = result["windows"][0]
    assert window["prefill_token_ids"] == [1, 2, 10, 11, 12]
    assert window["fixed_input_token_ids"] == list(range(13, 21))
    assert window["request_id"].startswith("parity:REPLAY:")
    assert len(result["skipped"]) == 1
