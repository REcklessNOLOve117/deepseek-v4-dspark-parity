from dsv4_parity.compare import compare_generation_pair


def run(run_id, mode, tokens):
    return {
        "run_id": run_id,
        "mode": mode,
        "prompts": [
            {
                "prompt_id": "p",
                "category": "code",
                "prompt_token_ids": [10],
                "token_ids": tokens,
                "finish_reason": "stop",
            }
        ],
    }


def test_compare_generation_first_divergence():
    result = compare_generation_pair(run("A1", "off", [1, 2, 3]), run("S1", "dspark", [1, 9, 3]))
    assert result["divergence_count"] == 1
    assert result["prompts"][0]["first_divergence"] == 1
    assert result["sequence_exact_match_rate"] == 0.0

