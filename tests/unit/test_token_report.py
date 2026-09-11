from dsv4_parity.token_report import build_token_divergence_report


class _Client:
    def detokenize(self, token_ids):
        return "|".join(map(str, token_ids))


def test_token_report_marks_first_divergence():
    comparison = {
        "pairs": [
            {
                "left_run_id": "A1",
                "right_run_id": "S1",
                "prompts": [
                    {
                        "prompt_id": "p",
                        "category": "code",
                        "first_divergence": 1,
                        "termination": "token_divergence",
                    }
                ],
            }
        ]
    }
    generations = {
        "A1": {"prompts": [{"prompt_id": "p", "token_ids": [1, 2, 3]}]},
        "S1": {"prompts": [{"prompt_id": "p", "token_ids": [1, 9, 3]}]},
    }
    report = build_token_divergence_report(comparison, generations, _Client())
    assert report["rows"][0]["left_token_text"] == "2"
    assert report["rows"][0]["right_token_text"] == "9"
