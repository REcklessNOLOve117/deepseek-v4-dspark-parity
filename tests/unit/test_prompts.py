from collections import Counter

from dsv4_parity.prompts import PROMPTS, validate_prompts


def test_prompt_registry_is_fixed_and_balanced():
    validate_prompts()
    assert len(PROMPTS) == 32
    assert Counter(row.category for row in PROMPTS) == {
        "code": 8,
        "tool": 8,
        "chinese_qa": 8,
        "reasoning": 8,
    }

