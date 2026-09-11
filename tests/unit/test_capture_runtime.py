from types import SimpleNamespace

from dsv4_parity.capture_runtime import _modular_draft_count, normalize_parity_request_id


def test_normalizes_vllm_completion_engine_request_id():
    assert normalize_parity_request_id("cmpl-parity:A1:code_01-0") == "parity:A1:code_01"
    assert (
        normalize_parity_request_id("cmpl-parity:A1:code_01-0-b26e903b")
        == "parity:A1:code_01"
    )


def test_preserves_direct_request_and_rejects_non_parity():
    assert normalize_parity_request_id("parity:S1:zh_03") == "parity:S1:zh_03"
    assert normalize_parity_request_id("cmpl-smoke:no-capture:code_01-0") is None


def test_modular_runner_prefill_represents_no_drafts_as_none():
    assert _modular_draft_count(SimpleNamespace(num_draft_tokens_per_req=None), 0) == 0
    assert _modular_draft_count(SimpleNamespace(num_draft_tokens_per_req=[7]), 0) == 7
