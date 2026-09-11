from dsv4_parity.container_vllm_wrapper import configure


BASE = [
    "/usr/local/bin/vllm",
    "serve",
    "/old/model",
    "--port",
    "8002",
    "--speculative-config",
    "old",
]


def test_wrapper_off_removes_speculation_and_changes_model():
    result = configure(BASE, {"model_path": "/model/0731", "mode": "off"})
    assert result[result.index("serve") + 1] == "/model/0731"
    assert "--speculative-config" not in result
    assert result[result.index("--host") + 1] == "127.0.0.1"


def test_wrapper_dspark_uses_seven_tokens():
    result = configure(BASE, {"model_path": "/model/0731", "mode": "dspark"})
    payload = result[result.index("--speculative-config") + 1]
    assert '"num_speculative_tokens":7' in payload
    assert '"draft_sample_method":"probabilistic"' in payload

