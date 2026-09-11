from dsv4_parity.replay_compare import _aggregate


def test_replay_aggregate_handles_flip_and_near_tie():
    base = {
        "bitwise_equal": False,
        "top1_match": False,
        "top5_overlap": 0.8,
        "top20_overlap": 0.9,
        "max_abs_error": 0.25,
        "mean_abs_error": 0.01,
        "relative_l2_error": 0.02,
        "cosine_similarity": 0.999,
        "near_tie_1e-4": False,
        "near_tie_1e-3": True,
        "near_tie_1e-2": True,
    }
    result = _aggregate([base])
    assert result["rows"] == 1
    assert result["top1_flip_rate"] == 1.0
    assert result["near_tie_flip_counts"]["1e-3"] == 1
