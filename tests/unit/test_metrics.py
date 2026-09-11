import math

import numpy as np

from dsv4_parity.metrics import logits_metrics, near_tie


def test_identical_logits():
    values = np.array([0.0, 1.0, -2.0, 0.5], dtype=np.float32)
    result = logits_metrics(values, values.copy())
    assert result["bitwise_equal"]
    assert result["max_abs_error"] == 0.0
    assert result["relative_l2_error"] == 0.0
    assert result["cosine_similarity"] == 1.0
    assert result["top1_match"]


def test_known_errors_and_topk():
    a = np.array([0.0, 3.0, 2.0, 1.0], dtype=np.float32)
    b = np.array([0.0, 2.0, 3.5, 1.0], dtype=np.float32)
    result = logits_metrics(a, b)
    assert result["max_abs_error"] == 1.5
    assert result["mean_abs_error"] == 0.625
    assert not result["top1_match"]
    assert result["top5_overlap"] == 1.0


def test_zero_norm_and_nonfinite_handling():
    zeros = np.zeros(3, dtype=np.float32)
    result = logits_metrics(zeros, np.ones(3, dtype=np.float32))
    assert math.isinf(result["relative_l2_error"])
    nonfinite = logits_metrics(
        np.array([0.0, np.nan, np.inf], dtype=np.float32),
        np.array([0.0, np.nan, -np.inf], dtype=np.float32),
    )
    assert nonfinite["reference_nonfinite_count"] == 2
    assert nonfinite["candidate_nonfinite_count"] == 2
    assert nonfinite["finite_mask_equal"]
    assert not nonfinite["bitwise_equal"]


def test_near_tie_thresholds():
    assert near_tie(1e-3, 1e-3)
    assert not near_tie(1.01e-3, 1e-3)
    assert not near_tie(None, 1e-3)

