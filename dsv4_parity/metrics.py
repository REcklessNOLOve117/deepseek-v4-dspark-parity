from __future__ import annotations

import math
from typing import Any

import numpy as np


def _token_set_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, a.size, b.size)
    if k == 0:
        return 1.0
    ia = np.argpartition(a, -k)[-k:]
    ib = np.argpartition(b, -k)[-k:]
    return len(set(ia.tolist()) & set(ib.tolist())) / k


def _top_two(values: np.ndarray) -> tuple[int | None, float | None, float | None]:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None, None, None
    ordered = finite[np.argsort(values[finite], kind="stable")]
    top1 = int(ordered[-1])
    if ordered.size == 1:
        return top1, float(values[top1]), math.inf
    top2 = int(ordered[-2])
    return top1, float(values[top1]), float(values[top1] - values[top2])


def logits_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare two complete vocabulary logit vectors in float64 reductions.

    Non-finite entries are counted and excluded from numeric reductions. If the
    finite masks differ, bitwise equality is necessarily false. Relative L2 is
    0 for two all-zero finite vectors and infinity when only the reference norm
    is zero.
    """

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {ref.shape} != {cand.shape}")
    if ref.ndim != 1:
        raise ValueError(f"expected 1D logits, got {ref.ndim}D")

    ref64 = ref.astype(np.float64, copy=False)
    cand64 = cand.astype(np.float64, copy=False)
    finite_ref = np.isfinite(ref64)
    finite_cand = np.isfinite(cand64)
    valid = finite_ref & finite_cand
    finite_mask_equal = bool(np.array_equal(finite_ref, finite_cand))
    ref_nonfinite = int((~finite_ref).sum())
    cand_nonfinite = int((~finite_cand).sum())

    if valid.any():
        delta = ref64[valid] - cand64[valid]
        abs_delta = np.abs(delta)
        max_abs = float(abs_delta.max(initial=0.0))
        mean_abs = float(abs_delta.mean())
        delta_norm = float(np.linalg.norm(delta))
        ref_norm = float(np.linalg.norm(ref64[valid]))
        cand_norm = float(np.linalg.norm(cand64[valid]))
        if ref_norm == 0.0:
            relative_l2 = 0.0 if delta_norm == 0.0 else math.inf
        else:
            relative_l2 = delta_norm / ref_norm
        denom = ref_norm * cand_norm
        if denom == 0.0:
            cosine = 1.0 if delta_norm == 0.0 else None
        else:
            cosine = float(np.dot(ref64[valid], cand64[valid]) / denom)
    else:
        max_abs = mean_abs = math.nan
        relative_l2 = math.nan
        cosine = None

    ref_top1, ref_top1_logit, ref_gap = _top_two(ref64)
    cand_top1, cand_top1_logit, cand_gap = _top_two(cand64)
    return {
        "vocab_size": int(ref.size),
        "valid_count": int(valid.sum()),
        "reference_nonfinite_count": ref_nonfinite,
        "candidate_nonfinite_count": cand_nonfinite,
        "finite_mask_equal": finite_mask_equal,
        "bitwise_equal": bool(
            finite_mask_equal and np.array_equal(ref.view(np.uint8), cand.view(np.uint8))
        ),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "reference_top1_token_id": ref_top1,
        "candidate_top1_token_id": cand_top1,
        "reference_top1_logit": ref_top1_logit,
        "candidate_top1_logit": cand_top1_logit,
        "reference_top1_top2_gap": ref_gap,
        "candidate_top1_top2_gap": cand_gap,
        "top1_match": ref_top1 is not None and ref_top1 == cand_top1,
        "top5_overlap": _token_set_overlap(ref64, cand64, 5),
        "top20_overlap": _token_set_overlap(ref64, cand64, 20),
    }


def near_tie(gap: float | None, threshold: float) -> bool:
    return gap is not None and math.isfinite(gap) and gap <= threshold

