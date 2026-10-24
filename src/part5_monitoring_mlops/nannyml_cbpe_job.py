import numpy as np


def estimate_cbpe_expected_f1(calibrated_probs: np.ndarray, threshold: float = 0.315) -> float:
    """
    Confidence-Based Performance Estimation (CBPE):
    Estimates post-deployment F1-Score purely from calibrated probabilities
    before delayed chargeback ground-truth labels arrive.
    """
    y_pred = (calibrated_probs >= threshold).astype(float)
    expected_tp = float(np.sum(y_pred * calibrated_probs))
    expected_fp = float(np.sum(y_pred * (1.0 - calibrated_probs)))
    expected_fn = float(np.sum((1.0 - y_pred) * calibrated_probs))

    denom = 2.0 * expected_tp + expected_fp + expected_fn
    if denom <= 0:
        return 0.0
    return round(float((2.0 * expected_tp) / denom), 5)
