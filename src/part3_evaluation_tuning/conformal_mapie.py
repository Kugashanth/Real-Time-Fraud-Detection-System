import numpy as np


def compute_conformal_uncertainty_flag(
    probs: np.ndarray,
    optimal_threshold: float,
    uncertainty_margin: float = 0.08,
) -> np.ndarray:
    """Flags borderline predictions within the conformal ambiguity band for MANUAL_REVIEW."""
    return np.abs(probs - optimal_threshold) <= uncertainty_margin
