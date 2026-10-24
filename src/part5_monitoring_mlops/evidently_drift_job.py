import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Computes Population Stability Index (PSI) between reference validation baseline and live production window."""
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, 1.0)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-4, 1.0)

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi_value), 5)


def evaluate_feature_drift(reference_arr: np.ndarray, current_arr: np.ndarray, psi_threshold: float = 0.25) -> dict:
    """Runs both PSI and Kolmogorov-Smirnov (K-S) drift checks."""
    psi_val = calculate_psi(reference_arr, current_arr)
    ks_stat, ks_p_val = ks_2samp(reference_arr, current_arr)
    return {
        "psi": psi_val,
        "ks_statistic": round(float(ks_stat), 5),
        "ks_p_value": round(float(ks_p_val), 6),
        "drift_detected": bool(psi_val >= psi_threshold or ks_p_val < 0.01),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=2000)
    cur_drifted = rng.normal(0.65, 1.25, size=2000)
    report = evaluate_feature_drift(ref, cur_drifted)
    print("Drift Check Result:", report)
