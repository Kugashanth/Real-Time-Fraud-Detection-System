import json
import numpy as np
from pathlib import Path
from sklearn.metrics import fbeta_score, confusion_matrix


def find_optimal_thresholds(
    y_val: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = 150.0,
    fp_cost: float = 2.5,
) -> dict[str, float]:
    """Sweeps probability cutoffs [0.01..0.99] to find F2-optimal and Business-Cost-optimal thresholds."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f2 = -1.0
    best_f2_thresh = 0.50
    min_cost = float("inf")
    best_cost_thresh = 0.50

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2.0, zero_division=0)
        if f2 > best_f2:
            best_f2 = float(f2)
            best_f2_thresh = float(t)

        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < min_cost:
            min_cost = float(cost)
            best_cost_thresh = float(t)

    result = {
        "optimal_cost_threshold": round(best_cost_thresh, 4),
        "min_validation_cost_usd": round(min_cost, 2),
        "optimal_f2_threshold": round(best_f2_thresh, 4),
        "max_validation_f2": round(best_f2, 5),
        "low_risk_max": round(min(0.12, best_cost_thresh * 0.4), 4),
        "high_risk_min": round(best_cost_thresh, 4),
    }
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/threshold_config.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
