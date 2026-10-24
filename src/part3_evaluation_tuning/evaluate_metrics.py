import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_imbalance_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fn_cost: float = 150.0,
    fp_cost: float = 2.5,
) -> dict[str, float]:
    """Computes PR-AUC, F1, F2, MCC, and Expected Financial Loss at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = float(fn * fn_cost + fp * fp_cost)

    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 5),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        "f2_score": round(float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)), 5),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 5),
        "expected_financial_cost_usd": round(total_cost, 2),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
