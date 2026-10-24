import numpy as np
from src.part3_evaluation_tuning.evaluate_metrics import compute_imbalance_metrics
from src.part3_evaluation_tuning.threshold_optimizer import find_optimal_thresholds


def test_cost_sensitive_threshold_optimization():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 950 + [1] * 50)
    y_prob = np.concatenate([rng.uniform(0.0, 0.35, 950), rng.uniform(0.25, 0.95, 50)])

    res = find_optimal_thresholds(y_true, y_prob, fn_cost=150.0, fp_cost=2.5)
    assert 0.01 <= res["optimal_cost_threshold"] <= 0.99

    metrics = compute_imbalance_metrics(y_true, y_prob, threshold=res["optimal_cost_threshold"])
    assert metrics["pr_auc"] > 0.5
    assert metrics["f2_score"] > 0.5
