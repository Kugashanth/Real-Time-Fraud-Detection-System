def should_trigger_retraining(
    feature_psi_scores: dict[str, float],
    estimated_pr_auc_drop: float,
    psi_limit: float = 0.25,
    max_pr_auc_drop: float = 0.05,
) -> dict[str, object]:
    """Determines if automated pipeline retraining should be triggered based on drift or CBPE SLA breach."""
    drifted_features = [feat for feat, psi in feature_psi_scores.items() if psi >= psi_limit]
    sla_breached = estimated_pr_auc_drop >= max_pr_auc_drop
    trigger = len(drifted_features) >= 2 or sla_breached
    return {
        "retrain_triggered": trigger,
        "drifted_features": drifted_features,
        "sla_breached": sla_breached,
    }
