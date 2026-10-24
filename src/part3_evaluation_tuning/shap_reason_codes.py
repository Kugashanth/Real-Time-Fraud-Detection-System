import shap
import numpy as np


def extract_top_shap_reasons(tree_model, X_row: np.ndarray, feature_names: list[str], top_k: int = 3) -> list[str]:
    """Computes local SHAP values for a single transaction and returns top-k human-readable reason codes."""
    explainer = shap.TreeExplainer(tree_model)
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):
        row_vals = shap_vals[1][0]
    else:
        row_vals = shap_vals[0]

    top_indices = np.argsort(np.abs(row_vals))[::-1][:top_k]
    return [f"{feature_names[idx]} ({row_vals[idx]:+.3f})" for idx in top_indices]
