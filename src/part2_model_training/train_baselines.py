from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from src.part1_data_engineering.resampling_pipeline import create_resampling_pipeline


def build_logistic_regression_pipeline(use_smote: bool = True, smote_ratio: float = 0.15):
    """Baseline Logistic Regression comparing SMOTE-ENN vs cost-sensitive learning."""
    if use_smote:
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        return create_resampling_pipeline(clf, sampling_strategy=smote_ratio)
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)


def build_random_forest_pipeline(use_balanced_rf: bool = True, smote_ratio: float = 0.15):
    """Bagging ensemble comparing BalancedRandomForest vs SMOTE RandomForest."""
    if use_balanced_rf:
        return BalancedRandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1)
    return create_resampling_pipeline(rf, sampling_strategy=smote_ratio)
