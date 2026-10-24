from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.part1_data_engineering.resampling_pipeline import create_resampling_pipeline


def build_xgboost_pipeline(use_smote: bool = True, smote_ratio: float = 0.15, scale_pos_weight: float = 1.0, **kwargs):
    """Gradient Boosted Trees (XGBoost hist) with SMOTE-ENN or scale_pos_weight."""
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "scale_pos_weight": 1.0 if use_smote else scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    xgb = XGBClassifier(**params)
    if use_smote:
        return create_resampling_pipeline(xgb, sampling_strategy=smote_ratio)
    return xgb


def build_lightgbm_benchmark(smote_ratio: float = 0.15):
    """Modern leaf-wise gradient boosting benchmark."""
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    return create_resampling_pipeline(lgbm, sampling_strategy=smote_ratio)
