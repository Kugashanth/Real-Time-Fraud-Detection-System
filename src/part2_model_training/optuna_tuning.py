import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.part2_model_training.train_boosters import build_xgboost_pipeline


def run_optuna_xgb_study(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 15) -> optuna.Study:
    """Bayesian hyperparameter optimization using Optuna TPE maximizing 5-fold PR-AUC."""
    def objective(trial: optuna.Trial) -> float:
        smote_ratio = trial.suggest_float("smote_ratio", 0.05, 0.25)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
        pipeline = build_xgboost_pipeline(use_smote=True, smote_ratio=smote_ratio, **params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="xgb_fraud_pr_auc")
    study.optimize(objective, n_trials=n_trials)
    return study
