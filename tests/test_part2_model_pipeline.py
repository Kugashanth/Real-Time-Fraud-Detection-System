from src.part1_data_engineering.split_and_scale import generate_synthetic_imbalanced_data
from src.part2_model_training.train_baselines import build_logistic_regression_pipeline, build_random_forest_pipeline
from src.part2_model_training.train_boosters import build_xgboost_pipeline


def test_all_three_model_families_train_and_score():
    df = generate_synthetic_imbalanced_data(n_samples=500, fraud_ratio=0.06)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    for builder in (
        lambda: build_logistic_regression_pipeline(use_smote=False),
        lambda: build_random_forest_pipeline(use_balanced_rf=True),
        lambda: build_xgboost_pipeline(use_smote=False, scale_pos_weight=15.0, n_estimators=30),
    ):
        model = builder()
        model.fit(X.iloc[:350], y.iloc[:350])
        probs = model.predict_proba(X.iloc[350:])[:, 1]
        assert len(probs) == 150
