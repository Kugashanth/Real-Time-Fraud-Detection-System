import pandas as pd
from src.part1_data_engineering.split_and_scale import generate_synthetic_imbalanced_data
from src.part1_data_engineering.schema_validation import validate_dataframe
from src.part1_data_engineering.resampling_pipeline import create_resampling_pipeline
from sklearn.linear_model import LogisticRegression


def test_pandera_schema_and_smote_pipeline():
    df = generate_synthetic_imbalanced_data(n_samples=600, fraud_ratio=0.05)
    validated = validate_dataframe(df)
    assert len(validated) == 600

    X = validated.drop(columns=["Class"])
    y = validated["Class"]
    pipe = create_resampling_pipeline(LogisticRegression(max_iter=500), sampling_strategy=0.25, method="smote")
    pipe.fit(X.iloc[:400], y.iloc[:400])

    # Verify test set length is never altered during predict_proba (zero leakage)
    X_test = X.iloc[400:]
    probs = pipe.predict_proba(X_test)
    assert probs.shape == (200, 2)
