import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def generate_synthetic_imbalanced_data(n_samples: int = 10000, fraud_ratio: float = 0.01) -> pd.DataFrame:
    """Generates realistic credit card dataset when raw CSV is absent for CI/testing."""
    rng = np.random.default_rng(42)
    n_fraud = max(20, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    data = {"Time": np.sort(rng.uniform(0, 172800, size=n_samples))}
    for i in range(1, 29):
        legit_v = rng.normal(0.0, 1.0, size=n_legit)
        fraud_v = rng.normal(-2.2 if i in (10, 12, 14, 17) else 0.5, 1.5, size=n_fraud)
        data[f"V{i}"] = np.concatenate([legit_v, fraud_v])

    data["Amount"] = np.concatenate([
        rng.exponential(scale=65.0, size=n_legit),
        rng.exponential(scale=210.0, size=n_fraud),
    ])
    data["Class"] = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame(data).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def split_and_save_datasets(
    df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
    artifacts_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split and RobustScaler fit strictly on Train split."""
    output_path = Path(output_dir)
    artifacts_path = Path(artifacts_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["Class"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["Class"], random_state=42)

    scaler = RobustScaler()
    scale_cols = ["Time", "Amount"]
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    train_df.to_parquet(output_path / "train.parquet", index=False)
    val_df.to_parquet(output_path / "val.parquet", index=False)
    test_df.to_parquet(output_path / "test.parquet", index=False)
    joblib.dump(scaler, artifacts_path / "preprocessor.joblib")

    return train_df, val_df, test_df


if __name__ == "__main__":
    raw_csv = Path("data/raw/creditcard.csv")
    df_in = pd.read_csv(raw_csv) if raw_csv.exists() else generate_synthetic_imbalanced_data()
    split_and_save_datasets(df_in)
    print("Saved stratified train/val/test parquet splits and preprocessor.joblib.")
