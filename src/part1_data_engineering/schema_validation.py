import pandas as pd
import pandera as pa


def build_transaction_schema() -> pa.DataFrameSchema:
    """Pandera schema gate to prevent corrupted or out-of-range transactions."""
    columns = {
        "Time": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "Amount": pa.Column(float, pa.Check.ge(0.0), nullable=False),
        "Class": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    }
    for i in range(1, 29):
        columns[f"V{i}"] = pa.Column(float, nullable=False)
    return pa.DataFrameSchema(columns=columns, coerce=True)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    schema = build_transaction_schema()
    return schema.validate(df, lazy=True)
