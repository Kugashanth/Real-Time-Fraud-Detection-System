import polars as pl


def engineer_velocity_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer rolling transaction velocity and amount ratio features."""
    return (
        df.sort("Time")
        .with_columns([
            (pl.col("Time") - pl.col("Time").shift(1)).fill_null(0.0).alias("time_since_last_tx"),
            pl.col("Amount").rolling_mean(window_size=10, min_periods=1).alias("rolling_mean_amount_10"),
            pl.col("Amount").rolling_std(window_size=10, min_periods=1).fill_null(1.0).alias("rolling_std_amount_10"),
        ])
        .with_columns([
            (pl.col("Amount") / (pl.col("rolling_mean_amount_10") + 1e-5)).alias("amount_to_rolling_mean_ratio")
        ])
    )
