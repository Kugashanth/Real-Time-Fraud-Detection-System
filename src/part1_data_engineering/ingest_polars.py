import polars as pl
import duckdb
from pathlib import Path


def load_transactions_polars(csv_path: str | Path) -> pl.DataFrame:
    """High-speed transaction ingestion using Polars."""
    df = pl.read_csv(csv_path)
    return df.sort("Time")


def query_summary_stats_duckdb(df: pl.DataFrame) -> pl.DataFrame:
    """Fast aggregation of fraud rate and monetary skew using DuckDB."""
    query = """
    SELECT
        Class,
        COUNT(*) AS tx_count,
        ROUND(AVG(Amount), 2) AS avg_amount,
        ROUND(MEDIAN(Amount), 2) AS median_amount,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY Amount), 2) AS p99_amount
    FROM df
    GROUP BY Class
    ORDER BY Class
    """
    return duckdb.query(query).pl()
