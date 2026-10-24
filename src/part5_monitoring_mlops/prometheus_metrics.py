from prometheus_client import Counter, Histogram, Gauge

FRAUD_PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of scored transactions partitioned by risk tier",
    ["risk_tier"],
)

FRAUD_PROBABILITY_HISTOGRAM = Histogram(
    "fraud_probability_distribution",
    "Distribution of calibrated fraud probabilities",
    buckets=[0.01, 0.05, 0.10, 0.20, 0.315, 0.50, 0.75, 0.90, 0.99],
)

FEATURE_PSI_GAUGE = Gauge(
    "feature_psi_score",
    "Population Stability Index (PSI) covariate shift score per feature",
    ["feature_name"],
)
