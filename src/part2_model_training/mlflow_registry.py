import mlflow
from typing import Any


def log_experiment_to_mlflow(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any = None,
) -> str:
    """Logs model parameters, PR-AUC/F2 metrics, and artifacts to MLflow."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model is not None:
            mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id
