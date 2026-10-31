from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from .job_utils import check_runtime_environment


def track_values(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, Any],
    artifacts: list[Any],
    model: Any,
    model_flavor: Any,
    profile: str = "sandbox",
):

    if check_runtime_environment() == "local":
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri("databricks-uc")

    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(f"tags.mlflow.runName = '{run_name}'"),
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if runs:
        run = mlflow.start_run(run_id=runs[0].info.run_id)  # type: ignore
    else:
        run = mlflow.start_run(
            experiment_id=experiment.experiment_id,  # type: ignore
            run_name=run_name,
        )

    mlflow.end_run()
