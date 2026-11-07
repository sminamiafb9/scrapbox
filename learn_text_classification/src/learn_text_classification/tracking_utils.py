from pathlib import Path
from typing import Any

from loguru import logger
import mlflow
from mlflow import pyfunc, pytorch, sklearn, spark, tensorflow, xgboost
from mlflow.tracking import MlflowClient

from . import job_utils

MLFLOW_FLAVORS = {
    "sklearn": sklearn,
    "spark": spark,
    "pytorch": pytorch,
    "tensorflow": tensorflow,
    "xgboost": xgboost,
    "pyfunc": pyfunc,
}


def track_values(
    log_volume: str | Path,
    experiment_name: str,
    run_name: str,
    child_run_name: str | None = None,
    profile: str = "sandbox",
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    artifacts: list[Any] | None = None,
    model: Any | None = None,
    model_flavor: Any | None = None,
    model_artifacts: dict[str, str] | None = None,
):
    if job_utils.check_runtime_environment() == "local":
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri("databricks-uc")
        logger.debug(
            "Configured MLflow tracking and registry URIs for local environment using profile '{}'.",
            profile,
        )

    client = MlflowClient()

    experiment_name = str(Path(log_volume) / experiment_name)
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id: str = experiment.experiment_id

    # Find or create the parent run.
    parent_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if parent_runs:
        parent_run_id: str = parent_runs[0].info.run_id
        logger.info("Found existing parent run: {}", parent_run_id)
    else:
        logger.info("Parent run '{}' not found. Creating new parent run.", run_name)

        parent_run = client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.runName": run_name,
            },
        )

        parent_run_id = parent_run.info.run_id

        # The parent run acts as a container for child runs.
        if child_run_name is not None:
            client.set_terminated(parent_run_id)

    # Determine the run to which values will be logged.
    if child_run_name is None:
        final_run_id = parent_run_id

    else:
        # Find an existing child run belonging to this parent.
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(f"tags.mlflow.runName = '{child_run_name}' AND tags.mlflow.parentRunId = '{parent_run_id}'"),
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )

        if child_runs:
            final_run_id = child_runs[0].info.run_id
            logger.info("Found existing child run: {}", final_run_id)

        else:
            logger.info(
                "Child run '{}' not found. Creating new child run.",
                child_run_name,
            )

            child_run = client.create_run(
                experiment_id=experiment_id,
                tags={
                    "mlflow.runName": child_run_name,
                    "mlflow.parentRunId": parent_run_id,
                },
            )

            final_run_id: str = child_run.info.run_id

    if not final_run_id:
        logger.error("Could not determine the final run ID. Exiting.")
        return

    # Start the selected run.
    run = mlflow.start_run(run_id=final_run_id)

    try:
        mlflow.set_tag("run_name", run_name)

        if child_run_name is not None:
            mlflow.set_tag("child_run_name", child_run_name)

        if params:
            mlflow.log_params(params)

        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        if model is not None:
            if model_flavor not in MLFLOW_FLAVORS:
                raise ValueError(
                    f"Unsupported MLflow model flavor: {model_flavor}. Supported flavors: {list(MLFLOW_FLAVORS)}"
                )

            flavor = MLFLOW_FLAVORS[model_flavor]

            if model_flavor == "pyfunc":
                flavor.log_model(
                    artifact_path="model",
                    python_model=model,
                    artifacts=model_artifacts,
                )
            elif model_flavor == "spark":
                raise NotImplementedError("Spark models are not supported for tracking.")
            else:
                flavor.log_model(
                    model,
                    artifact_path="model",
                )

        if artifacts:
            for artifact_path in artifacts:
                try:
                    mlflow.log_artifact(artifact_path)
                    logger.info("Logged artifact: {}", artifact_path)

                except FileNotFoundError:
                    logger.warning(
                        "Artifact not found at {}. Skipping.",
                        artifact_path,
                    )

    finally:
        mlflow.end_run()

        logger.info(
            "Successfully finished tracking run: {}",
            run.info.run_id,
        )
