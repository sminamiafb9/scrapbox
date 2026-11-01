from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

# Assuming this function exists in the same module to determine the runtime environment
# and set the appropriate MLflow tracking URI/registry.
# For this implementation, we will assume it returns "local" or "databricks".
# def check_runtime_environment() -> str:
#     # Placeholder: Replace with actual implementation
#     return "local"


def track_values(
    experiment_name: str,
    run_name: str,
    parent_run_name: str | None = None,  # Added for nesting support
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    artifacts: list[Any] | None = None,
    model: Any | None = None,
    model_flavor: Any | None = None,
    profile: str = "sandbox",
):
    """
    Tracks values (params, metrics, tags, artifacts, model) to MLflow,
    managing experiment and run names with support for parent/child nesting.

    Args:
        experiment_name: The name of the MLflow experiment.
        run_name: The name of the specific run.
        parent_run_name: Optional. If provided, the run will be nested under this parent run.
        params: Dictionary of parameters to log.
        metrics: Dictionary of metrics to log.
        tags: Dictionary of key-value tags to log.
        artifacts: List of artifacts to log (e.g., file paths).
        model: The model object or identifier to log.
        model_flavor: The flavor/version of the model.
        profile: Tracking profile ("local" or "databricks").
    """

    # 1. Environment Setup
    runtime = "local"  # Placeholder for actual check_runtime_environment() result
    if runtime == "local":
        mlflow.set_tracking_uri("file://")  # Local run tracking
        mlflow.set_registry_uri("local-run")
    elif runtime == "databricks":
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri("databricks-uc")
    else:
        print(
            f"Warning: Unknown runtime environment '{runtime}'. MLflow not configured."
        )
        return

    client = MlflowClient()

    # 2. Experiment Management
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id

    # 3. Parent and Run Name Management (Nesting Logic)

    # 3-3. Handle parent_run_name (Search or Create Parent Run)
    parent_run_id = None
    if parent_run_name:
        parent_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(f"tags.mlflow.runName = '{parent_run_name}'"),
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        if parent_runs:
            parent_run_id = parent_runs[0].info.run_id
            print(f"Found existing parent run: {parent_run_id}")
        else:
            print(f"Parent run '{parent_run_name}' not found. Creating new parent run.")
            parent_run = mlflow.start_run(
                experiment_id=experiment_id, run_name=parent_run_name
            )
            parent_run_id = parent_run.info.run_id

    # 3-4. Handle run_name (Search or Create Run, respecting nesting)
    final_run_id = None
    if run_name:
        # Search for the run, optionally restricting search if a parent exists
        filter_string = f"tags.mlflow.runName = '{run_name}'"
        if parent_run_name:
            # If a parent exists, search within it
            filter_string += f" AND tags.mlflow.runId LIKE '%/{parent_run_name}/%'"

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )

        if runs:
            final_run_id = runs[0].info.run_id
            print(f"Found existing run: {final_run_id}")
        else:
            print(f"Run '{run_name}' not found. Starting new run.")
            final_run_id = mlflow.start_run(
                experiment_id=experiment_id, run_name=run_name
            ).info.run_id

    if not final_run_id:
        print("Error: Could not determine the final run ID. Exiting.")
        return

    run = mlflow.start_run(run_id=final_run_id)

    # 4. Tracking Values

    # 4-2. Log Experiment Name (already done by set_experiment, but we can log it explicitly)
    # 4-3. Log Parent/Run names as run tags for lineage
    if parent_run_name:
        run.set_tag("parent_run_name", parent_run_name)
    if run_name:
        run.set_tag("run_name", run_name)
        if parent_run_name:
            run.set_tag("parent_run_id", parent_run_id)

    # 4-5. Track parameters, metrics, tags, artifacts, model

    # Log Parameters
    if params:
        mlflow.log_params(params)

    # Log Metrics (using a placeholder structure for various metric types)
    if metrics:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    # Log Tags
    if tags:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    # Log Model
    if model:
        mlflow.log_param("model", model)
        if model_flavor:
            mlflow.set_tag("model_flavor", model_flavor)

    # Log Artifacts (Paths/Files)
    if artifacts:
        for artifact_path in artifacts:
            # Assuming artifact_path is a string file path
            try:
                # For file paths, log as a file URI
                mlflow.log_artifact(artifact_path)
                print(f"Logged artifact: {artifact_path}")
            except FileNotFoundError:
                print(f"Warning: Artifact not found at {artifact_path}. Skipping.")

    # 5. End Run
    mlflow.end_run()
    print(f"Successfully finished tracking run: {run.info.run_id}")


# Example usage stub (to be replaced by actual implementation in script)
# if __name__ == "__main__":
#     # tracking_utils.track_values(
#     #     experiment_name="TextClassification",
#     #     run_name="run_001",
#     #     parent_run_name=None,
#     #     params={"learning_rate": 0.001, "epochs": 10},
#     #     metrics={"accuracy": 0.95, "loss": 0.15},
#     #     tags={"source": "experiment_script"},
#     #     artifacts=["model.pkl"],
#     #     model="sklearn_model",
#     #     model_flavor="v1.0",
#     #     profile="databricks"
#     # )
