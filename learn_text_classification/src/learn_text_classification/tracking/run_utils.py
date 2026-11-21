import mlflow
import pandas as pd


def search_finished_runs_by_model_id(
    experiment_name: str,
    model_id: str,
) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(f"tags.model_id = '{model_id}' and attributes.status = 'FINISHED'"),
        order_by=["attributes.start_time DESC"],
    )

    if not isinstance(runs, pd.DataFrame):
        raise TypeError(f"Unexpected type: {type(runs)}")

    return runs
