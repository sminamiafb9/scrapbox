# Databricks notebook source
from databricks.sdk.runtime import dbutils

from learn_text_classification import config_utils, logging_utils

# COMMAND ----------
dbutils.widgets.text("experiment_name", "baseline")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")

# COMMAND ----------
experiment = config.get_experiment(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment {experiment_name} not found")

# COMMAND ----------
for step_name, step_config in experiment.steps.items():
    run_script = step_config.run
    timeout_sec = 60 * 60 * 1
    dbutils.notebook.run(
        run_script,  # type: ignore
        timeout_sec,
        {
            "experiment_name": experiment_name,
            "step_name": step_name,
        },
    )
