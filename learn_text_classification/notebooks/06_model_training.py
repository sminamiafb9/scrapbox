# Databricks notebook source
from dataclasses import replace
import runpy

from databricks.sdk.runtime import dbutils
from loguru import logger

from learn_text_classification import config_utils, logging_utils
from learn_text_classification.tracking.models import TrainerOutputs
from learn_text_classification.tracking.tracking_utils import MLflowTracker

# COMMAND ----------
dbutils.widgets.text("experiment_name", "baseline")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")

# COMMAND ----------
experiment = config.get_experiment(experiment_name)
if experiment is None:
    raise ValueError(f"Experiment {experiment_name!r} not found")

# COMMAND ----------
# Tracking設定
tracking_config = config.tracking
tracker = MLflowTracker(
    experiment_name=tracking_config.experiment_path,
)

# Parent Run開始
parent_run_id = tracker.start_parent_run(
    run_name=experiment_name,
    tags={
        "experiment_name": experiment_name,
    },
)

# Trainer/Tunerに渡すTrackingConfig
tracking_config = replace(
    tracking_config,
    parent_run_id=parent_run_id,
)

# COMMAND ----------
try:
    logger.info("Running experiment: {}", experiment_name)
    logger.info("Target notebook: {}", experiment.run)

    namespace = runpy.run_path(
        experiment.run,
        init_globals={
            "parameters": experiment.params,
            "tracking": tracking_config,
        },
    )

    outputs = namespace.get("outputs")
    if not isinstance(outputs, TrainerOutputs):
        raise TypeError(f"Expected TrainerOutputs, got {type(outputs).__name__}.")

except Exception:
    logger.exception("Experiment failed: {}", experiment_name)

    tracker.end_parent_run(
        run_id=parent_run_id,
        status="FAILED",
    )
    raise

else:
    tracker.end_parent_run(
        run_id=parent_run_id,
        outputs=outputs,
    )
