# Databricks notebook source
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import runpy
from typing import Any

from loguru import logger
from omegaconf import MISSING

from learn_text_classification import config_utils, job_utils, logging_utils
from learn_text_classification.tracking.models import TrackingConfig, TrainerOutputs
from learn_text_classification.tracking.tracking_utils import MLflowTracker


# COMMAND ----------
@dataclass
class RunConfig:
    run: str = MISSING
    params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TrainingConfig:
    runs: dict[str, RunConfig]

    def get_run(self, key: str) -> RunConfig:
        return self.runs[key]


# COMMAND ----------
experiment_key = job_utils.get_job_option("experiment_key", "baseline")
run_name = job_utils.get_job_option("run_name").strip()

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")
training_config = config_utils.validate(config["training"], TrainingConfig)
tracking_config = config_utils.validate(config.get("tracking", {}), TrackingConfig)

# COMMAND ----------
run_config = training_config.get_run(experiment_key)
if run_config is None:
    raise ValueError(f"Experiment {experiment_key!r} not found")

# COMMAND ----------
# Tracking設定
tracker = MLflowTracker(
    experiment_name=tracking_config.experiment_path,
)

# Parent Run開始
if not run_name:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{experiment_key}_{timestamp}"

parent_run_id = tracker.start_parent_run(
    run_name=run_name,
    tags={
        "experiment_key": experiment_key,
    },
)

# Trainer/Tunerに渡すTrackingConfig
tracking_config = replace(
    tracking_config,
    parent_run_id=parent_run_id,
)

# COMMAND ----------
try:
    logger.info("Running experiment: {}", experiment_key)
    logger.info("Target notebook: {}", run_config.run)

    namespace = runpy.run_path(
        run_config.run,
        init_globals={
            "parameters": run_config.params,
            "tracking": tracking_config,
        },
    )

    outputs = namespace.get("outputs")
    if not isinstance(outputs, TrainerOutputs):
        raise TypeError(f"Expected TrainerOutputs, got {type(outputs).__name__}.")

except Exception:
    logger.exception("Experiment failed: {}", experiment_key)

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
