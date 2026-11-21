# Databricks notebook source
from dataclasses import dataclass
from typing import cast

from databricks.connect import DatabricksSession
from loguru import logger
import mlflow
from omegaconf import MISSING
from pyspark.ml import Model
from pyspark.ml.classification import LogisticRegressionModel

from learn_text_classification import config_utils, logging_utils
from learn_text_classification.job_utils import check_runtime_environment
from learn_text_classification.spark_utils.inference_model import BinaryRelevanceInferenceModel
from learn_text_classification.tracking import run_utils


# COMMAND ----------
@dataclass
class BinaryRelevanceAssemblingConfig:
    experiment_name: str = MISSING
    model_id: str = MISSING
    save_path: str = MISSING


# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config_by_key("./config", BinaryRelevanceAssemblingConfig, "assembling")

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
if check_runtime_environment() == "local":
    DATABRICKS_PROFILE = "sandbox"
    mlflow.set_tracking_uri(f"databricks://{DATABRICKS_PROFILE}")
    mlflow.set_registry_uri(f"databricks-uc://{DATABRICKS_PROFILE}")

# COMMAND ----------
runs = run_utils.search_finished_runs_by_model_id(config.experiment_name, config.model_id)
logger.info("Search Results:")
logger.info(run_utils)

# COMMAND ----------
models: dict[str, Model] = {}
for _, row in runs.iterrows():
    label = row["tags.label"]
    model_path = row["tags.model_volume_path"]
    models[label] = cast(Model, LogisticRegressionModel.load(model_path))

inference = BinaryRelevanceInferenceModel(models=models)

# COMMAND ----------
inference.save(config.save_path, spark)
