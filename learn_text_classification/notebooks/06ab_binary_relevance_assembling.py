# Databricks notebook source
from typing import cast

from databricks.connect import DatabricksSession
from loguru import logger
import mlflow
from pyspark.ml import Model
from pyspark.ml.classification import LogisticRegressionModel

from learn_text_classification import config_utils, logging_utils
from learn_text_classification.job_utils import check_runtime_environment
from learn_text_classification.spark_utils.inference_model import BinaryRelevanceInferenceModel
from learn_text_classification.tracking import run_utils

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
if check_runtime_environment() == "local":
    DATABRICKS_PROFILE = "sandbox"
    mlflow.set_tracking_uri(f"databricks://{DATABRICKS_PROFILE}")
    mlflow.set_registry_uri(f"databricks-uc://{DATABRICKS_PROFILE}")

# COMMAND ----------
experiment_name = "/Users/shotaro.minami8e40a5@gmail.com/.mlflow/learn_text_classification"
model_id = "logistic_regression"

# COMMAND ----------
runs = run_utils.search_finished_runs_by_model_id(config.assembling.experiment_name, config.assembling.model_id)
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
inference.save(config.assembling.save_path, spark)
