# Databricks notebook source
from dataclasses import dataclass

from databricks.connect import DatabricksSession
from omegaconf import MISSING

from learn_text_classification import config_utils, logging_utils
from learn_text_classification.spark_utils.model_loader import load_model
from learn_text_classification.tracking.models import InferenceOutputs, TrackingConfig


# COMMAND ----------
@dataclass
class InferenceConfig:
    model_uri: str = MISSING
    input: str = MISSING


# COMMAND ----------
# 設定ファイルから Inference 固有の設定を読み込む。
logging_utils.setup_logging()

config = config_utils.load_config("./config")
inference_config = config_utils.validate(config["inference"], InferenceConfig)

# runpy.run_path() の init_globals から受け取る。
raw_tracking = globals().get("tracking", {})
tracking_config = config_utils.validate(raw_tracking, TrackingConfig)

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
df = spark.read.table(inference_config.input)

# COMMAND ----------
model = load_model(inference_config.model_uri, spark)
predictions = model.transform(df)

# COMMAND ----------
inference_outputs = InferenceOutputs(
    predictions=predictions,
)
