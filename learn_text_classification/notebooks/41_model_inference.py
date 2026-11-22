# Databricks notebook source
from dataclasses import dataclass

from omegaconf import MISSING

from learn_text_classification import config_utils, logging_utils
from learn_text_classification.tracking.models import TrackingConfig


# COMMAND ----------
@dataclass
class InferenceConfig:
    model_uri: str = MISSING
    input: str = MISSING


# COMMAND ----------
logging_utils.setup_logging()

config = config_utils.load_config("./config")
inference_config = config_utils.validate(config["inference"], InferenceConfig)
tracking_config = config_utils.validate(config.get("tracking", {}), TrackingConfig)
