# Databricks notebook source
from learn_text_classification import config_utils, logging_utils
from learn_text_classification.tracking.models import TrackingConfig

# COMMAND ----------
logging_utils.setup_logging()

config = config_utils.load_config("./config")
tracking_config = config_utils.validate(config.get("tracking", {}), TrackingConfig)
