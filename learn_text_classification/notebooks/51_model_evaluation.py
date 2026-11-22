# Databricks notebook source

from learn_text_classification import config_utils, logging_utils
from learn_text_classification.tracking.models import (
    InferenceOutputs,
    TrackingConfig,
)

# COMMAND ----------
# runpy.run_path() の init_globals から受け取る。
logging_utils.setup_logging()

raw_inference_outputs = globals().get("inference_outputs")
raw_tracking = globals().get("tracking", {})
tracking_config = config_utils.validate(raw_tracking, TrackingConfig)

# COMMAND ----------
if not isinstance(raw_inference_outputs, InferenceOutputs):
    raise TypeError(f"Expected InferenceOutputs, got {type(raw_inference_outputs).__name__}.")

predictions = raw_inference_outputs.predictions

# COMMAND ----------
labels = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]

required_columns = [
    "Id",
    *labels,
    *[f"{label}_preds" for label in labels],
]

missing_columns = sorted(set(required_columns) - set(predictions.columns))

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
