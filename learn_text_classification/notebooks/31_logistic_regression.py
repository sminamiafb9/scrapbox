# Databricks notebook source
from dataclasses import dataclass
from typing import Any, cast

from databricks.connect import DatabricksSession
from loguru import logger
from omegaconf import OmegaConf
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F


# COMMAND ----------
@dataclass
class LogisticRegressionParams:
    dataset: str = "sandbox.shotaro_minami8e40a5.wrime_embeddings"
    label_col: str = "Joy"
    features_col: str = "features"

    regularization: float = 0.01
    elastic_net: float = 0.3
    max_itr: int = 1000

    metric_name: str = "areaUnderROC"


# COMMAND ----------
params = OmegaConf.structured(LogisticRegressionParams)
if "parameters" in globals():
    _ov: dict[str, Any] = globals()["parameters"]
    params = OmegaConf.merge(params, _ov)

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
datasets = spark.read.table(params.dataset)
train_df = datasets.filter(F.col("Split") == "train")
dev_df = datasets.filter(F.col("Split") == "dev")

# COMMAND ----------
lr = LogisticRegression(
    featuresCol=params.features_col,
    labelCol=params.label_col,
    regParam=params.regularization,
    elasticNetParam=params.elastic_net,
    maxIter=params.max_itr,
    standardization=True,
    predictionCol="preds",
    probabilityCol="probs",
    rawPredictionCol="scores",
)
model = cast(LogisticRegressionModel, lr.fit(train_df))
predictions = model.transform(dev_df)
evaluator = BinaryClassificationEvaluator(
    labelCol=params.label_col,
    rawPredictionCol="scores",
    metricName=params.metric_name,  # type: ignore
)
auc = evaluator.evaluate(predictions, {evaluator.labelCol: params.label_col})

# COMMAND ----------
outputs = {"score": auc}
logger.info(f"{outputs = }")

# 検証
# # COMMAND ----------
# import mlflow
#
# profile = "sandbox"
# mlflow.set_tracking_uri(f"databricks://{profile}")
# mlflow.set_registry_uri("databricks-uc")
#
# mlflow.set_experiment("/test-mlflow")
# with mlflow.start_run(run_name="sample2"):
#     params = {"hoge": 1, "piyo": 2}
#     mlflow.log_params(params)
#
#     metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.91, "f1_score": 0.92}
#     mlflow.log_metrics(metrics)
#     mlflow.set_tag("test-tag", "test-value")
#     print("MLflowへの記録が完了しました")
#
# # COMMAND ----------
#
