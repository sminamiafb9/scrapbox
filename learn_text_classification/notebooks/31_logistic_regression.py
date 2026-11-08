# Databricks notebook source
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from databricks.connect import DatabricksSession
from loguru import logger
from omegaconf import OmegaConf
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from learn_text_classification.tracking import (
    ModelArtifact,
    TrackingConfig,
    TrackingRecord,
    TrainerOutputs,
)


# COMMAND ----------
@dataclass
class LogisticRegressionParams:
    dataset: str = "sandbox.shotaro_minami8e40a5.wrime_embeddings"
    label_col: str = "Joy"
    features_col: str = "features"
    regularization: float = 0.01
    elastic_net: float = 0.3
    max_itr: int = 1000
    metric_name: Literal["areaUnderROC", "areaUnderPR"] = "areaUnderROC"


# COMMAND ----------
# 入力変数
# runpy.run_path() の init_globals から受け取る。
# 未指定の場合はインタラクティブ実行用の既定値を使用する。

raw_parameters = globals().get("parameters", {})
params = OmegaConf.to_object(
    OmegaConf.merge(
        OmegaConf.structured(LogisticRegressionParams()),
        raw_parameters,
    )
)
params = cast(LogisticRegressionParams, params)

raw_tracking = globals().get("tracking", {})
config = OmegaConf.to_object(
    OmegaConf.merge(
        OmegaConf.structured(TrackingConfig()),
        raw_tracking,
    )
)
config = cast(TrackingConfig, config)


# COMMAND ----------
record = TrackingRecord(
    params={
        "dataset": params.dataset,
        "label_col": params.label_col,
        "features_col": params.features_col,
        "regularization": params.regularization,
        "elastic_net": params.elastic_net,
        "max_itr": params.max_itr,
    },
    tags={
        "label": params.label_col,
    },
)

model_artifact: ModelArtifact | None = None

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
    metricName=params.metric_name,
)

auc = float(evaluator.evaluate(predictions))

record.metrics["auc"] = auc

# COMMAND ----------
if config.save_model:
    if not config.parent_run_id:
        raise ValueError("parent_run_id is required when save_model=True.")

    if Path(config.parent_run_id).name != config.parent_run_id or config.parent_run_id in {".", ".."}:
        raise ValueError("parent_run_id must be a single path component.")

    # tmp_volumeと同じVolume直下にmodelsディレクトリを作る。
    if config.volume_base is None:
        raise ValueError("volume_base is required when save_model=True.")
    spark_model_path = Path(config.volume_base) / "models" / config.parent_run_id / "spark_model"

    model.write().overwrite().save(str(spark_model_path))
    logger.info(f"Saved model to {spark_model_path = }")

    model_artifact = ModelArtifact(
        uri=str(spark_model_path),
        format="spark_ml",
        flavor="pyspark.ml",
        metadata={
            "storage": "unity_catalog_volume",
            "model_type": "logistic_regression",
        },
    )
else:
    logger.info("Skipping saving model")

# COMMAND ----------
outputs = TrainerOutputs(
    status="SUCCEEDED",
    tracking=record,
    model=model_artifact,
)

# COMMAND ----------
logger.info(f"{outputs.to_dict() = }")
