# Databricks notebook source
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import uuid

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from omegaconf import OmegaConf
import pandas as pd
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import array_to_vector
from pyspark.sql import functions as F

from learn_text_classification import job_utils, tracking_utils


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

    profile: str = "sandbox"
    log_volume: Path = Path("/Volumes/sandbox/shotaro_minami8e40a5/learn_text_classification/.mlflow/")
    tmp_volume: Path = Path("/Volumes/sandbox/shotaro_minami8e40a5/learn_text_classification/.tmp/")
    tmp_path: Path = Path("./tmp")
    experiment_name: str = "learn_text_classification"
    run_name: str = "logistic_regression"
    child_run_name: str | None = None
    log_model: bool = True  # False


# COMMAND ----------
params = OmegaConf.structured(LogisticRegressionParams)
if "parameters" in globals():
    _ov: dict[str, Any] = globals()["parameters"]
    params = OmegaConf.merge(params, _ov)
params = cast(LogisticRegressionParams, params)

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
class SparkLogisticRegressionPyFunc(PythonModel):
    def load_context(
        self,
        context: PythonModelContext,
    ) -> None:
        self.spark = DatabricksSession.builder.getOrCreate()
        artifact_path = str(context.artifacts["spark_model"])  # type: ignore
        self.model = LogisticRegressionModel.load(artifact_path)

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> Any:
        spark_df = self.spark.createDataFrame(model_input[["features"]])

        spark_df = spark_df.withColumn(
            "features",
            array_to_vector(F.col("features")),
        )

        predictions = self.model.transform(spark_df)

        return predictions.select("preds").toPandas()


# COMMAND ----------
additional_args: dict[str, Any] = {}
if params.child_run_name is not None:
    additional_args["child_run_name"] = params.child_run_name

if params.log_model:
    random_suffix = uuid.uuid4().hex
    dir_name = "spark_logistic_regression_" + params.label_col + "_" + random_suffix
    spark_model_path = str(params.tmp_volume / dir_name)
    model.write().overwrite().save(spark_model_path)

    w = WorkspaceClient(profile=params.profile)
    local_model_path = Path(params.tmp_path) / dir_name
    job_utils.download_volume_directory(w, spark_model_path, local_model_path)

    additional_args.update(
        {
            "model": SparkLogisticRegressionPyFunc(),
            "model_flavor": "pyfunc",
            "model_artifacts": {
                "spark_model": "./tmp",
            },
        }
    )

tracking_utils.track_values(
    params.log_volume,
    params.experiment_name,
    params.run_name,
    params={
        "regularization": params.regularization,
        "elastic_net": params.elastic_net,
        "max_itr": params.max_itr,
    },
    metrics={"auc": auc},
    tags={"label": params.label_col},
    **additional_args,
)

# COMMAND ----------
outputs = {"score": auc}
logger.info(f"{outputs = }")

# COMMAND ----------
