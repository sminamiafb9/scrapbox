# Databricks notebook source

from databricks.connect import DatabricksSession
from pyspark.ml.classification import LogisticRegressionModel

from learn_text_classification.spark_utils.inference_model import BinaryRelevanceInferenceModel

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()
df = spark.read.table("sandbox.shotaro_minami8e40a5.wrime_embeddings")

print(df)

# COMMAND ----------
path = "/Volumes/sandbox/shotaro_minami8e40a5/learn_text_classification/models/test/"

inference_model = BinaryRelevanceInferenceModel.load(
    path,
    spark,
    LogisticRegressionModel.load,
)

# COMMAND ----------
scores = inference_model.transform(df)
scores.show()
