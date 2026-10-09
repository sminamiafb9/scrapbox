# Databricks notebook source

from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()

df = spark.createDataFrame(
    ["hoge", "piyo", "fuga", "foo", "bar"],
    schema="name string",
)
display(df)
# COMMAND ----------
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


@udf(returnType=StringType())
def greet(name: str) -> str:
    return f"Hello {name}"


df_with_udf = df.withColumn(
    "greeting",
    greet(col("name")),
)
display(df_with_udf)
