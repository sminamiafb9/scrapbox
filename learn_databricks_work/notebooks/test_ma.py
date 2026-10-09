# Databricks notebook source
from databricks.connect import DatabricksSession
from pyspark.sql.functions import expr

spark = DatabricksSession.builder.getOrCreate()

# 3件の日本語サンプル
samples = [
    ("今日は東京で日本語を勉強します。",),
    ("昨日、本を読んだ。",),
    ("形態素解析を使って単語の基本形を調べます。",),
]

df = spark.createDataFrame(samples, schema=["text"])

display(df)

# COMMAND ----------

# Unity Catalog UDF を呼び出す
res = df.withColumn("tokens", expr("sandbox.shotaro_minami8e40a5.ma(text)"))

display(res)
