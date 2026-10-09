# Databricks notebook source
from databricks.connect import DatabricksSession
from pyspark.sql.functions import expr

spark = DatabricksSession.builder.getOrCreate()
df = spark.createDataFrame(["hoge", "piyo"], schema=["name"])
display(df)

# COMMAND ----------

res = df.withColumn("greet", expr("sandbox.shotaro_minami8e40a5.greet(name)"))
display(res)
