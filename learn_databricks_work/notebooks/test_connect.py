# Databricks notebook source
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()
df = spark.read.table("sandbox.information_schema.catalogs")
display(df)
