# Databricks notebook source
from learn_databricks_work import main

main()

df = spark.range(10)

display(df)
