# Databricks notebook source
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

from learn_text_classification import config_utils, logging_utils

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
raw_data = (
    spark.read.format("csv")
    .option("header", True)
    .option("sep", "\t")
    .option("inferSchema", True)
    .load(str(config.prepare.raw_data))
)
logging_utils.show_df(raw_data, level="DEBUG")

# COMMAND ----------
metadata_cols = [
    "Sentence",
]
emotions = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]

reader_cols = [f"Avg. Readers_{emotion}" for emotion in emotions] + [
    "Avg. Readers_Sentiment"
]

df_simple = raw_data.select(
    F.expr("uuid()").alias("Id"),
    *[F.col(f"`{col}`") for col in metadata_cols],
    F.when(F.col("`Train/Dev/Test`").isNull(), F.lit("train"))
    .otherwise(F.col("`Train/Dev/Test`"))
    .alias("Split"),
    *[F.col(f"`Avg. Readers_{emotion}`").alias(emotion) for emotion in emotions],
)

df_simple.printSchema()
logging_utils.show_df(df_simple, level="DEBUG")

# COMMAND ----------
(
    df_simple.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(config.prepare.dataset)
)

loaded = spark.read.table(config.prepare.dataset)
loaded.printSchema()
logging_utils.show_df(loaded, level="INFO")
