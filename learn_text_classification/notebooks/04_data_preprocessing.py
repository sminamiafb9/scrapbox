# Databricks notebook source
from databricks.connect import DatabricksSession
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import seaborn as sns

from learn_text_classification import config_utils, logging_utils

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
df = spark.table(config.prepare.dataset)
logging_utils.show_df(df, level="DEBUG")

# COMMAND ----------
label_cols = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]

df = df.withColumns(
    {
        col: F.when(F.col(col) >= 1, 1)
        .when(F.col(col) == 0, 0)
        .otherwise(None)
        .cast("int")
        for col in label_cols
    }
)

df.printSchema()
logging_utils.show_df(df, level="DEBUG")

# COMMAND ----------
ratio_df = df.groupBy("Split").agg(*[F.avg(F.col(c)).alias(c) for c in label_cols])
logging_utils.show_df(ratio_df, level="DEBUG")

# COMMAND ----------
plot_df = ratio_df.toPandas().set_index("Split")[label_cols] * 100

plt.figure(figsize=(12, 4))
sns.heatmap(
    plot_df,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=0,
    vmax=100,
)

plt.title("Positive Label Ratio by Split (%)")
plt.xlabel("Emotion")
plt.ylabel("Split")
plt.tight_layout()
plt.show()

# COMMAND ----------
(
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(config.preprocessing.dataset)
)

loaded = spark.read.table(config.preprocessing.dataset)
loaded.printSchema()
logging_utils.show_df(loaded, level="INFO")
