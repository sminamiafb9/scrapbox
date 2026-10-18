# Databricks notebook source
from databricks.connect import DatabricksSession
from loguru import logger
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import seaborn as sns

from learn_text_classification import config_utils, logging_utils

logging_utils.setup_logging()
config = config_utils.load_config("./config")

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

df = spark.table(config.prepare.dataset)
logging_utils.show_df(df, level="DEBUG")

# COMMAND ----------
labels = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]

long_df = df.unpivot(
    ids=["Sentence", "Split"],
    values=[F.col(label) for label in labels],
    variableColumnName="Emotion",
    valueColumnName="Intensity",
)

# COMMAND ----------

frequency_table = (
    long_df.groupBy("Split", "Emotion", "Intensity")
    .agg(F.count("*"))
    .orderBy("Split", "Emotion", "Intensity")
).toPandas()

frequency_table["total"] = frequency_table.groupby(["Split", "Emotion"])[
    "count(1)"
].transform("sum")
frequency_table["ratio"] = frequency_table["count(1)"] / frequency_table["total"]
frequency_table = frequency_table.drop(columns=["total"])

logger.debug(frequency_table.head(10))

# COMMAND ----------
pivot_df = frequency_table.pivot_table(
    index=["Emotion", "Split"],
    columns="Intensity",
    values="count(1)",
)
pivot_df.style.format("{:.3f}").background_gradient(cmap="Blues", axis=1)

# COMMAND ----------
g = sns.FacetGrid(
    frequency_table,
    col="Emotion",
    col_wrap=3,
    height=3.5,
    aspect=1.3,
    sharey=True,
)

g.map_dataframe(sns.barplot, x="Split", y="ratio", hue="Intensity", palette="Set2")
handles, labels = g.axes[0].get_legend_handles_labels()
g.figure.legend(
    handles,
    labels,
    title="Intensity",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    frameon=True,
)
g.set_axis_labels("Split", "Ratio")
g.figure.subplots_adjust(top=0.9, hspace=0.4)
g.figure.suptitle("Emotion Intensity Ratio Comparison Across Splits", fontsize=16)

plt.show()

# COMMAND ----------
