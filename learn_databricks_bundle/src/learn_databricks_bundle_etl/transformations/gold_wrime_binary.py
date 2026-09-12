from pyspark import pipelines as dp  # type: ignore
from pyspark.sql import functions as F


@dp.table
def wrime_gold():
    df = spark.read.table("wrime_silver")  # type: ignore # noqa: F821

    return df.select(
        "sentence",
        "user_id",
        "datetime",
        "dataset_split",
        (F.col("avg__readers_joy") >= 2).cast("int").alias("joy"),
        (F.col("avg__readers_sadness") >= 2).cast("int").alias("sadness"),
        (F.col("avg__readers_anticipation") >= 2).cast("int").alias("anticipation"),
        (F.col("avg__readers_surprise") >= 2).cast("int").alias("surprise"),
        (F.col("avg__readers_anger") >= 2).cast("int").alias("anger"),
        (F.col("avg__readers_fear") >= 2).cast("int").alias("fear"),
        (F.col("avg__readers_disgust") >= 2).cast("int").alias("disgust"),
        (F.col("avg__readers_trust") >= 2).cast("int").alias("trust"),
    )
