from pyspark import pipelines as dp  # type: ignore
from pyspark.sql import functions as F


@dp.table
def wrime_silver():

    df = spark.read.table("wrime")  # type: ignore  # noqa: F821

    # Datetime
    df = df.withColumn("datetime", F.to_timestamp("Datetime"))

    # データセット区分
    df = df.withColumn("dataset_split", F.lower(F.col("Train/Dev/Test")))

    # 感情スコアを数値化
    emotion_groups = [
        "Writer",
        "Reader1",
        "Reader2",
        "Reader3",
        "Avg__Readers",
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

    for group in emotion_groups:
        for emotion in emotions:
            old_name = f"{group}_{emotion}"
            new_name = f"{group.lower()}_{emotion.lower()}"

            df = df.withColumn(new_name, F.col(old_name).cast("double"))

    return df.select(
        F.col("Sentence").alias("sentence"),
        F.col("UserID").alias("user_id"),
        F.col("datetime"),
        F.col("dataset_split"),
        "writer_joy",
        "writer_sadness",
        "writer_anticipation",
        "writer_surprise",
        "writer_anger",
        "writer_fear",
        "writer_disgust",
        "writer_trust",
        "reader1_joy",
        "reader1_sadness",
        "reader1_anticipation",
        "reader1_surprise",
        "reader1_anger",
        "reader1_fear",
        "reader1_disgust",
        "reader1_trust",
        "reader2_joy",
        "reader2_sadness",
        "reader2_anticipation",
        "reader2_surprise",
        "reader2_anger",
        "reader2_fear",
        "reader2_disgust",
        "reader2_trust",
        "reader3_joy",
        "reader3_sadness",
        "reader3_anticipation",
        "reader3_surprise",
        "reader3_anger",
        "reader3_fear",
        "reader3_disgust",
        "reader3_trust",
        "avg__readers_joy",
        "avg__readers_sadness",
        "avg__readers_anticipation",
        "avg__readers_surprise",
        "avg__readers_anger",
        "avg__readers_fear",
        "avg__readers_disgust",
        "avg__readers_trust",
    )
