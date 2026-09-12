import dlt
from pyspark.sql import functions as F

POS_COLS = ["joy", "anticipation", "trust"]
NEG_COLS = ["sadness", "anger", "fear", "disgust"]
THRESHOLD = 1  # 有意な感情とみなす強度の閾値


@dlt.table(
    name="wrime_gold_sentiment_3class",
    comment="wrime_goldから作成した3値分類（0:Negative, 1:Neutral, 2:Positive）用の派生テーブル",
    table_properties={"quality": "gold"},
)
def wrime_gold_sentiment_3class():
    df_gold = dlt.read("wrime_gold")

    df_scored = df_gold.withColumn("max_pos_score", F.greatest(*[F.col(c) for c in POS_COLS])).withColumn(
        "max_neg_score", F.greatest(*[F.col(c) for c in NEG_COLS])
    )

    label_expr = (
        F.when(
            (F.col("max_pos_score") < THRESHOLD) & (F.col("max_neg_score") < THRESHOLD),
            1,  # いずれの感情も弱いため Neutral
        )
        .when(F.col("max_pos_score") > F.col("max_neg_score"), 2)  # Positive
        .when(F.col("max_neg_score") > F.col("max_pos_score"), 0)  # Negative
        .otherwise(1)  # 判定不能・スコア拮抗時は Neutral
    )

    return df_scored.withColumn("label_3class", label_expr)
