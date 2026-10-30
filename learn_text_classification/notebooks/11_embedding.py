# Databricks notebook source
from dataclasses import dataclass
import math
from pathlib import Path
import uuid

from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.ml.functions import array_to_vector
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructField,
    StructType,
)
from sentence_transformers import SentenceTransformer

from learn_text_classification import config_utils, job_utils, logging_utils

# COMMAND ----------
dbutils.widgets.text("experiment_name", "baseline")
experiment_name = dbutils.widgets.get("experiment_name")

dbutils.widgets.text("step_name", "embedding")
step_name = dbutils.widgets.get("step_name")


# COMMAND ----------
@dataclass
class EmbeddingConfig:
    cache_folder: Path
    batch_size: int
    transfer_batch_size: int

    embedding_model_name: str
    dataset: str


# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")
embedding_config = config_utils.resolve_step_config(
    config,
    experiment_name,
    step_name,
    EmbeddingConfig,
)

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
job_utils.exit_if_table_exists(embedding_config.params.dataset)

# COMMAND ----------
df = spark.read.table(config.preprocessing.dataset)
logging_utils.show_df(df, level="DEBUG")

# COMMAND ----------
embedding_config.params.cache_folder.mkdir(parents=True, exist_ok=True)

# COMMAND ----------
embedding_model = SentenceTransformer(
    embedding_config.params.embedding_model_name,
    cache_folder=str(embedding_config.params.cache_folder),
)

schema = StructType(
    [
        StructField("Id", StringType(), False),
        StructField("Embedding", ArrayType(FloatType()), True),
    ]
)

temporary_prefix = uuid.uuid4().hex
temporary_table = f"{embedding_config.params.dataset}_{temporary_prefix}"

empty_df = spark.createDataFrame([], schema)
empty_df.write.format("delta").mode("overwrite").saveAsTable(temporary_table)

# COMMAND ----------
all_count = df.count()
batch_size = embedding_config.params.batch_size
transfer_batch_size = embedding_config.params.transfer_batch_size
batch_count = max(1, math.ceil(all_count / transfer_batch_size))

logger.info(
    "Starting embedding: total={}, transfer_batch_size={}, "
    "embedding_batch_size={}, batch_count={}",
    all_count,
    transfer_batch_size,
    batch_size,
    batch_count,
)

# COMMAND ----------
partitioned_df = df.withColumn(
    "_batch_id",
    F.pmod(
        F.xxhash64(F.col("Id")),
        F.lit(batch_count),
    ),
)
logging_utils.show_df(partitioned_df, level="DEBUG")

# COMMAND ----------
done_count = 0
for batch_id in range(batch_count):
    batch = (
        partitioned_df.filter(F.col("_batch_id") == batch_id)
        .drop("_batch_id")
        .collect()
    )
    if not batch:
        break

    ids = [row["Id"] for row in batch]
    texts = [row["Sentence"] if row["Sentence"] is not None else "" for row in batch]

    for start in range(0, len(batch), batch_size):
        end = start + batch_size

        chunk_ids = ids[start:end]
        chunk_texts = texts[start:end]

        embeddings = embedding_model.encode(  # type: ignore
            chunk_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        rows = [
            (id_, embedding.tolist())
            for id_, embedding in zip(
                chunk_ids,
                embeddings,  # type: ignore
                strict=False,
            )
        ]

        batch_df = spark.createDataFrame(rows, schema=schema)
        batch_df.write.format("delta").mode("append").saveAsTable(temporary_table)
        done_count += len(chunk_ids)

    logger.info(
        "Processed: {}/{} (batch {}/{}, batch_size={})",
        done_count,
        all_count,
        batch_id + 1,
        batch_count,
        len(batch),
    )

logger.info("Embedding completed: {}/{}", done_count, all_count)

# COMMAND ----------
embeddings = spark.read.table(temporary_table)
logging_utils.show_df(embeddings, level="DEBUG")

# COMMAND ----------
datasets = (
    df.join(embeddings, on="Id", how="inner")
    .withColumn("features", array_to_vector(F.col("Embedding")))
    .drop(F.col("Embedding"))
)

(
    datasets.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(embedding_config.params.dataset)
)

loaded = spark.read.table(embedding_config.params.dataset)
loaded.printSchema()
logging_utils.show_df(loaded, level="INFO")

# COMMAND ----------
spark.sql(f"drop table if exists {temporary_table}")
