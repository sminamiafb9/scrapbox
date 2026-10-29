# Databricks notebook source
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from loguru import logger
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
    embedding_batch_size: int
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
empty_df = spark.createDataFrame([], schema)
empty_df.write.format("delta").mode("overwrite").saveAsTable(
    embedding_config.params.dataset
)

# COMMAND ----------
all_count = df.count()
iterator = df.toLocalIterator()

done_count = 0
batch_size = embedding_config.params.embedding_batch_size
while True:
    batch = list(islice(iterator, batch_size))
    if not batch:
        break

    ids = [row["Id"] for row in batch]
    texts = [row["Sentence"] if row["Sentence"] is not None else "" for row in batch]

    embeddings = embedding_model.encode(  # type: ignore
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    rows = [
        (id_, embedding.tolist())
        for id_, embedding in zip(ids, embeddings, strict=False)  # type: ignore
    ]
    batch_df = spark.createDataFrame(rows, schema=schema)

    batch_df.write.format("delta").mode("append").saveAsTable(
        embedding_config.params.dataset
    )

    done_count += len(batch)
    logger.info("Processed: {}/{}", done_count, all_count)
