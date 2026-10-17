# Databricks notebook source
from urllib.request import urlopen

from loguru import logger

from learn_text_classification import config_utils, logging_utils

logging_utils.setup_logging()
config = config_utils.load_config("./config")


# COMMAND ----------

output_dir = config.ingest.volume
output_path = output_dir / "wrime-ver2.tsv"
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"{output_dir.exists() = }")

# COMMAND ----------

with urlopen(config.ingest.url, timeout=60) as response, output_path.open("wb") as f:
    f.write(response.read())

logger.info("Downloaded: {}", output_path)
logger.debug("File size: {} bytes", output_path.stat().st_size)

# COMMAND ----------
