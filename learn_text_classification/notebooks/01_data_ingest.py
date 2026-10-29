# Databricks notebook source
from urllib.request import urlopen

from loguru import logger

from learn_text_classification import config_utils, job_utils, logging_utils

# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")

# COMMAND ----------
output_dir = config.ingest.volume
output_path = output_dir / "wrime-ver2.tsv"
job_utils.exit_if_file_exists(output_path)

# COMMAND ----------
output_dir.mkdir(parents=True, exist_ok=True)
with urlopen(config.ingest.url, timeout=60) as response, output_path.open("wb") as f:
    f.write(response.read())

logger.info("Downloaded: {}", output_path)
logger.debug("File size: {} bytes", output_path.stat().st_size)
