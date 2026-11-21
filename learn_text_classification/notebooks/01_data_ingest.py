# Databricks notebook source
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from loguru import logger
from omegaconf import MISSING

from learn_text_classification import config_utils, job_utils, logging_utils


# COMMAND ----------
@dataclass
class IngestConfig:
    url: str = MISSING
    volume: Path = MISSING


# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config_by_key("./config", IngestConfig, "ingest")

# COMMAND ----------
output_dir = config.volume
output_path = output_dir / "wrime-ver2.tsv"
job_utils.exit_if_file_exists(output_path)

# COMMAND ----------
output_dir.mkdir(parents=True, exist_ok=True)
with urlopen(config.url, timeout=60) as response, output_path.open("wb") as f:
    f.write(response.read())

logger.info("Downloaded: {}", output_path)
logger.debug("File size: {} bytes", output_path.stat().st_size)
