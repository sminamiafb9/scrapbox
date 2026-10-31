from pathlib import Path
import sys
from typing import Literal, TypeGuard, get_args

from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


RUNTIME_ENVIRONMENT_KEY = "runtime_environment"

Runtime = Literal["local", "databricks"]
RUNTIME_ORDER = get_args(Runtime)
_runtime: Runtime | None = None


def is_valid_runtime(value: str) -> TypeGuard[Runtime]:
    return value in RUNTIME_ORDER


def check_runtime_environment():
    global _runtime
    if _runtime is not None:
        return _runtime

    dbutils.widgets.text(RUNTIME_ENVIRONMENT_KEY, RUNTIME_ORDER[0])
    runtime = dbutils.widgets.get(RUNTIME_ENVIRONMENT_KEY)
    if not is_valid_runtime(runtime):
        raise ValueError(f"Invalid runtime environment: {runtime}")
    logger.info("Runtime environment: {runtime}")
    _runtime = runtime

    return _runtime


def exit_if_table_exists(table_name: str, return_value: str = "") -> None:
    exists = spark.catalog.tableExists(table_name)
    if exists:
        logger.info("Table {} already exists. Skipping.", table_name)

        if check_runtime_environment() == "databricks":
            dbutils.notebook.exit(return_value)  # type: ignore
        else:
            sys.exit(0)


def exit_if_file_exists(file_path: str | Path, return_value: str = "") -> None:
    file_path = Path(file_path)
    if file_path.exists():
        logger.info("File {} already exists. Skipping.", file_path)

        if check_runtime_environment() == "databricks":
            dbutils.notebook.exit(return_value)  # type: ignore
        else:
            sys.exit(0)
