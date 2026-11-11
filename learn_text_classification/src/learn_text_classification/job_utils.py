import contextlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Literal, TypeGuard, get_args

from databricks.sdk import WorkspaceClient
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


def check_runtime_environment() -> Runtime:
    global _runtime
    if _runtime is not None:
        return _runtime

    dbutils.widgets.text(RUNTIME_ENVIRONMENT_KEY, RUNTIME_ORDER[0])
    runtime = dbutils.widgets.get(RUNTIME_ENVIRONMENT_KEY)
    if not is_valid_runtime(runtime):
        raise ValueError(f"Invalid runtime environment: {runtime}")
    logger.info(f"Runtime environment: {runtime}")
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


def download_volume_directory(
    workspace_client: WorkspaceClient,
    volume_path: str,
    local_path: Path,
) -> None:
    volume_path = volume_path.rstrip("/")
    local_path.mkdir(parents=True, exist_ok=True)

    for entry in workspace_client.files.list_directory_contents(volume_path):
        if entry.path is None:
            continue

        remote_path = entry.path.rstrip("/")
        relative_path = PurePosixPath(remote_path).relative_to(PurePosixPath(volume_path))
        destination = local_path.joinpath(*relative_path.parts)

        if entry.is_directory:
            download_volume_directory(
                workspace_client,
                remote_path,
                destination,
            )
        else:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            workspace_client.files.download_to(
                remote_path,
                str(destination),
            )


def load_json_from_widget(
    name: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Databricks WidgetからJSONを読み込み、dictとして返す。"""
    default_value = default if default is not None else {}

    with contextlib.suppress(Exception):
        dbutils.widgets.text(name, json.dumps(default_value))

    raw_value = dbutils.widgets.get(name)

    try:
        value = json.loads(raw_value)
        return value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Widget '{name}' must contain valid JSON.") from exc
