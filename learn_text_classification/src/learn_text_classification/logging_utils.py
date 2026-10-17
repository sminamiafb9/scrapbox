import sys
from typing import Literal, cast, get_args

from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.sql import DataFrame

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VALID_LOG_LEVELS = get_args(LogLevel)
_current_log_level: LogLevel = "DEBUG"

LOG_LEVEL_PARAMETER_KEY = "log_level"


def setup_logging() -> None:
    """ログレベルと出力形式を設定する。"""
    global _current_log_level

    dbutils.widgets.text(LOG_LEVEL_PARAMETER_KEY, "DEBUG")
    log_level = dbutils.widgets.get(LOG_LEVEL_PARAMETER_KEY)

    if log_level in VALID_LOG_LEVELS:
        _current_log_level = cast(LogLevel, log_level)

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
    )


def show_df(
    df: DataFrame,
    n: int = 5,
    level: LogLevel = "DEBUG",
) -> None:
    """指定したログレベルが有効な場合のみ DataFrame の内容をログ出力する。"""

    levels = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    if levels[level] < levels[_current_log_level]:
        return

    rows = df.limit(n).collect()

    logger.log(
        level,
        "DataFrame sample ({} rows):\n{}",
        len(rows),
        "\n".join(str(row) for row in rows),
    )
