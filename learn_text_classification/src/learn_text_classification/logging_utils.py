import sys
from typing import Literal

from loguru import logger

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(log_level: LogLevel = "INFO") -> None:
    """ログレベルと出力形式を設定する。"""

    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
    )
