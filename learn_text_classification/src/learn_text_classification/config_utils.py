from dataclasses import dataclass
from pathlib import Path
from typing import cast

from databricks.sdk.runtime import dbutils
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@dataclass
class IngestConfig:
    url: str
    volume: Path


@dataclass
class PrepareConfig:
    raw_data: Path
    dataset: str


@dataclass
class Config:
    ingest: IngestConfig
    prepare: PrepareConfig


PROFILE_ORDER = ("local", "sandbox")
PROFILE_PARAMETER_KEY = "profile"


def load_config(config_dir: str | Path) -> Config:
    config_dir = Path(config_dir)

    dbutils.widgets.text(PROFILE_PARAMETER_KEY, PROFILE_ORDER[0])
    profile = dbutils.widgets.get(PROFILE_PARAMETER_KEY)
    if profile not in PROFILE_ORDER:
        raise ValueError(
            f"Invalid profile: {profile!r}.\nAvailable profiles: {PROFILE_ORDER}"
        )
    logger.info("Using profile: {}", profile)

    config: DictConfig = OmegaConf.structured(Config)

    profile_index = PROFILE_ORDER.index(profile)

    for profile_name in PROFILE_ORDER[: profile_index + 1]:
        config_path = config_dir / f"{profile_name}.yml"

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        override = OmegaConf.load(config_path)
        merged = OmegaConf.merge(config, override)

        if not isinstance(merged, DictConfig):
            raise TypeError("The top-level of the configuration must be a DictConfig.")

        config = merged

    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Required config values are missing: {sorted(missing)}")

    result = OmegaConf.to_object(config)
    logger.info("\n{}\n", OmegaConf.to_yaml(result))
    return cast(Config, result)
