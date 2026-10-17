from dataclasses import dataclass
import os
from pathlib import Path
from typing import cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf


@dataclass
class IngestConfig:
    url: str
    volume: Path


@dataclass
class Config:
    ingest: IngestConfig


PROFILE_ORDER = ("local", "sandbox")
APP_PROFILE_ENV_KEY = "APP_PROFILE"


def load_config(config_dir: str | Path) -> Config:
    config_dir = Path(config_dir)

    config: DictConfig = OmegaConf.structured(Config)

    profile = os.getenv(APP_PROFILE_ENV_KEY, PROFILE_ORDER[0])
    if profile not in PROFILE_ORDER:
        raise ValueError(
            f"Invalid profile: {profile!r}.\nAvailable profiles: {PROFILE_ORDER}"
        )
    logger.info("Using profile: {}", profile)

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
