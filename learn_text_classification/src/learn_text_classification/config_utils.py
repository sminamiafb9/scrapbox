from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from databricks.sdk.runtime import dbutils
from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf

from .tracking import TrackingConfig

type ProfileKey = str

PROFILE_ORDER = ("local", "sandbox")
PROFILE_PARAMETER_KEY = "profile"
CONFIG_PARAMETER_KEY = "config"


@dataclass
class ExperimentConfig:
    run: str = MISSING
    params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class EmbeddingConfig:
    cache_folder: Path = MISSING
    batch_size: int = MISSING
    transfer_batch_size: int = MISSING
    embedding_model_name: str = MISSING
    dataset: str = MISSING


@dataclass
class BinaryRelevanceAssemblingConfig:
    experiment_name: str = "/Users/shotaro.minami8e40a5@gmail.com/.mlflow/learn_text_classification"
    model_id: str = "logistic_regression"
    save_path: str = "/Volumes/sandbox/shotaro_minami8e40a5/learn_text_classification/models/test"


@dataclass
class PreprocessingConfig:
    dataset: str = MISSING


@dataclass
class IngestConfig:
    url: str = MISSING
    volume: Path = MISSING


@dataclass
class PrepareConfig:
    raw_data: Path = MISSING
    dataset: str = MISSING


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    experiments: dict[ProfileKey, ExperimentConfig] = field(default_factory=lambda: {})
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    assembling: BinaryRelevanceAssemblingConfig = field(default_factory=BinaryRelevanceAssemblingConfig)

    def get_experiment(self, experiment_key: str) -> ExperimentConfig | None:
        return self.experiments.get(experiment_key)


def load_config(config_dir: str | Path) -> Config:
    config_dir = Path(config_dir)

    dbutils.widgets.text(PROFILE_PARAMETER_KEY, PROFILE_ORDER[0])
    profile = dbutils.widgets.get(PROFILE_PARAMETER_KEY)

    dbutils.widgets.text(CONFIG_PARAMETER_KEY, "")
    config_name = dbutils.widgets.get(CONFIG_PARAMETER_KEY).strip()

    if profile not in PROFILE_ORDER:
        raise ValueError(f"Invalid profile: {profile!r}.\nAvailable profiles: {PROFILE_ORDER}")

    logger.info("Using profile: {}", profile)
    logger.info("Using config: {}", config_name or "(none)")

    config: DictConfig = OmegaConf.structured(Config)

    profile_names = PROFILE_ORDER[: PROFILE_ORDER.index(profile) + 1]

    for profile_name in profile_names:
        config_path = config_dir / f"{profile_name}.yml"

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        override = OmegaConf.load(config_path)

        if "experiments" in override:
            logger.warning(
                "'experiments' in {} is deprecated. Move experiments to a separate config file.",
                config_path,
            )

        merged = OmegaConf.merge(config, override)

        if not isinstance(merged, DictConfig):
            raise TypeError("The top-level of the configuration must be a DictConfig.")

        config = merged

    if config_name:
        config_path = config_dir / "experiments" / config_name

        if config_path.suffix not in (".yml", ".yaml"):
            config_path = config_path.with_suffix(".yml")

        if not config_path.is_file():
            raise FileNotFoundError(f"Experiment config file not found: {config_path}")

        override = OmegaConf.load(config_path)
        merged = OmegaConf.merge(config, override)

        if not isinstance(merged, DictConfig):
            raise TypeError("The top-level of the configuration must be a DictConfig.")

        config = merged

    missing = OmegaConf.missing_keys(config)
    if missing:
        raise ValueError(f"Required config values are missing: {sorted(missing)}")

    result = OmegaConf.to_object(config)
    logger.info("\n{}\n", OmegaConf.to_yaml(config))

    return cast(Config, result)
