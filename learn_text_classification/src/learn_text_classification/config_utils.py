from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from databricks.sdk.runtime import dbutils
from loguru import logger
from omegaconf import DictConfig, OmegaConf

P = TypeVar("P")
type StepKey = str
type ProfileKey = str


@dataclass
class StepConfig:
    run: str
    params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class ResolvedStepConfig[P]:
    run: str
    params: P


@dataclass
class ExperimentConfig:
    steps: dict[StepKey, StepConfig]
    params: dict[str, Any] = field(default_factory=lambda: {})

    def get_step(self, step_name: str) -> StepConfig | None:
        return self.steps.get(step_name, None)


@dataclass
class PreprocessingConfig:
    dataset: str


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
    preprocessing: PreprocessingConfig
    experiments: dict[ProfileKey, ExperimentConfig]

    def get_experiment(self, profile_name: str) -> ExperimentConfig | None:
        return self.experiments.get(profile_name, None)


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


def resolve_step_config[P](
    config: Config,
    experiment_name: str,
    step_name: str,
    params_cls: type[P],
) -> ResolvedStepConfig[P]:
    experiment = config.get_experiment(experiment_name)
    if experiment is None:
        raise ValueError(f"Profile {experiment_name!r} not found")

    step = experiment.get_step(step_name)
    if step is None:
        raise ValueError(f"Step {step_name!r} not found in profile {experiment_name!r}")

    params: DictConfig = OmegaConf.merge(
        OmegaConf.structured(params_cls),
        step.params,
    )  # type: ignore

    resolved = OmegaConf.to_object(params)
    return ResolvedStepConfig(
        run=step.run,
        params=resolved,
    )  # type: ignore
