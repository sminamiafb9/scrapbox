from pathlib import Path

from databricks.sdk.runtime import dbutils
from loguru import logger
from omegaconf import DictConfig, OmegaConf

PROFILE_ORDER = ("local", "sandbox")
PROFILE_PARAMETER_KEY = "profile"
CONFIG_PARAMETER_KEY = "config"


def load_config(config_dir: str | Path) -> DictConfig:
    config_dir = Path(config_dir)

    dbutils.widgets.text(PROFILE_PARAMETER_KEY, PROFILE_ORDER[0])
    profile = dbutils.widgets.get(PROFILE_PARAMETER_KEY)

    dbutils.widgets.text(CONFIG_PARAMETER_KEY, "")
    config_name = dbutils.widgets.get(CONFIG_PARAMETER_KEY).strip()

    if profile not in PROFILE_ORDER:
        raise ValueError(f"Invalid profile: {profile!r}.\nAvailable profiles: {PROFILE_ORDER}")

    logger.info("Using profile: {}", profile)
    logger.info("Using config: {}", config_name or "(none)")

    config: DictConfig = OmegaConf.create()

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

    logger.info("\n{}\n", OmegaConf.to_yaml(config))

    return config


def validate[T](config: DictConfig, cls: type[T]) -> T:
    validated = OmegaConf.merge(OmegaConf.structured(cls), config)
    return OmegaConf.to_object(validated)


def load_config_by_key[T](config_dir: str | Path, cls: type[T], key: str) -> T:
    config = load_config(config_dir)
    obj = validate(config[key], cls)
    logger.info("loaded config class:\n{}\n", OmegaConf.to_yaml(obj))
    return obj
