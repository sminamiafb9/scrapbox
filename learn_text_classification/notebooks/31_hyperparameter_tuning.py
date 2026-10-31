# Databricks notebook source
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import runpy
from typing import Any, Literal, NamedTuple

from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from loguru import logger
import optuna

from learn_text_classification import config_utils, logging_utils

# COMMAND ----------
dbutils.widgets.text("experiment_name", "baseline")
experiment_name = dbutils.widgets.get("experiment_name")

dbutils.widgets.text("step_name", "training")
step_name = dbutils.widgets.get("step_name")


# COMMAND ----------
@dataclass
class SuggestConfig:
    method: str
    params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class HyperparameterTuningConfig:
    trainer: Path
    direction: Literal["minimize", "maximize"] = "minimize"
    n_startup_trials: int = 10
    n_trials: int = 50
    study_params: dict[str, Any] = field(default_factory=lambda: {})
    optimize_params: dict[str, Any] = field(default_factory=lambda: {})
    fixed_params: dict[str, Any] = field(default_factory=lambda: {})
    suggest_params: dict[str, SuggestConfig] = field(default_factory=lambda: {})


# COMMAND ----------
logging_utils.setup_logging()
config = config_utils.load_config("./config")
hp_tuning_config = config_utils.resolve_step_config(
    config, experiment_name, step_name, HyperparameterTuningConfig
)

spark = DatabricksSession.builder.getOrCreate()


# COMMAND ----------
def run_script(script: str | Path, **kwargs: Any) -> dict[str, Any]:
    result = runpy.run_path(str(script), init_globals=kwargs)
    return result


def create_objective_fn(
    script_path: Path,
    fixed_params: dict[str, Any] | None = None,
    suggest_params: dict[str, SuggestConfig] | None = None,
    output_key: str = "outputs",
    score_key: str = "score",
) -> Callable[[optuna.Trial], float]:
    fixed_params = fixed_params or {}
    suggest_params = suggest_params or {}

    def objective(
        trial: optuna.Trial,
    ) -> float:
        params = {}
        for name, sp in suggest_params.items():
            suggest_method = getattr(trial, sp.method, None)
            if suggest_method is None or not callable(suggest_method):
                raise ValueError(f"Invalid suggest method: {sp.method}")

            params[name] = suggest_method(
                name,
                **sp.params,
            )
        outputs = run_script(script_path, parameters=params | fixed_params)
        return outputs[output_key][score_key]

    return objective


def optimize(
    direction: Literal["minimize", "maximize"],
    objective: Callable[[optuna.Trial], float],
    n_startup_trials: int = 10,
    n_trials: int = 50,
    study_params: dict[str, Any] | None = None,
    optimize_params: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(
            seed=123,
            n_startup_trials=n_startup_trials,
        ),
        **(study_params or {}),
    )

    study.optimize(objective, n_trials=n_trials, **(optimize_params or {}))

    class OptimizeResult(NamedTuple):
        best_value: Any
        best_params: Any

    return OptimizeResult(study.best_value, study.best_params)


# COMMAND ----------
label_cols = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]
for emotion in label_cols:
    obj_fn = create_objective_fn(
        hp_tuning_config.params.trainer,
        hp_tuning_config.params.fixed_params | {"label_col": emotion},
        hp_tuning_config.params.suggest_params,
    )

    best_value, best_params = optimize(
        hp_tuning_config.params.direction,
        obj_fn,
        hp_tuning_config.params.n_startup_trials,
        hp_tuning_config.params.n_trials,
        hp_tuning_config.params.study_params,
        hp_tuning_config.params.optimize_params,
    )

    logger.info(f"{emotion= }, {best_value = }, {best_params = }")
