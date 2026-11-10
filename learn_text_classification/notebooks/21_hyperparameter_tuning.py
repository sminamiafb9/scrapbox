# Databricks notebook source
from collections.abc import Callable
from dataclasses import dataclass, field
import math
from pathlib import Path
import runpy
from typing import Any, Literal, cast

from loguru import logger
from omegaconf import OmegaConf
import optuna

from learn_text_classification import job_utils, logging_utils
from learn_text_classification.tracking import (
    MLflowTracker,
    TrackingConfig,
    TrackingRecord,
    TrainerOutputs,
)

# COMMAND ----------
DISABLED_PARENT_RUN = "DUMMY-PARENT-RUN"


# COMMAND ----------
@dataclass
class SuggestConfig:
    method: str
    params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class HyperparameterTuningConfig:
    trainer: Path = Path("./31_logistic_regression.py")
    direction: Literal["minimize", "maximize"] = "minimize"
    n_startup_trials: int = 10
    n_trials: int = 50
    study_params: dict[str, Any] = field(default_factory=lambda: {})
    optimize_params: dict[str, Any] = field(default_factory=lambda: {})
    fixed_params: dict[str, Any] = field(default_factory=lambda: {})
    suggest_params: dict[str, SuggestConfig] = field(default_factory=lambda: {})


# COMMAND ----------
# Runnerからの入力変数
# runpy.run_path() の init_globals から受け取る。
raw_parameters: dict[str, Any] = globals().get("parameters", {})
raw_tracking: dict[str, Any] = globals().get("tracking", {})

# COMMAND ----------
# Hyperparameter Tuning設定
logging_utils.setup_logging()

hp_tuning_config = OmegaConf.to_object(
    OmegaConf.merge(
        OmegaConf.structured(HyperparameterTuningConfig),
        raw_parameters,
    )
)
hp_tuning_config = cast(HyperparameterTuningConfig, hp_tuning_config)

# Tracking設定
tracking_config = OmegaConf.to_object(
    OmegaConf.merge(
        OmegaConf.structured(TrackingConfig()),
        raw_tracking,
    )
)
tracking_config = cast(TrackingConfig, tracking_config)
parent_run_id = tracking_config.parent_run_id or DISABLED_PARENT_RUN
volume_base = tracking_config.volume_base


# COMMAND ----------
# Tracking初期化
tracker = MLflowTracker()
trainer_path = Path(hp_tuning_config.trainer)
if not trainer_path.is_file():
    raise FileNotFoundError(f"Trainer notebook not found: {trainer_path}")


# COMMAND ----------
def run_trainer(
    parameters: dict[str, Any],
    tracking: TrackingConfig,
) -> TrainerOutputs:
    """Trainer Notebookを実行し、outputsを取得する。"""
    namespace = runpy.run_path(
        str(trainer_path),
        init_globals={
            "parameters": parameters,
            "tracking": tracking,
        },
    )

    outputs = namespace.get("outputs")

    if not isinstance(outputs, TrainerOutputs):
        raise TypeError("Trainer must return TrainerOutputs through the top-level 'outputs' variable.")

    return outputs


# COMMAND ----------
def create_objective_fn(
    fixed_params: dict[str, Any],
    suggest_params: dict[str, SuggestConfig],
) -> Callable[[optuna.Trial], float]:
    """Optuna TrialのObjective関数を生成する。"""

    def objective(trial: optuna.Trial) -> float:
        trial_params: dict[str, Any] = {}

        # Optunaによるパラメータ生成
        for name, suggest_config in suggest_params.items():
            suggest_method = getattr(
                trial,
                suggest_config.method,
                None,
            )

            if suggest_method is None or not callable(suggest_method):
                raise ValueError(f"Invalid suggest method: {suggest_config.method}")

            trial_params[name] = suggest_method(
                name,
                **suggest_config.params,
            )

        parameters = fixed_params | trial_params

        # Child Run開始
        child_run_id = tracker.start_child_run(
            parent_run_id=parent_run_id,
            run_name=f"trial-{trial.number}",
            tags={
                "optuna.trial_number": str(trial.number),
            },
        )

        trial_tracking = TrackingConfig(
            parent_run_id=parent_run_id,
            save_model=False,
            volume_base=volume_base,
        )

        # Trainer実行
        try:
            trainer_outputs = run_trainer(
                parameters=parameters,
                tracking=trial_tracking,
            )

            if trainer_outputs.status != "SUCCEEDED":
                raise RuntimeError(f"Trainer failed in trial {trial.number}: {trainer_outputs.failure}")

            score = trainer_outputs.tracking.metrics.get("score")

            if score is None:
                raise ValueError("TrainerOutputs.tracking.metrics['score'] is required.")

            score = float(score)

            if not math.isfinite(score):
                raise ValueError(f"Trial score must be finite. Got: {score}")

        except Exception:
            logger.exception(f"Trial {trial.number} failed.")

            tracker.end_child_run(
                run_id=child_run_id,
                status="FAILED",
            )
            raise

        # Trial結果をChild Runに登録して終了
        tracker.end_child_run(
            run_id=child_run_id,
            status="FINISHED",
            record=TrackingRecord(
                params=trainer_outputs.tracking.params,
                metrics=trainer_outputs.tracking.metrics,
            ),
        )

        return score

    return objective


# COMMAND ----------
def optimize(
    objective: Callable[[optuna.Trial], float],
    config: HyperparameterTuningConfig,
) -> tuple[float, dict[str, Any]]:
    """Optunaによるハイパーパラメータ最適化を実行する。"""
    study = optuna.create_study(
        direction=config.direction,
        sampler=optuna.samplers.TPESampler(
            seed=123,
            n_startup_trials=config.n_startup_trials,
        ),
        **config.study_params,
    )

    study.optimize(
        objective,
        n_trials=config.n_trials,
        catch=(Exception,),
        **config.optimize_params,
    )

    # 完了したTrialが存在することを確認
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        raise RuntimeError("Hyperparameter tuning failed: no trial completed successfully.")

    return float(study.best_value), dict(study.best_params)


# COMMAND ----------
# Hyperparameter Tuning実行
objective = create_objective_fn(
    fixed_params=hp_tuning_config.fixed_params,
    suggest_params=hp_tuning_config.suggest_params,
)

best_value, best_params = optimize(
    objective=objective,
    config=hp_tuning_config,
)

logger.info(f"Hyperparameter tuning completed: best_value={best_value}, best_params={best_params}")


# COMMAND ----------
# Best Parametersによる最終学習
final_parameters = hp_tuning_config.fixed_params | best_params

save_model = job_utils.check_runtime_environment() == "databricks"
final_tracking = TrackingConfig(
    parent_run_id=parent_run_id,
    save_model=save_model,
    volume_base=volume_base,
)

outputs = run_trainer(
    parameters=final_parameters,
    tracking=final_tracking,
)

logger.info(f"Final TrainerOutputs: {outputs.to_dict()}")
