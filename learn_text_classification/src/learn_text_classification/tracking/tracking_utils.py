import json
import math
from typing import Any, Literal, cast
from uuid import uuid4

from loguru import logger
from mlflow.tracking import MlflowClient

from ..job_utils import check_runtime_environment
from .models import FailureInfo, ModelArtifact, TrackingRecord, TrainerOutputs

RunStatus = Literal["FINISHED", "FAILED", "KILLED"]

_VALID_RUN_STATUSES = {"FINISHED", "FAILED", "KILLED"}


class MLflowTracker:
    def __init__(
        self,
        experiment_name: str | None = None,
    ) -> None:
        self._runtime = check_runtime_environment()
        self._is_local = self._runtime == "local"
        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None

        if not self._is_local:
            self._client = MlflowClient()
            self._initialize_experiment(experiment_name)

    @property
    def is_mlflow_enabled(self) -> bool:
        """MLflow Trackingが選択されているかを返す。"""
        return not self._is_local

    def _initialize_experiment(self, experiment_name: str | None) -> None:
        """Databricks Workspace上のExperimentを取得または作成する。"""
        if not experiment_name or not experiment_name.startswith("/"):
            raise ValueError("experiment_name must be an absolute Workspace path, for example '/Shared/my-experiment'.")

        client = self._get_client()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                name=experiment_name,
            )
        else:
            experiment_id = experiment.experiment_id

        self._experiment_id = experiment_id

    @staticmethod
    def _validate_run_status(status: str) -> RunStatus:
        if status not in _VALID_RUN_STATUSES:
            raise ValueError(f"Invalid MLflow run status: {status!r}. Expected one of {_VALID_RUN_STATUSES}.")

        return status  # type: ignore[return-value]

    @staticmethod
    def _resolve_trainer_status(status: str) -> RunStatus:
        """TrainerOutputsのstatusをMLflow Run Statusへ変換する。"""
        if status == "SUCCEEDED":
            return "FINISHED"

        if status == "FAILED":
            return "FAILED"

        raise ValueError(f"Invalid trainer status: {status!r}")

    @staticmethod
    def _serialize_param(value: Any) -> str:
        """MLflow Parameterに登録できる文字列へ変換する。"""
        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )

        if value is None:
            return "null"

        if isinstance(value, bool):
            return str(value).lower()

        return str(value)

    def _log_local(self, event: str, **data: Any) -> None:
        """Local Console Trackingを実行する。"""
        logger.info(
            "[Tracking] {}: {}",
            event,
            json.dumps(
                data,
                ensure_ascii=False,
                default=str,
                allow_nan=False,
            ),
        )

    def _get_client(self) -> MlflowClient:
        if self._client is None:
            raise RuntimeError("MLflow client is not initialized.")

        return self._client

    def _create_local_run(self, run_name: str | None) -> str:
        run_id = str(uuid4())

        self._log_local(
            "start_run",
            run_id=run_id,
            run_name=run_name,
        )

        return run_id

    # ------------------------------------------------------------------
    # Parent Run
    # ------------------------------------------------------------------

    def start_parent_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Parent Runを開始し、Run IDを返す。"""
        if self._is_local:
            run_id = self._create_local_run(run_name)

            self._log_local(
                "start_parent_run",
                run_id=run_id,
                tags=tags or {},
            )

            return run_id

        client = self._get_client()

        if self._experiment_id is None:
            raise RuntimeError("MLflow experiment is not initialized.")

        run_tags = dict(tags or {})

        if run_name:
            run_tags["mlflow.runName"] = run_name

        run = client.create_run(
            experiment_id=self._experiment_id,
            tags=run_tags,
        )

        run_id = run.info.run_id

        logger.info("Started Parent Run: {}", run_id)

        return run_id

    def end_parent_run(
        self,
        run_id: str,
        status: str = "FINISHED",
        outputs: TrainerOutputs | None = None,
    ) -> None:
        """Parent Runを終了する。"""
        if outputs is not None:
            self.log_trainer_outputs(run_id, outputs)
            status = self._resolve_trainer_status(outputs.status)

        resolved_status = self._validate_run_status(status)

        if self._is_local:
            self._log_local(
                "end_parent_run",
                run_id=run_id,
                status=resolved_status,
            )
            return

        self._get_client().set_terminated(
            run_id=run_id,
            status=resolved_status,
        )

        logger.info(
            "Ended Parent Run: {} (status={})",
            run_id,
            resolved_status,
        )

    # ------------------------------------------------------------------
    # Child Run
    # ------------------------------------------------------------------

    def start_child_run(
        self,
        parent_run_id: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """指定されたParent Runに紐づくChild Runを開始する。"""
        if not parent_run_id:
            raise ValueError("parent_run_id must not be empty.")

        if self._is_local:
            run_id = self._create_local_run(run_name)

            self._log_local(
                "start_child_run",
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags or {},
            )

            return run_id

        client = self._get_client()

        # Parent RunからExperiment IDを取得する。
        parent_run = client.get_run(parent_run_id)
        experiment_id = cast(str, parent_run.info.experiment_id)

        run_tags = dict(tags or {})
        run_tags["mlflow.parentRunId"] = parent_run_id

        if run_name:
            run_tags["mlflow.runName"] = run_name

        run = client.create_run(
            experiment_id=experiment_id,
            tags=run_tags,
        )

        run_id = run.info.run_id

        logger.info(
            "Started Child Run: {} (parent={})",
            run_id,
            parent_run_id,
        )

        return run_id

    def end_child_run(
        self,
        run_id: str,
        status: str = "FINISHED",
        record: TrackingRecord | None = None,
    ) -> None:
        """TrackingRecordを登録してChild Runを終了する。"""
        if record is not None:
            self.log_record(run_id, record)

        resolved_status = self._validate_run_status(status)

        if self._is_local:
            self._log_local(
                "end_child_run",
                run_id=run_id,
                status=resolved_status,
            )
            return

        self._get_client().set_terminated(
            run_id=run_id,
            status=resolved_status,
        )

        logger.info(
            "Ended Child Run: {} (status={})",
            run_id,
            resolved_status,
        )

    # ------------------------------------------------------------------
    # TrackingRecord
    # ------------------------------------------------------------------

    def _set_tags(
        self,
        run_id: str,
        tags: dict[str, str],
    ) -> None:
        """Runに複数のTagを登録する。"""
        if self._is_local:
            self._log_local(
                "set_tags",
                run_id=run_id,
                tags=tags,
            )
            return

        client = self._get_client()

        for key, value in tags.items():
            client.set_tag(
                run_id,
                key,
                value,
            )

    def log_record(
        self,
        run_id: str,
        record: TrackingRecord,
    ) -> None:
        """TrackingRecordのparams、metrics、tagsを登録する。"""
        if self._is_local:
            self._log_local(
                "log_record",
                run_id=run_id,
                record=record.to_dict(),
            )
            return

        client = self._get_client()

        for key, value in record.params.items():
            client.log_param(
                run_id,
                key,
                self._serialize_param(value),
            )

        for key, value in record.metrics.items():
            metric_value = float(value)

            if not math.isfinite(metric_value):
                raise ValueError(f"metrics[{key!r}] must be finite.")

            client.log_metric(
                run_id,
                key,
                metric_value,
            )

        if record.tags:
            self._set_tags(
                run_id,
                record.tags,
            )

    # ------------------------------------------------------------------
    # ModelArtifact
    # ------------------------------------------------------------------

    def log_model_artifact(
        self,
        run_id: str,
        model_artifact: ModelArtifact,
    ) -> None:
        """ModelArtifactの参照情報をRun Tagとして登録する。

        モデル本体のアップロードは行わない。
        """
        artifact_tags = {
            "model_volume_path": model_artifact.uri,
            "model_format": model_artifact.format,
            "model_flavor": model_artifact.flavor,
        }

        if self._is_local:
            self._log_local(
                "log_model_artifact",
                run_id=run_id,
                artifact=model_artifact.to_dict(),
            )
            return

        self._set_tags(
            run_id,
            artifact_tags,
        )

    # ------------------------------------------------------------------
    # TrainerOutputs
    # ------------------------------------------------------------------

    def log_trainer_outputs(
        self,
        run_id: str,
        outputs: TrainerOutputs,
    ) -> None:
        """TrainerOutputsに含まれるTracking情報を登録する。"""
        self.log_record(
            run_id,
            outputs.tracking,
        )

        if outputs.model is not None:
            self.log_model_artifact(
                run_id,
                outputs.model,
            )

        if outputs.failure is not None:
            self._log_failure(
                run_id,
                outputs.failure,
            )

        if self._is_local:
            self._log_local(
                "log_trainer_outputs",
                run_id=run_id,
                outputs=outputs.to_dict(),
            )

    def _log_failure(
        self,
        run_id: str,
        failure: FailureInfo,
    ) -> None:
        """FailureInfoをRun Tagとして登録する。"""
        failure_tags = {
            "failure_code": failure.code,
            "failure_message": failure.message,
        }

        if self._is_local:
            self._log_local(
                "log_failure",
                run_id=run_id,
                failure=failure.to_dict(),
            )
            return

        self._set_tags(
            run_id,
            failure_tags,
        )
