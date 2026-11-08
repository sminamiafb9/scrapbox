import io
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from loguru import logger
import pytest

from learn_text_classification.tracking import (
    FailureInfo,
    MLflowTracker,
    ModelArtifact,
    TrackingRecord,
    TrainerOutputs,
    tracking_utils,
)


@pytest.fixture
def loguru_capture():
    stream = io.StringIO()
    handler_id = logger.add(stream, format="{message}", level="INFO")

    yield stream

    logger.remove(handler_id)


@pytest.fixture
def mock_mlflow_client(monkeypatch):
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="experiment-123")

    monkeypatch.setattr(
        tracking_utils,
        "MlflowClient",
        MagicMock(return_value=client),
    )
    monkeypatch.setattr(
        tracking_utils,
        "check_runtime_environment",
        lambda: "databricks",
    )
    return client


@pytest.fixture
def mock_mlflow(
    mock_mlflow_client: MagicMock,
) -> MagicMock:
    """Databricks用のMLflow Client Mock。"""
    return mock_mlflow_client


@pytest.fixture
def local_tracker(monkeypatch: pytest.MonkeyPatch) -> tracking_utils.MLflowTracker:
    """Local環境のTrackerを生成する。"""
    monkeypatch.setattr(
        tracking_utils,
        "check_runtime_environment",
        lambda: "local",
    )

    return tracking_utils.MLflowTracker()


@pytest.fixture
def databricks_tracker(
    mock_mlflow: MagicMock,
) -> tracking_utils.MLflowTracker:
    """Databricks環境のTrackerを生成する。"""
    return tracking_utils.MLflowTracker(
        experiment_name="/hoge/piyo/experiment_name",
    )


def make_run(run_id: str, experiment_id: str = "experiment-123") -> SimpleNamespace:
    """MLflow RunのMockを生成する。"""
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            experiment_id=experiment_id,
        ),
    )


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------


def test_local_initialization_does_not_initialize_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking_utils,
        "check_runtime_environment",
        lambda: "local",
    )

    client_factory = MagicMock()
    monkeypatch.setattr(
        tracking_utils,
        "MlflowClient",
        client_factory,
    )

    tracker = MLflowTracker()

    assert tracker.is_mlflow_enabled is False
    client_factory.assert_not_called()


def test_databricks_initialization_propagates_experiment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking_utils,
        "check_runtime_environment",
        lambda: "databricks",
    )

    client = MagicMock()
    client.get_experiment_by_name.side_effect = RuntimeError("Experiment initialization failed")

    monkeypatch.setattr(
        tracking_utils,
        "MlflowClient",
        MagicMock(return_value=client),
    )

    with pytest.raises(RuntimeError, match="Experiment initialization failed"):
        tracking_utils.MLflowTracker(
            experiment_name="/hoge/piyo/experiment_name",
        )


# ----------------------------------------------------------------------
# Parent Run
# ----------------------------------------------------------------------


def test_start_parent_run_local(
    local_tracker: tracking_utils.MLflowTracker,
    loguru_capture: io.StringIO,
) -> None:
    run_id = local_tracker.start_parent_run(
        run_name="parent-run",
        tags={"source": "test"},
    )

    output = loguru_capture.getvalue()

    assert run_id
    assert "start_parent_run" in output
    assert "parent-run" in output


def test_start_parent_run_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.create_run.return_value = make_run("parent-123")

    run_id = databricks_tracker.start_parent_run(
        run_name="parent-run",
        tags={"source": "test"},
    )

    assert run_id == "parent-123"

    mock_mlflow.create_run.assert_called_once_with(
        experiment_id="experiment-123",
        tags={
            "source": "test",
            "mlflow.runName": "parent-run",
        },
    )


def test_start_parent_run_propagates_mlflow_error(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.create_run.side_effect = RuntimeError("Create run failed")

    with pytest.raises(RuntimeError, match="Create run failed"):
        databricks_tracker.start_parent_run()


def test_end_parent_run_local(
    local_tracker: tracking_utils.MLflowTracker,
    loguru_capture: io.StringIO,
) -> None:
    local_tracker.end_parent_run(
        run_id="parent-123",
        status="FINISHED",
    )

    output = loguru_capture.getvalue()

    assert "end_parent_run" in output
    assert "parent-123" in output


def test_end_parent_run_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    databricks_tracker.end_parent_run(
        run_id="parent-123",
        status="FINISHED",
    )

    mock_mlflow.set_terminated.assert_called_once_with(
        run_id="parent-123",
        status="FINISHED",
    )


def test_end_parent_run_with_successful_outputs(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    outputs = TrainerOutputs(
        status="SUCCEEDED",
    )

    databricks_tracker.end_parent_run(
        run_id="parent-123",
        outputs=outputs,
    )

    mock_mlflow.set_terminated.assert_called_once_with(
        run_id="parent-123",
        status="FINISHED",
    )


def test_end_parent_run_with_failed_outputs(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    outputs = TrainerOutputs(
        status="FAILED",
        failure=FailureInfo(
            code="TRAINING_ERROR",
            message="Training failed",
        ),
    )

    databricks_tracker.end_parent_run(
        run_id="parent-123",
        outputs=outputs,
    )

    mock_mlflow.set_terminated.assert_called_once_with(
        run_id="parent-123",
        status="FAILED",
    )


# ----------------------------------------------------------------------
# Child Run
# ----------------------------------------------------------------------


def test_start_child_run_local(
    local_tracker: tracking_utils.MLflowTracker,
    loguru_capture: io.StringIO,
) -> None:
    run_id = local_tracker.start_child_run(
        parent_run_id="parent-123",
        run_name="child-run",
    )

    output = loguru_capture.getvalue()

    assert run_id
    assert "start_child_run" in output
    assert "parent-123" in output


def test_start_child_run_requires_parent_run_id(
    local_tracker: tracking_utils.MLflowTracker,
) -> None:
    with pytest.raises(ValueError, match="parent_run_id"):
        local_tracker.start_child_run(parent_run_id="")


def test_start_child_run_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.get_run.return_value = make_run(
        "parent-123",
        experiment_id="experiment-parent",
    )
    mock_mlflow.create_run.return_value = make_run("child-123")

    run_id = databricks_tracker.start_child_run(
        parent_run_id="parent-123",
        run_name="child-run",
        tags={"source": "test"},
    )

    assert run_id == "child-123"

    mock_mlflow.get_run.assert_called_once_with("parent-123")

    mock_mlflow.create_run.assert_called_once_with(
        experiment_id="experiment-parent",
        tags={
            "source": "test",
            "mlflow.parentRunId": "parent-123",
            "mlflow.runName": "child-run",
        },
    )


def test_start_child_run_propagates_get_run_error(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.get_run.side_effect = RuntimeError("Parent run not found")

    with pytest.raises(RuntimeError, match="Parent run not found"):
        databricks_tracker.start_child_run(
            parent_run_id="parent-123",
        )

    mock_mlflow.create_run.assert_not_called()


def test_start_child_run_propagates_create_run_error(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.get_run.return_value = make_run("parent-123")
    mock_mlflow.create_run.side_effect = RuntimeError("Create child failed")

    with pytest.raises(RuntimeError, match="Create child failed"):
        databricks_tracker.start_child_run(
            parent_run_id="parent-123",
        )


def test_end_child_run_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    databricks_tracker.end_child_run(
        run_id="child-123",
        status="FINISHED",
    )

    mock_mlflow.set_terminated.assert_called_once_with(
        run_id="child-123",
        status="FINISHED",
    )


# ----------------------------------------------------------------------
# TrackingRecord
# ----------------------------------------------------------------------


def test_log_record_local(
    local_tracker: tracking_utils.MLflowTracker,
    loguru_capture: io.StringIO,
) -> None:
    record = TrackingRecord(
        params={"learning_rate": 0.01},
        metrics={"accuracy": 0.95},
        tags={"model": "logistic_regression"},
    )

    local_tracker.log_record("run-123", record)

    output = loguru_capture.getvalue()

    assert "log_record" in output
    assert "accuracy" in output


def test_log_record_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    record = TrackingRecord(
        params={
            "learning_rate": 0.01,
            "max_iter": 100,
            "enabled": True,
            "options": {"solver": "lbfgs"},
        },
        metrics={
            "accuracy": 0.95,
            "loss": 0.12,
        },
        tags={
            "model": "logistic_regression",
            "stage": "training",
        },
    )

    databricks_tracker.log_record(
        run_id="run-123",
        record=record,
    )

    assert mock_mlflow.log_param.call_args_list == [
        call("run-123", "learning_rate", "0.01"),
        call("run-123", "max_iter", "100"),
        call("run-123", "enabled", "true"),
        call("run-123", "options", '{"solver": "lbfgs"}'),
    ]

    assert mock_mlflow.log_metric.call_args_list == [
        call("run-123", "accuracy", 0.95),
        call("run-123", "loss", 0.12),
    ]

    assert mock_mlflow.set_tag.call_args_list == [
        call("run-123", "model", "logistic_regression"),
        call("run-123", "stage", "training"),
    ]


def test_log_record_propagates_mlflow_error(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    mock_mlflow.log_param.side_effect = RuntimeError("Log parameter failed")

    record = TrackingRecord(
        params={"learning_rate": 0.01},
    )

    with pytest.raises(RuntimeError, match="Log parameter failed"):
        databricks_tracker.log_record("run-123", record)


# ----------------------------------------------------------------------
# ModelArtifact
# ----------------------------------------------------------------------


def test_log_model_artifact_local(
    local_tracker: tracking_utils.MLflowTracker,
    loguru_capture: io.StringIO,
) -> None:
    artifact = ModelArtifact(
        uri="/Volumes/catalog/schema/models/model",
        format="spark",
        flavor="spark",
    )

    local_tracker.log_model_artifact("run-123", artifact)

    output = loguru_capture.getvalue()

    assert "log_model_artifact" in output
    assert "/Volumes/catalog/schema/models/model" in output


def test_log_model_artifact_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    artifact = ModelArtifact(
        uri="/Volumes/catalog/schema/models/model",
        format="spark",
        flavor="spark",
    )

    databricks_tracker.log_model_artifact(
        run_id="run-123",
        model_artifact=artifact,
    )

    assert mock_mlflow.set_tag.call_args_list == [
        call("run-123", "model_volume_path", "/Volumes/catalog/schema/models/model"),
        call("run-123", "model_format", "spark"),
        call("run-123", "model_flavor", "spark"),
    ]

    mock_mlflow.log_artifact.assert_not_called()


# ----------------------------------------------------------------------
# TrainerOutputs
# ----------------------------------------------------------------------


def test_log_trainer_outputs_local(
    local_tracker: tracking_utils.MLflowTracker,
    loguru_capture: io.StringIO,
) -> None:
    outputs = TrainerOutputs(
        status="SUCCEEDED",
        tracking=TrackingRecord(
            params={"max_iter": 100},
            metrics={"accuracy": 0.95},
        ),
        model=ModelArtifact(
            uri="/Volumes/catalog/schema/models/model",
            format="spark",
            flavor="spark",
        ),
    )

    local_tracker.log_trainer_outputs("run-123", outputs)

    output = loguru_capture.getvalue()

    assert "log_record" in output
    assert "log_model_artifact" in output
    assert "log_trainer_outputs" in output


def test_log_trainer_outputs_databricks(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    outputs = TrainerOutputs(
        status="SUCCEEDED",
        tracking=TrackingRecord(
            params={"max_iter": 100},
            metrics={"accuracy": 0.95},
            tags={"stage": "training"},
        ),
        model=ModelArtifact(
            uri="/Volumes/catalog/schema/models/model",
            format="spark",
            flavor="spark",
        ),
    )

    databricks_tracker.log_trainer_outputs(
        run_id="run-123",
        outputs=outputs,
    )

    mock_mlflow.log_param.assert_called_once_with(
        "run-123",
        "max_iter",
        "100",
    )

    mock_mlflow.log_metric.assert_called_once_with(
        "run-123",
        "accuracy",
        0.95,
    )

    assert mock_mlflow.set_tag.call_count == 4


def test_log_trainer_outputs_with_failure(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    outputs = TrainerOutputs(
        status="FAILED",
        failure=FailureInfo(
            code="TRAINING_ERROR",
            message="Training failed",
            details={"reason": "invalid input"},
        ),
    )

    databricks_tracker.log_trainer_outputs(
        run_id="run-123",
        outputs=outputs,
    )

    assert call("run-123", "failure_code", "TRAINING_ERROR") in (mock_mlflow.set_tag.call_args_list)

    assert call("run-123", "failure_message", "Training failed") in (mock_mlflow.set_tag.call_args_list)


# ----------------------------------------------------------------------
# Run Status
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    ["FINISHED", "FAILED", "KILLED"],
)
def test_end_child_run_accepts_valid_status(
    status: str,
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    databricks_tracker.end_child_run(
        run_id="child-123",
        status=status,
    )

    mock_mlflow.set_terminated.assert_called_once_with(
        run_id="child-123",
        status=status,
    )


def test_end_child_run_rejects_invalid_status(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="Invalid MLflow run status"):
        databricks_tracker.end_child_run(
            run_id="child-123",
            status="INVALID",
        )

    mock_mlflow.set_terminated.assert_not_called()


# ----------------------------------------------------------------------
# No fallback on MLflow errors
# ----------------------------------------------------------------------


def test_mlflow_error_does_not_fallback_to_console(
    databricks_tracker: tracking_utils.MLflowTracker,
    mock_mlflow: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console_logger = MagicMock()

    monkeypatch.setattr(
        databricks_tracker,
        "_log_local",
        console_logger,
    )

    mock_mlflow.log_metric.side_effect = RuntimeError("MLflow unavailable")

    record = TrackingRecord(
        metrics={"accuracy": 0.95},
    )

    with pytest.raises(RuntimeError, match="MLflow unavailable"):
        databricks_tracker.log_record("run-123", record)

    console_logger.assert_not_called()


def test_databricks_initialization_uses_existing_experiment(
    mock_mlflow_client,
):
    tracker = MLflowTracker(experiment_name="/Shared/my-experiment")

    mock_mlflow_client.get_experiment_by_name.assert_called_once_with("/Shared/my-experiment")
    mock_mlflow_client.create_experiment.assert_not_called()
    assert tracker._experiment_id == "experiment-123"


def test_databricks_initialization_creates_missing_experiment(
    mock_mlflow_client,
):
    mock_mlflow_client.get_experiment_by_name.return_value = None
    mock_mlflow_client.create_experiment.return_value = "experiment-456"

    tracker = MLflowTracker(experiment_name="/Shared/my-experiment")

    mock_mlflow_client.create_experiment.assert_called_once_with(name="/Shared/my-experiment")
    assert tracker._experiment_id == "experiment-456"
