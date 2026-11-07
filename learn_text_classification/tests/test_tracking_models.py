import json

import pytest

from learn_text_classification.tracking.models import (
    FailureInfo,
    ModelArtifact,
    TrackingRecord,
    TrainerOutputs,
)


class TestTrackingRecord:
    def test_default_values_are_independent(self) -> None:
        first = TrackingRecord()
        second = TrackingRecord()

        first.params["key"] = "value"
        first.artifacts.append("artifact")

        assert second.params == {}
        assert second.artifacts == []

    def test_to_dict(self) -> None:
        record = TrackingRecord(
            params={"learning_rate": 0.01},
            metrics={"accuracy": 0.95},
            tags={"stage": "training"},
            artifacts=["model.pkl"],
        )

        assert record.to_dict() == {
            "params": {"learning_rate": 0.01},
            "metrics": {"accuracy": 0.95},
            "tags": {"stage": "training"},
            "artifacts": ["model.pkl"],
        }

    def test_from_dict(self) -> None:
        data = {
            "params": {"learning_rate": 0.01},
            "metrics": {"accuracy": 0.95},
            "tags": {"stage": "training"},
            "artifacts": ["model.pkl"],
        }

        record = TrackingRecord.from_dict(data)

        assert record.params == data["params"]
        assert record.metrics == data["metrics"]
        assert record.tags == data["tags"]
        assert record.artifacts == data["artifacts"]

    def test_rejects_non_finite_metrics(self) -> None:
        with pytest.raises(ValueError):
            TrackingRecord(metrics={"accuracy": float("nan")})

        with pytest.raises(ValueError):
            TrackingRecord(metrics={"accuracy": float("inf")})

    def test_rejects_non_json_serializable_params(self) -> None:
        with pytest.raises(TypeError):
            TrackingRecord(params={"invalid": object()})

    def test_rejects_non_string_tags(self) -> None:
        with pytest.raises(TypeError):
            TrackingRecord(tags={"invalid": 123})  # type: ignore

    def test_rejects_non_string_artifacts(self) -> None:
        with pytest.raises(TypeError):
            TrackingRecord(artifacts=[123])  # type: ignore

    def test_rejects_non_finite_params(self) -> None:
        with pytest.raises(ValueError):
            TrackingRecord(params={"invalid": float("nan")})


class TestModelArtifact:
    def test_to_dict_and_from_dict(self) -> None:
        artifact = ModelArtifact(
            uri="/Volumes/catalog/schema/models/model",
            format="mlflow",
            flavor="sklearn",
            metadata={"version": 1},
        )

        restored = ModelArtifact.from_dict(artifact.to_dict())

        assert restored == artifact

    @pytest.mark.parametrize(
        "field",
        ["uri", "format", "flavor"],
    )
    def test_rejects_empty_required_fields(self, field: str) -> None:
        values = {
            "uri": "model/path",
            "format": "mlflow",
            "flavor": "sklearn",
        }
        values[field] = ""

        with pytest.raises(ValueError):
            ModelArtifact(**values)  # type: ignore


class TestFailureInfo:
    def test_to_dict_and_from_dict(self) -> None:
        failure = FailureInfo(
            code="TRAINING_ERROR",
            message="Training failed",
            details={"reason": "invalid input"},
        )

        restored = FailureInfo.from_dict(failure.to_dict())

        assert restored == failure

    def test_rejects_empty_required_fields(self) -> None:
        with pytest.raises(ValueError):
            FailureInfo(code="", message="error")

        with pytest.raises(ValueError):
            FailureInfo(code="ERROR", message="")


class TestTrainerOutputs:
    def test_to_dict_and_from_dict(self) -> None:
        outputs = TrainerOutputs(
            status="SUCCEEDED",
            tracking=TrackingRecord(
                metrics={"accuracy": 0.95},
            ),
            model=ModelArtifact(
                uri="/Volumes/catalog/schema/models/model",
                format="mlflow",
                flavor="sklearn",
            ),
        )

        restored = TrainerOutputs.from_dict(outputs.to_dict())

        assert restored == outputs

    def test_json_round_trip(self) -> None:
        outputs = TrainerOutputs(
            status="FAILED",
            tracking=TrackingRecord(
                params={"epochs": 10},
                metrics={"loss": 0.25},
            ),
            failure=FailureInfo(
                code="TRAINING_ERROR",
                message="Training failed",
                details={"reason": "timeout"},
            ),
        )

        json_value = outputs.to_json()
        restored = TrainerOutputs.from_json(json_value)

        assert restored == outputs

    def test_json_is_valid(self) -> None:
        outputs = TrainerOutputs(status="SUCCEEDED")

        data = json.loads(outputs.to_json())

        assert data["status"] == "SUCCEEDED"
        assert data["tracking"] == {
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }
        assert data["model"] is None
        assert data["failure"] is None

    def test_from_json_with_missing_tracking(self) -> None:
        outputs = TrainerOutputs.from_json('{"status": "SUCCEEDED"}')

        assert outputs.status == "SUCCEEDED"
        assert outputs.tracking == TrackingRecord()

    def test_rejects_invalid_status(self) -> None:
        with pytest.raises(ValueError):
            TrainerOutputs(status="UNKNOWN")  # type: ignore[arg-type]

    def test_rejects_non_json_serializable_values(self) -> None:
        outputs = TrainerOutputs(status="SUCCEEDED")

        # 生成後に不正な値を設定
        outputs.tracking.params["invalid"] = float("nan")

        with pytest.raises(ValueError):
            outputs.to_json()
