from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal

TrainerStatus = Literal["SUCCEEDED", "FAILED"]

_VALID_STATUSES = {"SUCCEEDED", "FAILED"}


def _validate_json_value(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number.")
        return

    if isinstance(value, list):
        for index, item in enumerate(value):  # type: ignore
            _validate_json_value(item, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():  # type: ignore
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            _validate_json_value(item, f"{path}.{key}")
        return

    raise TypeError(f"{path} contains a non-JSON-serializable value: {type(value).__name__}")


def _validate_dict(value: dict[str, Any], name: str) -> None:
    _validate_json_value(value, name)


@dataclass
class TrackingRecord:
    params: dict[str, Any] = field(default_factory=lambda: {})
    metrics: dict[str, float] = field(default_factory=lambda: {})
    tags: dict[str, str] = field(default_factory=lambda: {})
    artifacts: list[str] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        _validate_dict(self.params, "params")
        _validate_dict(self.metrics, "metrics")
        _validate_dict(self.tags, "tags")

        for key, value in self.metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"metrics[{key!r}] must be finite.")

        if not all(isinstance(value, str) for value in self.tags.values()):  # type: ignore
            raise TypeError("tags values must be strings.")

        for value in self.tags.values():
            _validate_json_value(value, "tags")

        if not all(isinstance(value, str) for value in self.artifacts):  # type: ignore
            raise TypeError("artifacts values must be strings.")

        _validate_json_value(self.artifacts, "artifacts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackingRecord":
        return cls(**data)


@dataclass
class ModelArtifact:
    uri: str
    format: str
    flavor: str
    metadata: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        for name, value in (
            ("uri", self.uri),
            ("format", self.format),
            ("flavor", self.flavor),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty.")

        _validate_dict(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "format": self.format,
            "flavor": self.flavor,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifact":
        return cls(**data)


@dataclass
class FailureInfo:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must not be empty.")

        if not self.message:
            raise ValueError("message must not be empty.")

        _validate_dict(self.details, "details")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureInfo":
        return cls(**data)


@dataclass
class TrainerOutputs:
    """Trainer / Tuner共通の実行結果。"""

    status: TrainerStatus
    tracking: TrackingRecord = field(default_factory=TrackingRecord)
    model: ModelArtifact | None = None
    failure: FailureInfo | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}. Got: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tracking": self.tracking.to_dict(),
            "model": (self.model.to_dict() if self.model is not None else None),
            "failure": (self.failure.to_dict() if self.failure is not None else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainerOutputs":
        return cls(
            status=data["status"],
            tracking=TrackingRecord.from_dict(data.get("tracking", {})),
            model=(ModelArtifact.from_dict(data["model"]) if data.get("model") is not None else None),
            failure=(FailureInfo.from_dict(data["failure"]) if data.get("failure") is not None else None),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> "TrainerOutputs":
        data: dict[str, Any] = json.loads(value)
        return cls.from_dict(data)


@dataclass
class TrackingConfig:
    save_model: bool = False
    parent_run_id: str | None = None
    workspace_base: Path | str | None = None
    project_name: Path | str | None = None
    volume_base: Path | str | None = None

    def __post_init__(self) -> None:
        if self.workspace_base is not None:
            workspace_base = Path(self.workspace_base)
            if not workspace_base.is_absolute():
                raise ValueError("workspace_base must be an absolute Workspace path.")

        if self.project_name is not None:
            experiment_name = Path(self.project_name)
            if experiment_name.is_absolute() or ".." in experiment_name.parts:
                raise ValueError("experiment_name must be a relative path.")

        if self.volume_base is not None:
            volume_base = Path(self.volume_base)
            if not volume_base.is_absolute() or not volume_base.as_posix().startswith("/Volumes/"):
                raise ValueError("volume_base must follow '/Volumes/<catalog>/<schema>/<volume>'.")

    @property
    def experiment_path(self) -> str | None:
        if self.workspace_base is None or self.project_name is None:
            return None

        return (Path(self.workspace_base) / Path(self.project_name)).as_posix()
