from pathlib import Path
from unittest.mock import MagicMock

import pytest

from learn_text_classification.config_utils import load_config


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    experiments_dir = config_dir / "experiments"
    experiments_dir.mkdir(parents=True)

    (config_dir / "local.yml").write_text(
        """
ingest:
  url: local-url
  volume: /tmp/raw
prepare:
  raw_data: /tmp/raw
  dataset: local-dataset
preprocessing:
  dataset: local-dataset
embedding:
  cache_folder: /tmp/cache
  batch_size: 16
  transfer_batch_size: 8
  embedding_model_name: local-model
  dataset: local-dataset
tracking:
  save_model: false
""",
        encoding="utf-8",
    )

    (config_dir / "sandbox.yml").write_text(
        """
ingest:
  url: sandbox-url
  volume: /Volumes/catalog/schema/volume
embedding:
  cache_folder: /Volumes/catalog/schema/volume/cache
tracking:
  save_model: true
""",
        encoding="utf-8",
    )

    (experiments_dir / "baseline.yml").write_text(
        """
experiments:
  baseline:
    run: baseline-run
    params:
      C: 1.0
""",
        encoding="utf-8",
    )

    (experiments_dir / "override.yml").write_text(
        """
embedding:
  batch_size: 32
experiments:
  baseline:
    run: baseline-run
    params:
      C: 2.0
      max_iter: 100
""",
        encoding="utf-8",
    )

    return config_dir


@pytest.fixture
def mock_widgets(monkeypatch):
    mock_dbutils = MagicMock()
    values = {
        "profile": "local",
        "config": "",
    }

    mock_dbutils.widgets.get.side_effect = lambda key: values[key]

    monkeypatch.setattr(
        "learn_text_classification.config_utils.dbutils",
        mock_dbutils,
    )

    return values, mock_dbutils.widgets


def test_load_local_config(config_dir, mock_widgets):
    _values, _ = mock_widgets

    config = load_config(config_dir)

    assert config.ingest.url == "local-url"
    assert config.embedding.batch_size == 16
    assert config.tracking.save_model is False
    assert config.experiments == {}


def test_load_sandbox_config(config_dir, mock_widgets):
    values, _ = mock_widgets
    values["profile"] = "sandbox"

    config = load_config(config_dir)

    assert config.ingest.url == "sandbox-url"
    assert config.ingest.volume == Path("/Volumes/catalog/schema/volume")
    assert config.embedding.cache_folder == Path("/Volumes/catalog/schema/volume/cache")
    assert config.embedding.batch_size == 16
    assert config.tracking.save_model is True


def test_load_experiment_config(config_dir, mock_widgets):
    values, _ = mock_widgets
    values["config"] = "baseline"

    config = load_config(config_dir)

    experiment = config.get_experiment("baseline")

    assert experiment is not None
    assert experiment.run == "baseline-run"
    assert experiment.params["C"] == 1.0


def test_experiment_config_overrides_base_config(config_dir, mock_widgets):
    values, _ = mock_widgets
    values["config"] = "override"

    config = load_config(config_dir)

    assert config.embedding.batch_size == 32

    experiment = config.get_experiment("baseline")
    assert experiment is not None
    assert experiment.params["C"] == 2.0
    assert experiment.params["max_iter"] == 100


def test_experiment_config_merges_params(config_dir, mock_widgets):
    values, _ = mock_widgets
    values["config"] = "override"  # baselineではなくoverrideを読み込む

    (config_dir / "local.yml").write_text(
        """
ingest:
  url: local-url
  volume: /tmp/raw
prepare:
  raw_data: /tmp/raw
  dataset: local-dataset
preprocessing:
  dataset: local-dataset
embedding:
  cache_folder: /tmp/cache
  batch_size: 16
  transfer_batch_size: 8
  embedding_model_name: local-model
  dataset: local-dataset
experiments:
  baseline:
    run: baseline-run
    params:
      C: 1.0
""",
        encoding="utf-8",
    )

    (config_dir / "experiments" / "override.yml").write_text(
        """
experiments:
  baseline:
    params:
      max_iter: 100
""",
        encoding="utf-8",
    )

    config = load_config(config_dir)

    experiment = config.get_experiment("baseline")
    assert experiment is not None
    assert experiment.run == "baseline-run"
    assert experiment.params["C"] == 1.0
    assert experiment.params["max_iter"] == 100


def test_invalid_profile_raises_error(config_dir, mock_widgets):
    values, _ = mock_widgets
    values["profile"] = "invalid"

    with pytest.raises(ValueError, match="Invalid profile"):
        load_config(config_dir)


def test_missing_profile_config_raises_error(tmp_path, mock_widgets):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_config(config_dir)


def test_missing_experiment_file_raises_error(config_dir, mock_widgets):
    values, _ = mock_widgets
    values["config"] = "missing"

    with pytest.raises(FileNotFoundError, match="Experiment config file"):
        load_config(config_dir)


def test_missing_required_config_raises_error(tmp_path, mock_widgets):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "local.yml").write_text(
        """
ingest:
  url: local-url
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Required config values are missing"):
        load_config(config_dir)
