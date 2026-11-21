from pathlib import Path

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
def mock_job_options(monkeypatch):
    values = {
        "profile": "local",
        "config": "",
    }

    monkeypatch.setattr(
        "learn_text_classification.config_utils.get_job_option",
        lambda name, default="": values.get(name, default),
    )

    return values


def test_load_local_config(config_dir: Path, mock_job_options):
    config = load_config(config_dir)

    assert config.ingest.url == "local-url"
    assert config.ingest.volume == "/tmp/raw"
    assert config.embedding.batch_size == 16
    assert "experiments" not in config


def test_load_sandbox_config(config_dir: Path, mock_job_options):
    mock_job_options["profile"] = "sandbox"

    config = load_config(config_dir)

    assert config.ingest.url == "sandbox-url"
    assert config.ingest.volume == "/Volumes/catalog/schema/volume"
    assert config.prepare.raw_data == "/tmp/raw"
    assert config.embedding.cache_folder == "/Volumes/catalog/schema/volume/cache"
    assert config.embedding.batch_size == 16
    assert config.tracking.save_model is True


def test_load_experiment_config(config_dir: Path, mock_job_options):
    mock_job_options["config"] = "baseline"

    config = load_config(config_dir)

    experiment = config.experiments.baseline

    assert experiment.run == "baseline-run"
    assert experiment.params.C == 1.0


def test_experiment_config_overrides_base_config(
    config_dir: Path,
    mock_job_options,
):
    mock_job_options["config"] = "override"

    config = load_config(config_dir)

    assert config.embedding.batch_size == 32

    experiment = config.experiments.baseline

    assert experiment.run == "baseline-run"
    assert experiment.params.C == 2.0
    assert experiment.params.max_iter == 100


def test_experiment_config_merges_params(
    config_dir: Path,
    mock_job_options,
):
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

    mock_job_options["config"] = "override"

    config = load_config(config_dir)

    experiment = config.experiments.baseline

    assert experiment.run == "baseline-run"
    assert experiment.params.C == 1.0
    assert experiment.params.max_iter == 100


def test_invalid_profile_raises_error(
    config_dir: Path,
    mock_job_options,
):
    mock_job_options["profile"] = "invalid"

    with pytest.raises(ValueError, match="Invalid profile"):
        load_config(config_dir)


def test_missing_profile_config_raises_error(
    tmp_path: Path,
    mock_job_options,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(config_dir)


def test_missing_experiment_file_raises_error(
    config_dir: Path,
    mock_job_options,
):
    mock_job_options["config"] = "missing"

    with pytest.raises(FileNotFoundError, match="Experiment config file not found"):
        load_config(config_dir)


def test_missing_required_config_raises_error(
    config_dir: Path,
    mock_job_options,
):
    (config_dir / "local.yml").write_text(
        """
ingest:
  url: ???
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Required config values are missing",
    ):
        load_config(config_dir)
