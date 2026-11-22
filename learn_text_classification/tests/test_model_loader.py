from unittest.mock import MagicMock, patch

import pytest

from learn_text_classification.spark_utils import model_loader


def test_load_model_from_volume() -> None:
    model_uri = "/Volumes/catalog/schema/volume/model"
    spark = MagicMock()

    expected = MagicMock()

    with (
        patch.object(
            model_loader.BinaryRelevanceInferenceModel,
            "load",
            return_value=expected,
        ) as load_binary_relevance,
        patch.object(
            model_loader.LogisticRegressionModel,
            "load",
        ) as load_logistic_regression,
    ):
        result = model_loader.load_model(model_uri, spark)

    assert result is expected
    load_binary_relevance.assert_called_once_with(
        model_uri,
        spark,
        load_logistic_regression,
    )


def test_load_model_unsupported_uri() -> None:
    spark = MagicMock()

    with pytest.raises(ValueError, match="Unsupported model URI"):
        model_loader.load_model("unsupported://model", spark)
