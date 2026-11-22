from pyspark.ml import Transformer
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession

from .inference_model import BinaryRelevanceInferenceModel


def load_model(model_uri: str, spark: SparkSession) -> Transformer:
    if model_uri.startswith("/Volumes/"):
        return BinaryRelevanceInferenceModel.load(
            model_uri,
            spark,
            LogisticRegressionModel.load,
        )

    raise ValueError(f"Unsupported model URI: {model_uri}")
