from collections.abc import Callable

from pyspark.ml import Model, Transformer
from pyspark.sql import DataFrame, SparkSession

ModelLoader = Callable[[str], Model]


class MultiLabelClassificationInferenceModel(Transformer):
    OUTPUT_ACCESSORS = (
        "getPredictionCol",
        "getProbabilityCol",
        "getRawPredictionCol",
    )

    def __init__(self, models: dict[str, Model]):
        super().__init__()

        if not models:
            raise ValueError("models must not be empty")

        self.models = models

    @staticmethod
    def _get_output_columns(model: Model) -> list[str]:
        return [
            getattr(model, accessor)()
            for accessor in MultiLabelClassificationInferenceModel.OUTPUT_ACCESSORS
            if hasattr(model, accessor)
        ]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        result = dataset

        for label, model in self.models.items():
            transformed = model.transform(dataset)
            columns = [c for c in self._get_output_columns(model) if c in transformed.columns]

            predictions = transformed.select("Id", *columns)

            for column in columns:
                predictions = predictions.withColumnRenamed(column, f"{label}_{column}")

            result = result.join(predictions, on="Id", how="left")

        return result

    def save(self, path: str, spark: SparkSession) -> None:
        manifest: list[tuple[str, str]] = []

        for i, (label, model) in enumerate(self.models.items()):
            model_dir = str(i)
            model.write().overwrite().save(f"{path}/models/{model_dir}")  # type: ignore
            manifest.append((label, model_dir))

        spark.createDataFrame(manifest, ["label", "model_dir"]).write.mode("overwrite").parquet(f"{path}/manifest")

    @classmethod
    def load(
        cls, path: str, spark: SparkSession, model_loader: ModelLoader
    ) -> "MultiLabelClassificationInferenceModel":
        rows = spark.read.parquet(f"{path}/manifest").collect()
        models = {row.label: model_loader(f"{path}/models/{row.model_dir}") for row in rows}
        return cls(models)
