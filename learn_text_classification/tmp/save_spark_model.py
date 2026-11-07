# Databricks notebook source
# 1. Configuration

import uuid

import mlflow
import pandas as pd

PROFILE = "sandbox"

Workspace = "/Workspace/Users/shotaro.minami8e40a5@gmail.com/.mlflow"
EXPERIMENT_NAME = f"{Workspace}connect-verification"

RUN_NAME = f"local-connect-{uuid.uuid4().hex[:8]}"

# tracking_utils.py と同じ接続設定
mlflow.set_tracking_uri(f"databricks://{PROFILE}")
mlflow.set_registry_uri("databricks-uc")

mlflow.set_experiment(EXPERIMENT_NAME)

print("Tracking URI:", mlflow.get_tracking_uri())
print("Registry URI:", mlflow.get_registry_uri())

# COMMAND ----------
# 2. Connect to Databricks

from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()

print("Spark version:", spark.version)
print("Spark session:", type(spark))

# COMMAND ----------
# 3. Create synthetic data
from pyspark.ml.functions import array_to_vector

rows = []

for i in range(100):
    x1 = float((i % 10) - 5)
    x2 = float(((i * 7) % 13) - 6)
    label = float(1 if x1 + 0.7 * x2 > 0 else 0)
    rows.append(([x1, x2], label))

df = spark.createDataFrame(
    rows,
    schema="features array<double>, label double",
)

# array<double> を VectorUDT に変換
df = df.withColumn("features", array_to_vector("features"))

train_df, test_df = df.randomSplit([0.8, 0.2], seed=123)

print("Train:", train_df.count())
print("Test:", test_df.count())

df.printSchema()

# COMMAND ----------
# 4. Train Spark Connect Logistic Regression

from pyspark.ml.classification import LogisticRegression

estimator = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
)

model = estimator.fit(train_df)

# COMMAND ----------
# 5. Save Spark Model locally
from pathlib import Path

from pyspark.ml.classification import LogisticRegressionModel

Volume = Path("/Volumes/sandbox/shotaro_minami8e40a5/learn_text_classification/.tmp/")

spark_model_dir = Volume / "spark_model"

model.write().save(str(spark_model_dir))

loaded_spark_model = LogisticRegressionModel.load(str(spark_model_dir))

print("Spark model saved:", spark_model_dir)
print("Loaded model:", type(loaded_spark_model))

# COMMAND ----------
# 6. Define PyFunc model

from mlflow.pyfunc import PythonModel, PythonModelContext


class ConnectLogisticRegressionPyFunc(PythonModel):
    def load_context(
        self,
        context: PythonModelContext,
    ) -> None:

        from pyspark.ml.connect.classification import (
            LogisticRegressionModel,
        )
        from pyspark.sql import SparkSession

        self.spark = SparkSession.builder.remote("local[*]").appName("local-pyfunc-inference").getOrCreate()
        artifact_path = str(context.artifacts["spark_model"])

        self.model = LogisticRegressionModel.load(artifact_path)

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.DataFrame:

        records = [(list(map(float, features)),) for features in model_input["features"]]

        spark_df = self.spark.createDataFrame(
            records,
            schema="features array<double>",
        )

        predictions = self.model.transform(spark_df)

        return predictions.select("prediction").toPandas()


# COMMAND ----------

# Sparkクラスタで実行 → Volumeに保存している
# Mlflowはローカルプロセスで動くため実行ファイルシステムに置く必要がある
# (Localの場合ローカル、Databricksの場合Workspace)

from pathlib import Path, PurePosixPath

from databricks.sdk import WorkspaceClient

# ローカル保存先
local_spark_model_dir = Path(".tmp/spark_model")
local_spark_model_dir.mkdir(parents=True, exist_ok=True)

# Databricks SDK
w = WorkspaceClient()


def download_directory(remote_dir: str, local_dir: Path):
    """Volumeのディレクトリをローカルに再帰的にダウンロードする。"""

    local_dir.mkdir(parents=True, exist_ok=True)

    entries = w.files.list_directory_contents(remote_dir)

    for entry in entries:
        remote_path = entry.path
        name = PurePosixPath(remote_path).name
        local_path = local_dir / name

        if entry.is_directory:
            download_directory(remote_path, local_path)
        else:
            w.files.download_to(
                remote_path,
                str(local_path),
            )


# Volumeからローカルへダウンロード
download_directory(
    str(spark_model_dir),
    local_spark_model_dir,
)

print("Spark model downloaded:", local_spark_model_dir)


# COMMAND ----------
# 7. Save PyFunc Model locally

# 一度ローカルでpyfuncモデルを保存する
local_tmp_dir = Path(".mlflow/pyfunc_model")
mlflow.pyfunc.save_model(
    path=str(local_tmp_dir),
    python_model=ConnectLogisticRegressionPyFunc(),
    artifacts={
        "spark_model": str(local_spark_model_dir),
    },
)

assert (local_tmp_dir / "MLmodel").exists()

print(
    "PyFunc model saved:",
)

# COMMAND ----------
# 8. Log artifacts to Databricks MLflow

with mlflow.start_run(run_name=RUN_NAME) as run:
    run_id = run.info.run_id

    mlflow.log_metric("auc", 0.5)
    mlflow.log_param("runtime", "databricks-connect")
    mlflow.log_param("model_type", "pyspark.ml.connect")

    # Sparkモデルの元データもRaw Artifactとして残す場合
    mlflow.log_artifacts(
        str(local_spark_model_dir),
        artifact_path="spark_model_raw",
    )

    # PyFuncモデルとして登録
    # ここで初めてmlflowへpyfuncモデルが登録される
    mlflow.pyfunc.log_model(
        name="pyfunc_model",
        python_model=ConnectLogisticRegressionPyFunc(),
        artifacts={
            "spark_model": str(local_spark_model_dir),
        },
    )

    print("Run ID:", run_id)
