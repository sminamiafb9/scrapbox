# SparkモデルのMLflow管理方法案

## 1. 方針

Spark Connectで学習したSpark MLモデルについて、モデル本体はUnity Catalog Volumeに保存し、MLflowは学習情報とモデル保存先を管理する方式を基本方針とする。

Databricks ServerlessではMLflow Artifact StoreとVolumeの連携に制約が確認されているため、モデル本体とMLflow Trackingの役割を分離する。

| 管理対象           | 管理先                                  | 役割               |
| -------------- | ------------------------------------ | ---------------- |
| Sparkモデル本体     | Unity Catalog Volume                 | モデルファイルの保存       |
| 学習パラメータ        | MLflow Tracking                      | 学習条件の記録          |
| 評価メトリクス        | MLflow Tracking                      | モデル性能の記録         |
| モデル保存先パス       | MLflow Tag                           | Volume上のモデルとの紐付け |
| 学習実行ID         | MLflow Run ID                        | モデルと学習実行の識別      |
| モデルの承認・バージョン管理 | Unity Catalog Model Registry（必要に応じて） | ライフサイクル管理        |

## 2. 全体構成

① Spark Training

Spark Connect / Workspace Compute

Spark MLによるモデル学習

② MLflow Tracking

Run IDを取得

パラメータ・メトリクスを記録

③ Unity Catalog Volume

モデル本体を保存

`/Volumes/catalog/schema/volume/models/<run_id>/`

④ MLflow Tracking

モデル保存先をTagに登録

`model_volume_path`

⑤ Batch Inference

Databricks Serverless Job

MLflow Run IDから保存先を取得し、Sparkモデルをロード

## 3. 学習・保存処理

MLflow Runを開始し、Run IDを取得してからVolumeにモデルを保存する。

これにより、学習後にモデルを一時保存してから別の場所へコピーする必要がなくなる。

```
import mlflowwith mlflow.start_run() as run:    run_id = run.info.run_id    # Spark MLで学習    model = estimator.fit(train_df)    # Volume上の保存先    model_path = (        f"/Volumes/main/ml/models/{run_id}/spark_model"    )    # Sparkモデルを保存    model.write().overwrite().save(model_path)    # MLflowに学習情報を記録    mlflow.log_param("runtime", "databricks-connect")    mlflow.log_param("model_type", "pyspark.ml.connect")    mlflow.log_metric("auc", auc)    # Volume上のモデル保存先をTagに登録    mlflow.set_tag("model_volume_path", model_path)    mlflow.set_tag("model_format", "spark_ml")
```

### Volumeのディレクトリ構成

```
/Volumes/main/ml/models/
    ├── <run_id_1>/
    │   └── spark_model/
    ├── <run_id_2>/
    │   └── spark_model/
    └── <run_id_3>/
        └── spark_model/
```

Run ID単位で保存先を分け、学習済みモデルを上書きしない運用とする。

## 4. 推論処理

推論時はMLflow TrackingからRun IDを指定してTagを取得し、Volume上のモデルをロードする。

```
from mlflow.tracking import MlflowClientfrom pyspark.ml.classification import LogisticRegressionModelclient = MlflowClient()# 対象Runの情報を取得run = client.get_run(run_id)# モデル保存先を取得model_path = run.data.tags["model_volume_path"]# Sparkモデルをロードmodel = LogisticRegressionModel.load(model_path)# 推論predictions = model.transform(input_df)
```

この方式であれば、MLflow Artifactのダウンロードを介さずにSparkモデルをロードできる。

## 5. 実行環境ごとの利用方針

| 実行環境・用途                   | モデル管理・利用方法                             | 方針              |
| ------------------------- | -------------------------------------- | --------------- |
| Workspace Notebook        | MLflow TrackingからRun情報を取得し、Volumeからロード | 対応              |
| Serverless Notebook / Job | MLflow TrackingからTagを取得し、Volumeからロード   | Volumeアクセスを事前検証 |
| UC Python UDF             | Spark Sessionに依存しないPythonモデルを利用        | 別モデルを用意         |
| Model Serving             | Spark Sessionに依存しないPythonモデルを利用        | 別モデルを用意         |

Spark MLモデルはSpark上でのバッチ推論を主用途とする。オンライン推論やUDFについては、必要に応じて別のPython実装を用意する。

## 6. MLflow Artifactとしてモデルを保存する方式との比較

| 比較項目           | MLflow Artifact管理     | Volumeパス管理           |
| -------------- | --------------------- | -------------------- |
| モデル本体の保存先      | MLflow Artifact Store | Unity Catalog Volume |
| MLflowによるモデル管理 | 標準機能を利用しやすい           | Tagなどで独自管理           |
| Serverlessとの相性 | Artifact Storeの制約を受ける | Volumeアクセス可否に依存      |
| モデルのロード        | Artifactを取得してロード      | Volumeから直接ロード        |
| モデルの重複コピー      | 環境によって発生              | 原則不要                 |
| 実装の自由度         | MLflowの仕組みに依存         | 高い                   |

現時点ではVolumeパス管理方式を第一候補とする。

ただし、Workspace ComputeではMLflow Artifact StoreとしてVolumeを利用できることが確認されているため、ServerlessからMLflow Artifactを直接ロードできるかどうかは、別途比較検証する余地がある。

## 7. 事前検証項目

実装に進む前に、以下を確認する。

- ServerlessのSpark Connectから `/Volumes/...` を参照できること
- `LogisticRegressionModel.load()` でVolume上のモデルをロードできること
- MLflow TrackingからRun IDを指定してTagを取得できること
- Serverless Jobの実行ユーザーにVolumeの読み取り権限があること
- モデル保存中に処理が失敗した場合、不完全なモデルを推論対象にしないこと
- モデルの削除・世代管理の運用を定義すること

## 8. 結論



推奨案

&#x20;Sparkモデル本体はVolume、学習情報とモデル保存先の管理はMLflow Trackingに分離する。



この構成により、以下を実現できる。

1. ServerlessのMLflow Artifact Storeに関する制約を回避できる可能性がある。
2. SparkモデルをVolumeから直接ロードできれば、Artifactのダウンロードやコピーが不要になる。
3. MLflow Run IDを起点に、学習条件・評価結果・モデル本体を紐付けられる。
4. バッチ推論とオンライン推論でモデル実装を分離できる。

最終的な採用判断は、ServerlessからVolume上のSparkモデルを直接ロードできるかを確認したうえで行う。

なお、モデル本体をMLflow Artifactとして管理する方式も完全に捨てる必要はない。ServerlessからArtifactを直接利用できることが確認できれば、標準的なMLflowモデル管理に寄せる選択肢も残しておく。