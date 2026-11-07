# Databricks Job / Tracking リファクタリング計画書

**Version:** 3.0  
**作成日:** 2026-11-07  
**対象:** Databricks Job / Model Training / Hyperparameter Tuning / Tracking / Spark ML Model Management

---

## 1. 概要

### 1.1 背景

現在のDatabricks Jobは、以下の構造で構成されている。

```text
05_model_training.py
  └── 21_hyperparameter_tuning.py
        └── 31_logistic_regression.py
              └── tracking_utils.py
```

今後、PyTorch、scikit-learn、TensorFlow、Transformers、Spark MLなど、異なるライブラリを利用したTrainerの追加を想定している。

そのため、モデル固有の処理とTrackingの実装詳細を分離し、保守性と拡張性を高める必要がある。

また、Spark Connectで学習したSpark MLモデルについては、Databricks ServerlessにおけるMLflow Artifact Storeの利用制約を考慮する必要がある。

Spark MLモデルについては、モデル本体をUnity Catalog Volumeに保存し、MLflow Trackingには学習情報とモデル保存先を記録する方式を基本方針とする。

### 1.2 リファクタリングの目的

1. TrainerからMLflowへの直接依存をなくす。
2. TrainerとTunerの出力形式を共通化する。
3. RunnerがTrainerとTunerを区別せず実行できるようにする。
4. Tuner内部でTrialとChild Runを管理する。
5. LocalとDatabricksでTracking Backendを切り替えられるようにする。
6. モデル固有のArtifact情報を上流に引き渡せるようにする。
7. Spark MLモデルをUnity Catalog Volumeで管理できるようにする。
8. MLflow Run IDを起点として学習情報とモデル本体を紐付ける。
9. 実行失敗を共通の出力形式で通知できるようにする。

### 1.3 実行環境の制約

| 実行環境 | Trackingの動作 |
|---|---|
| Local | MLflowへ接続せず、Tracking情報をConsoleに出力する |
| Databricks | MLflowにTracking情報を登録する |

Spark MLモデルについては、Unity Catalog Volumeへの保存とロードを基本とする。ただし、Serverless環境からのアクセスについては事前検証を完了条件とする。

---

## 2. 基本設計方針

### 2.1 TrainerとTunerの共通化

TunerもTrainerの一種として扱う。

Runnerから見ると、実行対象が通常のTrainerなのかTunerなのかを区別する必要はない。

両者とも以下の共通契約を持つ。

```text
Parameters
    |
    v
Trainer / Tuner
    |
    v
TrainerOutputs
```

Runnerは実行対象から `TrainerOutputs` を受け取り、その結果をTrackingする。

### 2.2 Trackingの責務分担

| 対象 | 責務 |
|---|---|
| Runner | Parent Runの管理、最終結果のTracking |
| Tuner | Trialの管理、Child Runの管理、Trial結果のTracking |
| Trainer | モデル学習、モデル保存、Tracking情報の生成と返却 |
| Tracking Backend | Tracking情報の保存処理、RunとArtifact参照情報の紐付け |

RunnerはChild Runの存在を知らない。

また、Runnerは実行対象がTunerであるかどうかも知らない。

TunerはMLflow Trackingに対する部分的な知識を持ち、Child Runを管理する。

TrainerはMLflowを知らない。

### 2.3 Runの階層

MLflowのRun階層は親子2層とする。

```text
Parent Run
  |
  +-- Child Run: Trial 0
  |
  +-- Child Run: Trial 1
  |
  +-- Child Run: Trial N
```

- Parent RunはRunnerが管理する。
- Child RunはTunerが管理する。
- 通常のTrainerを直接実行する場合はChild Runを作成しない。

### 2.4 Model Artifactの管理方針

Model Artifactの管理方式はモデルの種類によって異なる。

| モデル種別 | モデル本体の保存先 | MLflowでの管理 |
|---|---|---|
| Spark ML | Unity Catalog Volume | 保存先PathをTagに登録 |
| PyTorch / sklearn等 | 各Trainerの保存方式に従う | 共通Artifact参照情報を利用 |
| その他 | モデル固有の保存方式 | 共通Artifact参照情報を利用 |

Spark MLモデルについては、MLflow Artifact Storeにモデル本体を保存することを必須としない。

モデル本体とTracking情報を分離することで、ServerlessにおけるArtifact Storeの制約を回避できる可能性を確保する。

---

## 3. 各レイヤーの責務

### 3.1 Runner（Model Training）

Job全体の実行とParent Runのライフサイクルを管理する。

**責務**

- ExperimentおよびStepの読み込み
- 実行対象Notebookの呼び出し
- Parent Runの開始・終了
- Parent Run IDの実行対象への受け渡し
- Notebookから返却されたJSONの復元
- TrainerOutputsの確認
- 最終結果のParent Runへの登録
- Model Artifact参照情報のTracking
- 実行失敗の検知とJobへの反映

**Runnerが知らないこと**

- 実行対象がTrainerかTunerか
- Trialの存在
- Child Runの存在
- Optunaなどの最適化処理
- モデル固有のPyFunc実装
- Spark MLモデルの内部構造

Runnerは、すべての実行対象を共通のNotebook実行インタフェースで扱う。

### 3.2 Tuner（Hyperparameter Tuning）

TunerはTrainerの一種として扱う。

**責務**

- ハイパーパラメータの生成
- Trainerの呼び出し
- Trialの管理
- Child Runの作成・終了
- Trial結果のTracking
- Trialの失敗管理
- Best Trialの選択
- 最終Trainerの実行
- 最終Trainerの結果をRunnerへ返却

**Trackingに関する責務**

- TrialごとにChild Runを作成する。
- TrialのParametersとMetricsをChild Runに登録する。
- Trialではモデルを保存しない。
- 最終学習結果のTrackingはRunnerに委譲する。

TunerはMLflowのChild Run管理に必要な範囲でMLflowの知識を持つ。

ただし、モデル固有の実装やPyFuncモデルの内部構造は知らない。

### 3.3 Trainer

Trainerはモデル固有の詳細を知る唯一のレイヤーとする。

**責務**

- Parametersの受け取り
- データ取得
- モデル学習
- モデル評価
- 必要に応じたモデル保存
- Tracking情報の生成
- ModelArtifact参照情報の生成
- TrainerOutputsの返却

**禁止事項**

- MLflow APIの直接呼び出し
- Run IDの管理
- Parent Run・Child Runの管理
- ExperimentやRun名の管理
- Tracking Backendへの依存

Trainerはモデル固有のPyFunc実装やSpark MLモデルの保存・ロード方式を知っていてよい。

ただし、その実装や保存方法を上流のRunner・Tunerに漏らさず、共通のArtifact情報として返却する。

Spark MLモデルの場合、Trainerは指定されたVolume上にモデルを保存し、その保存先URIをModelArtifactとして返す。

### 3.4 Tracking Backend

Trackingの保存処理を実行環境に応じて切り替える。

| Backend | 実行環境 | 責務 |
|---|---|---|
| Console Backend | Local | Tracking情報のログ出力 |
