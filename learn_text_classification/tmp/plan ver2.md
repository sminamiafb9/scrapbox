# Databricks Job / Tracking リファクタリング計画書

**Version:** 2.0  
**作成日:** 2026-11-07  
**対象:** Databricks Job / Model Training / Hyperparameter Tuning / Tracking

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

今後、PyTorch、scikit-learn、TensorFlow、Transformersなど、異なるライブラリを利用したTrainerの追加を想定している。

そのため、モデル固有の処理とTrackingの実装詳細を分離し、保守性と拡張性を高める必要がある。

### 1.2 リファクタリングの目的

1. TrainerからMLflowへの直接依存をなくす。
2. TrainerとTunerの出力形式を共通化する。
3. RunnerがTrainerとTunerを区別せず実行できるようにする。
4. Tuner内部でTrialとChild Runを管理する。
5. LocalとDatabricksでTracking Backendを切り替えられるようにする。
6. モデル固有のArtifact情報を上流に引き渡せるようにする。
7. 実行失敗を共通の出力形式で通知できるようにする。

### 1.3 実行環境の制約

| 実行環境 | Trackingの動作 |
|---|---|
| Local | MLflowへ接続せず、Tracking情報をConsoleに出力する |
| Databricks | MLflowにTracking情報を登録する |

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

Trackingの責務は次のように分ける。

| 対象 | 責務 |
|---|---|
| Runner | Parent Runの管理、最終結果のTracking |
| Tuner | Trialの管理、Child Runの管理、Trial結果のTracking |
| Trainer | Tracking情報の生成と返却 |
| Tracking Backend | Tracking情報の保存処理 |

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
- 実行失敗の検知とJobへの反映

**Runnerが知らないこと**

- 実行対象がTrainerかTunerか
- Trialの存在
- Child Runの存在
- Optunaなどの最適化処理
- モデル固有のPyFunc実装

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
- TrainerOutputsの返却

**禁止事項**

- MLflow APIの直接呼び出し
- Run IDの管理
- Parent Run・Child Runの管理
- ExperimentやRun名の管理
- Tracking Backendへの依存

Trainerはモデル固有のPyFunc実装を知っていてよい。

その実装や保存方法を上流のRunner・Tunerに漏らさず、共通のArtifact情報として返却する。

### 3.4 Tracking Backend

Trackingの保存処理を実行環境に応じて切り替える。

| Backend | 実行環境 | 責務 |
|---|---|---|
| Console Backend | Local | Tracking情報のログ出力 |
| MLflow Backend | Databricks | MLflowへの登録 |

RunnerはParent Runの管理にBackendを利用する。

TunerはChild Runの管理に必要なTracking機能を利用する。

TrainerはBackendを利用しない。

---

## 4. 実行フロー

### 4.1 Tunerを実行する場合

```text
Runner
  |
  | 1. Parent Run開始
  |
  | 2. dbutils.notebook.run()
  |    Parent Run IDを引数として渡す
  |
  v
Tuner
  |
  | 3. Trial開始
  |    Child Runを作成
  |
  +-----------------------------+
  |                             |
  v                             v
Trial 0                       Trial 1
  |                             |
  | runpy.run_path()            | runpy.run_path()
  v                             v
Trainer                       Trainer
  |                             |
  | TrainerOutputs              | TrainerOutputs
  v                             v
Tuner                         Tuner
  |                             |
  +-------------+---------------+
                |
                | 4. Trial結果をChild Runに記録
                |    Parameters / Metricsのみ
                |
                v
          Best Parameters
                |
                | 5. Best Parametersで最終学習
                v
          Final Trainer
                |
                | 6. TrainerOutputs
                |    Model Artifactを含む
                v
              Tuner
                |
                | 7. JSON化
                |    dbutils.notebook.exit()
                v
              Runner
                |
                | 8. JSON復元
                |    TrainerOutputsとして扱う
                |
                | 9. Parent Runに最終結果を登録
                |
                | 10. Parent Run終了
                v
              完了
```

### 4.2 Trainerを直接実行する場合

```text
Runner
  |
  | Parent Run開始
  |
  | dbutils.notebook.run()
  v
Trainer
  |
  | runpy.run_path()などによる実行
  v
TrainerOutputs
  |
  | JSON化
  v
Runner
  |
  | Parent RunにTracking
  v
完了
```

通常のTrainerを直接実行する場合、Child Runは作成しない。

---

## 5. TrainerOutputsの構造化

### 5.1 基本方針

TrainerとTunerの共通出力として `TrainerOutputs` を定義する。

出力はJSONに変換可能でなければならない。

TunerからRunnerへのNotebook間連携ではJSON文字列を使用する。

TunerからTrainerへの内部連携ではPython Objectを利用する。

### 5.2 データ構造案

```python
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TrackingRecord:
    params: dict[str, Any] = field(
        default_factory=dict
    )
    metrics: dict[str, float] = field(
        default_factory=dict
    )
    tags: dict[str, str] = field(
        default_factory=dict
    )
    artifacts: list[str] = field(
        default_factory=list
    )


@dataclass
class ModelArtifact:
    uri: str
    format: str
    flavor: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class FailureInfo:
    code: str
    message: str
    details: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TrainerOutputs:
    status: Literal["SUCCEEDED", "FAILED"]
    score: float | None = None
    tracking: TrackingRecord = field(
        default_factory=TrackingRecord
    )
    model: ModelArtifact | None = None
    failure: FailureInfo | None = None
```

上記は初期設計案であり、ModelArtifactの詳細な仕様は別途検討する。

### 5.3 JSON変換

`TrainerOutputs` は明示的な `to_dict()` / `from_dict()` を用意する。

```python
result = outputs.to_dict()

result_json = json.dumps(
    result,
    ensure_ascii=True,
    allow_nan=False,
)
```

復元時は次のようにする。

```python
result_dict = json.loads(result_json)

outputs = TrainerOutputs.from_dict(
    result_dict
)
```

`dataclasses.asdict()` のみでは任意のPython ObjectをJSON化できないため、JSON化対象を明確にする。

以下のようなObjectをそのまま出力に含めない。

- Spark DataFrame
- Spark Model
- PythonModelインスタンス
- ファイルハンドル
- JSON化できない独自クラス
- 任意のライブラリ固有オブジェクト

### 5.4 JSONに含める情報

Runnerに返すのは最終結果のみとする。

| 情報 | JSONに含めるか |
|---|---|
| 最終Parameters | 含める |
| 最終Metrics | 含める |
| 最終Tags | 含める |
| 最終Model Artifact参照 | 含める |
| 成功・失敗状態 | 含める |
| FailureInfo | 失敗時に含める |
| 全Trialの詳細結果 | 原則含めない |
| TrialごとのModel | 含めない |

Trialの詳細はTunerがChild Runに記録する。

---

## 6. Notebook間連携と実行方式

### 6.1 Tuner → Trainer

現状の `runpy.run_path()` による呼び出し方式を維持する。

同一Pythonプロセス内の連携であるため、Python Objectを利用できる。

```python
namespace = runpy.run_path(
    trainer_path,
    init_globals={
        "parameters": trial_parameters,
    },
)

outputs = namespace["outputs"]
```

Trainer側は従来どおり、実行結果をグローバル変数 `outputs` に設定する。

### 6.2 Tuner → Runner

`dbutils.notebook.run()` によるNotebook実行を維持する。

Notebookの終了値はJSON文字列とする。

Tuner側：

```python
dbutils.notebook.exit(
    json.dumps(
        outputs.to_dict(),
        ensure_ascii=True,
        allow_nan=False,
    )
)
```

Runner側：

```python
result_json = dbutils.notebook.run(
    notebook_path,
    timeout_seconds,
    arguments,
)

result_dict = json.loads(result_json)

outputs = TrainerOutputs.from_dict(
    result_dict
)
```

### 6.3 dbutils.notebook.runの仕様

`dbutils.notebook.run()` はNotebookを実行し、`dbutils.notebook.exit()` で返却されたStringを呼び出し元に返す。

戻り値の最大長は5 MBである。

そのため、Notebook間の受け渡しに任意のPython Objectを直接利用することはできない。

JSON文字列に変換可能な辞書やDataclassなどを利用する。

### 6.4 TunerからTrainerへのdbutils.notebook.run利用について

TunerからTrainerへの呼び出しは、現状の `runpy.run_path()` を維持する。

理由は以下のとおり。

- Trainer内部ではPython Objectを受け渡せる。
- TrialごとにNotebook実行境界を増やさずに済む。
- 既存のNotebook単独実行の構造を維持できる。
- Tuner内部でTrialを管理しやすい。

ただし、TrainerをNotebookとして単独実行する用途と、Tunerから呼び出す用途の双方に対応できるよう、Notebook固有の終了処理と学習ロジックは分離する。

---

## 7. Model Artifactの設計

### 7.1 基本方針

Model Artifactの具体的な実装はTrainerが知る。

特にPyFuncモデルの実装は、モデル固有のTrainerが管理する。

RunnerとTunerはPyFuncの内部実装を知らない。

そのため、モデル本体をそのままJSONで受け渡すのではなく、Artifactへの参照情報を共通形式で受け渡す。

### 7.2 Model Artifactの受け渡し

```text
Trainer
  |
  | モデル固有のPyFunc実装
  |
  | モデルを保存・パッケージ化
  |
  v
Model Artifact
  |
  | URI / Format / Flavor / Metadata
  |
  v
Tuner
  |
  | Artifact参照をそのまま引き渡す
  |
  v
Runner
  |
  | 共通形式のArtifactとして登録
  v
MLflow
```

### 7.3 責務

| レイヤー | Model Artifactに関する責務 |
|---|---|
| Trainer | モデル固有の実装、保存、Artifact生成 |
| Tuner | 最終TrainerのArtifact参照を引き渡す |
| Runner | Artifact参照を受け取り、Parent Runに登録 |
| Tracking Backend | Artifactの登録処理 |

### 7.4 詳細検討が必要な事項

以下は本計画の時点では未確定とする。

- PyFuncモデルをどの形式で保存するか。
- TrainerがMLflow形式のModel Directoryまで生成するか。
- Runnerがどの共通APIでモデルを登録するか。
- Artifactの一時保存先をどこにするか。
- LocalとDatabricksでArtifact URIをどう扱うか。
- Model Registryへの登録をTracking Backendに含めるか。

特に、MLflowのModel登録は通常のファイルArtifact登録と異なるため、単純なURIの受け渡しだけで実現できるかを検証する。

この部分は独立した設計項目として詳細化する。

---

## 8. Hyperparameter TuningのTracking設計

### 8.1 TrialのTracking

TrialではParametersとMetricsだけを保存する。

```text
Parent Run
  |
  +-- Child Run: Trial 0
  |     ├── Parameters
  |     └── Metrics
  |
  +-- Child Run: Trial 1
  |     ├── Parameters
  |     └── Metrics
  |
  +-- Child Run: Trial N
        ├── Parameters
        └── Metrics
```

Trialでは以下を保存しない。

- Model
- Model Artifact
- モデルの大容量ファイル

### 8.2 最終学習

Best TrialのParametersを利用してTrainerを再実行する。

```text
Trial 0 ─┐
Trial 1 ─┼── Best Parameters
Trial N ─┘
              |
              v
        Final Trainer
              |
              v
       Final Model Artifact
```

最終学習の結果はTunerからRunnerに返す。

RunnerがParent Runに最終結果を登録する。

### 8.3 Trialのモデル保存制御

Trialではモデルを保存しないことを基本方針とする。

TunerからTrainerを呼び出す際に、実行モードを渡す。

```python
trial_parameters = {
    **parameters,
    "save_model": False,
}

final_parameters = {
    **best_parameters,
    "save_model": True,
}
```

これにより、最終モデルだけを保存する。

---

## 9. 失敗状態の管理

### 9.1 基本方針

失敗状態の管理は、各実行レイヤーが自分の責務の範囲で行う。

| 対象 | 失敗管理 |
|---|---|
| Trainer | 学習処理の失敗を検知し、TrainerOutputsで通知 |
| Tuner | Trialの失敗管理、失敗許容回数の判定、Tuning全体の失敗判定 |
| Runner | TrainerOutputsの失敗を受け取り、JobとParent Runを失敗扱いにする |

### 9.2 Trainerの失敗通知

Trainerは失敗時にも共通の出力形式を返す。

```python
TrainerOutputs(
    status="FAILED",
    failure=FailureInfo(
        code="TRAINING_FAILED",
        message="Training failed.",
    ),
)
```

ただし、プロセス自体が終了するような例外やNotebook実行エラーは、通常の戻り値では返せないため、呼び出し側で例外として処理する。

### 9.3 Tunerの失敗管理

TunerはTrialごとの成功・失敗を管理する。

| 状況 | Tunerの動作 |
|---|---|
| Trial成功 | Child Runを成功として終了 |
| Trial失敗 | Child Runを失敗として終了 |
| 失敗が許容範囲内 | 次のTrialを継続 |
| 失敗回数が閾値に到達 | Tuning全体を失敗にする |
| Tuning成功 | Best Parametersで最終学習を実行 |
| 最終学習失敗 | Tuning全体を失敗にする |

失敗許容回数や判定方法はTunerの設定で管理する。

### 9.4 Runnerの失敗処理

RunnerはTuner内部のTrial失敗を知らない。

Tunerから返された最終的なStatusだけを確認する。

```python
if outputs.status == "FAILED":
    tracking_manager.end_run(
        parent_run_id,
        status="FAILED",
    )

    raise RuntimeError(
        outputs.failure.message
    )
```

また、`dbutils.notebook.run()` 自体が例外を発生させた場合にもParent Runを失敗として終了する。

Notebookが正常終了してJSONでFAILEDを返した場合も、Runner側で明示的に失敗扱いにする。

---

## 10. Tracking Backend

### 10.1 Backendの種類

| Backend | 実行環境 | 処理 |
|---|---|---|
| Console Backend | Local | Tracking情報をログ出力 |
| MLflow Backend | Databricks | MLflowに登録 |

### 10.2 Backendのインタフェース案

```python
class TrackingBackend(Protocol):

    def start_run(
        self,
        name: str,
        parent_id: str | None = None,
    ) -> str:
        ...

    def log_record(
        self,
        run_id: str,
        record: TrackingRecord,
    ) -> None:
        ...

    def log_model(
        self,
        run_id: str,
        artifact: ModelArtifact,
    ) -> None:
        ...

    def end_run(
        self,
        run_id: str,
        status: str,
    ) -> None:
        ...
```

このインタフェースは概念案である。

RunnerとTunerが必要とする機能を整理したうえで、最終的なAPIを確定する。

### 10.3 Backendの選択

```python
def create_tracking_backend(environment):

    if environment == "local":
        return ConsoleTrackingBackend()

    if environment == "databricks":
        return MlflowTrackingBackend()

    raise ValueError(
        f"Unsupported environment: {environment}"
    )
```

Console BackendではMLflow Clientを生成しない。

---

## 11. ディレクトリ構成案

```text
learn_text_classification/
│
├── runners/
│   └── model_training.py
│
├── trainers/
│   ├── base.py
│   ├── hyperparameter_tuning.py
│   │
│   └── models/
│       ├── logistic_regression.py
│       ├── pytorch.py
│       ├── sklearn.py
│       └── tensorflow.py
│
├── tracking/
│   ├── models.py
│   ├── factory.py
│   │
│   └── backends/
│       ├── base.py
│       ├── console.py
│       └── mlflow.py
│
├── execution/
│   ├── trainer_runner.py
│   └── config.py
│
└── utils/
    ├── config_utils.py
    ├── job_utils.py
    └── logging_utils.py
```

### モジュールの責務

| モジュール | 役割 |
|---|---|
| runners/model_training.py | Job全体の実行とParent Run管理 |
| trainers/base.py | Trainerの共通入出力契約 |
| trainers/hyperparameter_tuning.py | Optuna、Trial、Child Run管理 |
| trainers/models/ | 個別モデルの学習処理 |
| tracking/models.py | TrackingRecord、ModelArtifactなど |
| tracking/factory.py | Backend選択 |
| tracking/backends/base.py | Backendインタフェース |
| tracking/backends/console.py | Local用Tracking |
| tracking/backends/mlflow.py | MLflow登録 |
| execution/trainer_runner.py | runpyなどを利用したTrainer実行 |

TunerによるChild Run管理はTuner自身の責務とし、RunnerにTrial管理機能を追加しない。

---

## 12. 設定ファイルの変更

### 12.1 設定例

```yaml
experiments:
  baseline:
    steps:
      training:
        run: hyperparameter_tuning

        params:
          trainer:
            name: logistic_regression
            params:
              max_itr: 30

          optimization:
            direction: maximize
            n_trials: 50
            n_startup_trials: 10

            failure:
              max_failed_trials: 5

            suggest_params:
              regularization:
                method: suggest_float
                params:
                  low: 0.00001
                  high: 10.0
                  log: true

              elastic_net:
                method: suggest_float
                params:
                  low: 0.0
                  high: 1.0

          tracking:
            trials: true
            trial_models: false
            best_model: true
```

### 12.2 設定の責務

| 設定 | 責務 |
|---|---|
| trainer | 実行するTrainerとそのパラメータ |
| optimization | Tunerの探索設定 |
| failure | Trial失敗の許容条件 |
| tracking | Trialや最終ModelのTracking方針 |

Trialモデルは保存しないことを基本方針とする。

---

## 13. リファクタリング実施計画

### Phase 1：共通出力構造の導入

**実施内容**

- TrackingRecordの定義
- TrainerOutputsの定義
- FailureInfoの定義
- ModelArtifactの暫定定義
- JSON変換処理の実装

**完了条件**

TrainerとTunerで共通の出力形式を利用できる。

### Phase 2：Trainerの変更

**実施内容**

- tracking_utilsへの依存を削除
- MLflow関連パラメータを削除
- Tracking情報を戻り値に含める
- runpy.run_pathによる呼び出し方式を維持
- Notebook単独実行との両立

**完了条件**

TrainerがMLflowに依存せず、TrainerOutputsを返却できる。

### Phase 3：Tracking Backendの実装

**実施内容**

- Console Backend
- MLflow Backend
- Backend Factory
- Parent Run管理に必要な機能
- Child Run管理に必要な機能
- Run IDを利用した管理方式

**完了条件**

LocalとDatabricksでBackendを切り替えられる。

### Phase 4：Tunerの変更

**実施内容**

- 共通TrainerOutputsの利用
- TrialごとのChild Run管理
- Trial Parameters・Metricsの登録
- Trialモデル保存の無効化
- Trial失敗管理
- Best Parametersの選択
- 最終Trainerの再実行
- 最終結果のJSON化

**完了条件**

TunerがTrainerと同じ出力契約を持ち、Trial管理を内部で完結できる。

### Phase 5：Runnerの変更

**実施内容**

- Parent Run管理
- dbutils.notebook.runの戻り値のJSON復元
- TrainerOutputsの復元
- 最終Tracking情報の登録
- Model Artifactの登録方式の実装
- 失敗状態の反映

**完了条件**

RunnerがTrainerとTunerを区別せず実行できる。

### Phase 6：Model Artifactの詳細設計・実装

**実施内容**

- PyFuncモデルの保存形式を確定
- Artifact参照形式を確定
- 保存先を確定
- Runner側の登録方式を確定
- Databricks上での結合テスト

**完了条件**

モデル固有の実装をRunnerに持ち込まず、最終モデルをParent Runに登録できる。

### Phase 7：設定・Jobの整理

**実施内容**

- local.ymlの変更
- job.ymlの確認
- 旧tracking_utils.pyの整理
- Local・Databricksの結合テスト

**完了条件**

新しい構造で既存Jobを実行できる。

---

## 14. テスト計画

### 14.1 Trainer単体テスト

| テスト対象 | 確認内容 |
|---|---|
| 学習 | 正常終了する |
| Parameters | 指定値が反映される |
| Score | 評価結果が返却される |
| Tracking | Parameters・Metricsが返却される |
| Model Artifact | 保存時に参照情報が返却される |
| JSON | TrainerOutputsをJSON化できる |
| Failure | 失敗時にFailureInfoが返却される |
| 依存関係 | MLflowに依存しない |

### 14.2 Tuner単体テスト

| テスト対象 | 確認内容 |
|---|---|
| Trial生成 | 指定回数のTrialが生成される |
| Trainer呼び出し | runpy.run_pathで実行できる |
| Parameters | Trialごとに異なる値が渡される |
| Tracking | TrialごとにChild Runが作成される |
| Trial Metrics | MetricsがChild Runに登録される |
| Trial Parameters | ParametersがChild Runに登録される |
| Trial Model | Modelが保存されない |
| Best Trial | 最適Trialが選択される |
| Final Trainer | Best Parametersで再学習する |
| JSON | 最終結果をJSON化できる |
| Failure | 失敗閾値に達した場合にFAILEDを返す |

### 14.3 Runner単体テスト

| テスト対象 | 確認内容 |
|---|---|
| Parent Run | 正常に開始・終了できる |
| Notebook実行 | dbutils.notebook.runで呼び出せる |
| JSON復元 | JSONからTrainerOutputsを復元できる |
| 成功 | Parent Runに最終結果が登録される |
| 失敗 | FAILEDを受け取るとParent Runが失敗する |
| 例外 | Notebook実行例外でもParent Runが失敗する |
| Model | Model Artifactを登録できる |

### 14.4 Local結合テスト

```text
Runner
  |
  v
Tuner
  |
  +-- Trial 0
  +-- Trial 1
  +-- Trial N
  |
  v
Final Trainer
  |
  v
TrainerOutputs
  |
  v
Runner
  |
  v
Console Output
```

確認事項：

- MLflowに接続しない。
- Trial結果がConsoleに出力される。
- Trialモデルが保存されない。
- 最終結果が返却される。
- 失敗が正しく通知される。

### 14.5 Databricks結合テスト

```text
Runner
  |
  v
Parent Run
  |
  v
Tuner
  |
  +-- Child Run: Trial 0
  +-- Child Run: Trial 1
  +-- Child Run: Trial N
  |
  v
Final Trainer
  |
  v
TrainerOutputs
  |
  v
Runner
  |
  v
Parent Run
  |
  +-- Best Parameters
  +-- Best Metrics
  +-- Final Model
```

確認事項：

- Parent Runが作成される。
- TunerがChild Runを作成する。
- TrialではParameters・Metricsのみが登録される。
- 最終モデルだけが保存される。
- 最終結果がParent Runに登録される。
- Tuning失敗時にParent RunとJobが失敗扱いになる。

---

## 15. リスク・未決事項

### 15.1 Model Artifactの受け渡し

最も重要な未決事項。

Trainerが生成したPyFuncモデルを、どのような形式で保存し、どのような共通APIでRunnerが登録するかを詳細設計する。

### 15.2 Artifact保存先

TunerとRunnerは別Notebook実行境界になるため、ローカルの一時ファイルパスをそのまま受け渡す設計は避ける。

共有ストレージ上のURIを利用する方式を検討する。

### 15.3 JSONサイズ

`dbutils.notebook.run()` の戻り値はStringであり、最大長は5 MBである。

Trial結果をすべてJSONに含めず、最終結果だけを返す。

### 15.4 runpyの実行制御

TrainerをNotebookとして単独実行する場合と、Tunerから呼び出す場合の両方に対応する必要がある。

Notebook固有の終了処理を学習ロジックから分離する。

### 15.5 Child Runの管理

TunerがChild Runを管理するため、以下を明確にする。

- Parent Run IDの受け渡し
- Child Runの開始・終了
- Trial失敗時のRun状態
- Tuning全体失敗時の処理
- 途中終了時のRun状態

### 15.6 Trial失敗の判定

失敗許容回数、失敗率、連続失敗回数など、どの条件でTuning全体を失敗にするかは設定仕様として確定する。

---

## 16. ver2での設計判断

| 設計事項 | 採用方針 |
|---|---|
| Tunerの位置付け | Trainerの一種 |
| Trainer / Tunerの出力 | TrainerOutputs |
| Runnerの実行対象認識 | TrainerとTunerを区別しない |
| RunnerのRun管理 | Parent Runのみ |
| TunerのRun管理 | Child Run |
| TrialのTracking | Parameters・Metricsのみ |
| Trialモデル保存 | 保存しない |
| 最終モデル | Best Parametersで再学習して保存 |
| 最終結果のTracking | RunnerがParent Runに登録 |
| TrainerのMLflow依存 | なし |
| TunerのMLflow依存 | Child Run管理に必要な範囲で持つ |
| Tuner → Trainer | runpy.run_path、Python Object |
| Tuner → Runner | dbutils.notebook.run、JSON |
| Trainerの単独実行 | 維持する |
| 失敗通知 | TrainerOutputsに共通化 |
| Trial失敗管理 | Tuner |
| Job失敗管理 | Runner |
| Model Artifact | 共通参照形式で受け渡す。詳細は別途設計 |

---

## 17. まとめ

ver2では、TunerをTrainerの一種として扱い、Runnerから実行対象の内部構造を隠す設計を採用する。

Trackingの責務は次のように分離する。

- **Trainer**：学習処理とTracking情報の生成
- **Tuner**：Trial、Child Run、Trial失敗の管理
- **Runner**：Parent Runと最終結果のTracking
- **Tracking Backend**：Tracking情報の保存

実行境界については次の方針とする。

- TunerとTrainerの内部連携は `runpy.run_path()` とPython Objectを維持する。
- TunerとRunnerの連携は `dbutils.notebook.run()` とJSONを利用する。
- TrainerOutputsはJSON化可能な構造とする。

モデル保存については、Trialでは保存せず、Best Parametersによる最終学習時だけモデルを保存する。

Model Artifactの詳細な形式と登録方式は、今後の独立した設計課題とする。

本計画をリファクタリング計画の **ver2** とする。