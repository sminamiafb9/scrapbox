# learn text classification

## MLOps Pipeline

### 1. Data Engineering

データの取得から加工までを行い、モデル開発に利用できるデータセットを作成する。

- **Data Ingestion**
  - 外部データソースからデータを取得する。
  - データのダウンロード、API・データベースからの取得、ファイルの読み込みなどを行う。
  - 取得元や取得日時などのメタデータを管理する。

- **Data Preparation**
  - 取得したデータを整理し、分析・加工に適した状態にする。
  - データの結合、重複データの除去、不要データの除外、フォーマットの統一などを行う。
  - 必要に応じて、欠損値やデータ品質の問題を検出する。

- **Data Profiling**
  - データの構造、品質、統計的な特徴を把握する。
  - データ型、欠損率、値の分布、外れ値、クラス分布、特徴量間の相関などを分析する。
  - 分析結果をレポートとして出力し、データの問題や前処理の必要性を把握する。

- **Data Preprocessing**
  - モデルへの入力に適した形式にデータを変換する。
  - 欠損値補完、カテゴリ変数のエンコーディング、数値特徴量のスケーリング、型変換などを行う。
  - 学習時と推論時で同一の変換処理を再現できるようにする。
  - データリークを防ぐため、必要な変換パラメータは学習データから推定する。

### 2. Model Development

準備したデータを用いてモデルの学習・評価を行い、採用候補となるモデルを管理する。

- **Data Split**
  - データセットを Training、Validation、Test に分割する。
  - モデルの学習、ハイパーパラメータの選択、最終性能の評価にそれぞれ適したデータを用意する。
  - 必要に応じて、層化分割、時系列分割、グループ単位の分割などを適用する。
  - データリークを防ぎ、評価結果の再現性を確保する。

- **Model Training**
  - 選択したモデルとハイパーパラメータを用いて学習を実行する。
  - 学習設定、乱数シード、実行環境、使用データのバージョンなどを記録する。
  - 学習済みモデルを成果物（Model Artifact）として保存する。
  - 必要に応じてチェックポイントや学習ログを保存する。

- **Model Evaluation**
  - 学習済みモデルの性能を評価する。
  - Accuracy、Precision、Recall、F1-score、AUC、RMSE など、タスクに適した評価指標を算出する。
  - ベースラインモデルや既存モデルと比較し、採用基準を満たしているかを確認する。
  - 最終性能の評価には、ハイパーパラメータ選択に使用していない Test データを用いる。
  - 評価結果をレポートとして出力し、モデルの採否を判断する。

- **Model Registry**
  - 評価済みモデルと関連情報を一元管理する。
  - モデルのバージョン、学習データ、コード、ハイパーパラメータ、評価指標などを記録する。
  - 評価結果に基づき、採用候補モデルを登録する。
  - 必要に応じて、モデルの再現性や過去の実験結果を追跡できるようにする。

## Development

### Requirements

- Python
- uv
- databricks (コマンドとvscode拡張)

Python環境および依存管理には `uv` を利用する。

---

### Setup

#### Install dependencies

```bash
uv sync
```

---

#### Run

```bash
uv run <command>
```

### Development Tools

本プロジェクトでは、開発環境とコード品質を維持するために以下のツールを利用します。

| Tool | Purpose |
| --- | --- |
| uv | Python環境・依存管理 |
| Ruff | Lint / Format |
| Pyright | Static Type Check |
| pytest | Test Framework |
| pytest-cov | Test Coverage Measurement |
| pytest-randomly | Randomized Test Execution |
| poethepoet | Development Task Runner |
| databricks-connect | Databricksクラスター接続・実行 |

---

#### Lint/Format

```bash
uv run poe lint
uv run poe format
```

---

#### Type Check


```bash
uv run poe typecheck
```

---

#### Test

```bash
uv run poe test
```

テスト実行時には `pytest-cov` により `learn_text_classification` のカバレッジを計測します。結果はターミナルに表示され、HTML レポートは `htmlcov/index.html` に出力されます。

`pytest-randomly` によりテストはランダムな順序で実行されます。順序に依存しないテストを維持してください。

---

#### Build

```bash
uv run poe build
```

ビルド成果物は `dist/` に出力されます。

---

#### Task Runner

[poethepoet] を利用して開発コマンドを管理します。

開発時に必要なコマンドを `pyproject.toml` に定義し、実行手順をコード化します。

```bash
# Lint / Format / Type Check / Test / Build を実行
uv run poe check
```

---

### Databricks連携

- ローカル環境がdatabricks環境パッケージ群と一致している
- vscodeのinteractiveモードでの開発 → deploy → job実行という流れを想定している
- deploy/runをpoeタスクとして設定しており、以下のコマンドで実行できる

```sh
databricks auth login -p sandbox
uv run poe bundle_deploy
uv run poe bundle_run
```

#### ~/.databrickscfgの編集

databricks-connect実行のため、DEFAULTプロファイルに以下を追記

```toml
serverless_compute_id = auto
```
