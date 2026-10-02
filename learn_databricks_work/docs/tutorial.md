# tutorial

## databricks jobとして動かす

```sh
databricks auth login -p sandbox
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev learn_databricks_work
```

タスクとして設定
```sh
databricks auth login -p sandbox
uv run poe bundle_deploy
uv run poe bundle_run
```

### 関連ファイル

- databricks.yml: targetsの指定とresourcesの指定
- resources/job.yml: taskの指定


## databricks connectでローカルからDatabricksのSparkに接続する

### ライブラリを追加

```sh
uv add databricks-connect
```

メインのコードも使うためdependenciesに追加する

### ~/.databrickscfgの編集

DEFAULTプロファイルに以下を追記

```toml
serverless_compute_id = auto
```

### コードの編集

```py
from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()
```

上記のコードでsparkのインスタンスが取得される

### notebook実行環境

- 拡張環境からDatabricksのパッケージ構成をpyproject.tomlに追記できるので、依存衝突を解決しつつvenvを作る
- ipykernelを入れておくとvscodeでnotebookを実行しつつ、spark実行もできるので概ね開発端末で作業ができるようになる
