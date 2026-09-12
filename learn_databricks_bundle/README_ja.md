# learn_databricks_bundle

`learn_databricks_bundle` プロジェクトは、デフォルトの Python テンプレートを使用して生成されています。

* `src/`: プロジェクトの Python ソースコード

  * `src/learn_databricks_bundle/`: ジョブやパイプラインから利用できる共通 Python コード
* `resources/`: リソース設定（ジョブ、パイプラインなど）
* `tests/`: 共通 Python コードのユニットテスト
* `fixtures/`: データセット用のフィクスチャ（主にテストで使用）

## はじめに

このプロジェクトの開発方法を選択してください。

### (a) Databricks ワークスペース上で直接開発する

以下を参照してください。

https://docs.databricks.com/dev-tools/bundles/workspace

### (b) Cursor や VS Code などの IDE を使用してローカルで開発する

以下を参照してください。

https://docs.databricks.com/dev-tools/vscode-ext.html

### (c) コマンドラインツールを使用する

以下を参照してください。

https://docs.databricks.com/dev-tools/cli/databricks-cli.html

IDE を使用して開発する場合は、`uv` を使ってこのプロジェクトの依存関係をインストールしてください。

* **uv パッケージマネージャーがインストールされていることを確認します。**

  `uv` は `pip` などの代替となる Python パッケージ管理ツールです。

  https://docs.astral.sh/uv/getting-started/installation/

* **以下のコマンドを実行して、プロジェクトの依存関係をインストールします。**

  ```bash
  uv sync --dev
  ```

# CLI を使用してこのプロジェクトを操作する

Databricks ワークスペースや IDE 拡張機能には、このプロジェクトを操作するためのグラフィカルなインターフェースが用意されています。

一方で、CLI を直接使用してこのプロジェクトを操作することもできます。

## 1. Databricks ワークスペースに認証する

まだ認証していない場合は、以下のコマンドを実行します。

```bash
databricks configure
```

## 2. 開発環境用のコピーをデプロイする

以下のコマンドを実行します。

```bash
databricks bundle deploy --target dev
```

`dev` はデフォルトのターゲットであるため、ここでは `--target` パラメータを省略することもできます。

このコマンドを実行すると、このプロジェクトで定義されているすべてのリソースがデプロイされます。

たとえば、デフォルトのテンプレートでは、以下のようなパイプラインが Databricks ワークスペースにデプロイされます。

```text
[dev yourname] learn_databricks_bundle_etl
```

デプロイされたリソースは、Databricks ワークスペースを開き、**Jobs & Pipelines** をクリックすると確認できます。

## 3. 本番環境用のコピーをデプロイする

以下のコマンドを実行します。

```bash
databricks bundle deploy --target prod
```

デフォルトのテンプレートには、パイプラインを毎日実行するジョブが含まれています。

このジョブは `resources/sample_job.job.yml` で定義されています。

開発モードでデプロイする場合、スケジュールは一時停止されます。詳細については、以下を参照してください。

https://docs.databricks.com/dev-tools/bundles/deployment-modes.html

## 4. ジョブまたはパイプラインを実行する

ジョブまたはパイプラインを実行するには、`run` コマンドを使用します。

```bash
databricks bundle run
```

## 5. ローカルでテストを実行する

ローカルでテストを実行するには、`pytest` を使用します。

```bash
uv run pytest
```
