# learn_databricks_bundle

このフォルダには、`learn_databricks_bundle` パイプラインのすべてのソースコードが定義されています。

* `explorations/`: このパイプラインで処理するデータを調査・探索するためのアドホックなノートブック。
* `transformations/`: すべてのデータセット定義およびデータ変換処理。
* `utilities/`（任意）: このパイプラインで使用するユーティリティ関数や Python モジュール。
* `data_sources/`（任意）: このパイプラインのソースデータを記述するビュー定義。

## はじめに

まず `transformations` フォルダを確認してください。関連するソースコードの大部分は、このフォルダに配置されています。

* 慣例として、`transformations` 配下の各データセットは、それぞれ個別のファイルに定義されています。
* `sample_trips_learn_databricks_bundle.py` というサンプルを確認し、構文に慣れてください。
  構文の詳細については、以下を参照してください。
  https://docs.databricks.com/dlt/python-ref.html
* Workspace UI を使用している場合は、`Run file` を使用すると、単一の変換処理を実行してプレビューできます。
* CLI を使用している場合は、以下のコマンドを実行すると、単一の変換処理を実行できます。

```bash
databricks bundle run learn_databricks_bundle_etl --refresh sample_trips_learn_databricks_bundle
```

その他のチュートリアルやリファレンスについては、以下を参照してください。

https://docs.databricks.com/dlt
