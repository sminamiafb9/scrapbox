# learn_text_classification

## Development

### Requirements

- Python
- uv
- google-colab-cli

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
| google-colab-cli | Google Colab連携 |

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

### Google Colab連携

- colab cliを用いてローカル開発 → colabでの実行の流れを想定する
- vm構築/ファイル転送/実行をpoeタスクとして設定しており、以下のコマンドで実行できる

```sh
# vmの起動とファイルの転送
uv run poe colab_deploy

# 実行
uv run poe colab_run {path/to/py or ipynb}

# 停止
uv run poe colab_stop  
```

#### カーネルとして利用する

- colabのVMで動作するjupyterにポートフォワーディングで接続させvscodeのinteractiveモードでのカーネルとして利用する

```sh
uv run poe colab_connect
# ポートフォワーディング状態で待機するためターミナルを1プロセス使う
```

- 起動状態でvscodeのinteractiveモードのカーネル選択を行う

```text
1. カーネル選択
2. 既存のjupyterサーバーを選択
3. 127.0.0.1を選択
4. カーネルを選択
```
