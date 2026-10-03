# learn_colab_work

## Development

### Requirements

- Python
- uv

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

テスト実行時には `pytest-cov` により `learn_colab_work` のカバレッジを計測します。結果はターミナルに表示され、HTML レポートは `htmlcov/index.html` に出力されます。

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
