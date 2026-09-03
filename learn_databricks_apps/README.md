# learn databricks apps

## setup

[Databricks CLI をインストールまたは更新する](https://docs.databricks.com/aws/ja/dev-tools/cli/install)

```bash
brew tap databricks/tap
brew trust databricks/tap
brew install databricks
```

### login

```bash
HOST=https://<databricks-instance>.cloud.databricks.com
databricks auth login --host $HOST
```

