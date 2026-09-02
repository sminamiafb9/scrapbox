# Step1

GradioのQuickStartを起動する

## ローカル環境での起動

```bash
uv run python -m apps.main
# or
uv run dbx-apps
# or
uv run databricks apps run-local
```

## 初回Appsの作成

```bash
databricks apps create \
    example-step1 \
    --git-url "https://github.com/sminamiafb9/scrapbox" \
    --git-provider gitHub

# appsは作されるが、source code pathを指定してもエラーになるためdeployを別途実施

databricks apps deploy \
    example-step1 \
    --git-branch main  \
    --git-source-code-path "learn_databricks_apps/step1"
```
