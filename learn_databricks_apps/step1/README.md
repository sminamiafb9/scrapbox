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

- 初回はDatabricks AppsのUIにある \[+ アプリを作成\] ボタンから作成
- Githubリポジトリを指定しておくと自動的にClone
- アプリ作成後にデプロイからapps.yamlのあるディレクトリ設定など実施
