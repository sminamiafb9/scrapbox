# tutorial

## databricks jobとして動かす

```sh
databricks auth login -p sandbox
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev learn_databricks_work
```

### 関連ファイル

- databricks.yml: targetsの指定とresourcesの指定
- resources/job.yml: taskの指定
