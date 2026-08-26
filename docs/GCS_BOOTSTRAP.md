# Google Cloud Storage Bootstrap

> 当前工程以 GCS 作为长期数据/模型对象仓库。Bucket 创建一次后，后续所有 Batch、Dataset、Model 都通过版本化路径写入。

## 1. 必需变量

```bash
export PROJECT_ID="<your-gcp-project-id>"
export REGION="asia-east1"
export BUCKET="<globally-unique-yujian-bucket-name>"
```

建议 Bucket 名包含项目 ID，例如：`yujian-ai-${PROJECT_ID}`。

## 2. Cloud Shell 一次性初始化

在 Google Cloud Console -> Cloud Shell 执行：

```bash
gcloud config set project "$PROJECT_ID"
gcloud services enable storage.googleapis.com run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcloud storage buckets create "gs://$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access

gcloud storage buckets update "gs://$BUCKET" --versioning
```

## 3. 推荐生命周期策略

- `raw/`：长期保留原始数据；不覆盖。
- `reviewed/`：长期保留。
- `datasets/`：冻结后永久保留。
- `models/`：永久保留 Candidate/Production；实验模型后续可加生命周期清理。
- `temp/`：可设置 30 天自动删除。

## 4. 服务账号

建议创建专用账号：

```bash
gcloud iam service-accounts create yujian-model-factory \
  --display-name="YuJian Model Factory"
```

最小权限原则：

- 运行采集/导入服务：`roles/storage.objectAdmin`（后续可进一步拆细）。
- 只读训练/评估任务：`roles/storage.objectViewer`。

优先使用 Workload Identity Federation / Cloud Run 身份，避免长期 JSON Key。

## 5. 数据写入规则

任何新采集数据必须先形成：

```text
BATCH_<YYYYMMDD>_<SOURCE>_<NNN>
```

例如：

```text
BATCH_20260826_WB_001
```

然后写入：

```text
gs://$BUCKET/raw/batches/BATCH_20260826_WB_001/
```

禁止直接把图片复制进 `datasets/`。Dataset 只能由审核/标注后的 Batch 冻结生成。

## 6. GitHub 与 GCP 分工

GitHub 保存：
- 代码
- Schema
- 配置
- Pipeline
- Dataset/Model Registry 的可审计定义

GCS 保存：
- 原始图片
- 审核图片
- 标注导出
- Dataset 快照
- 模型权重
- Evaluation 产物

## 7. 下一阶段

完成 Bucket 后，将 Bucket 名与 Project ID 写入部署环境变量，不要提交凭证到 GitHub。
