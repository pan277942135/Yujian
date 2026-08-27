# YuJian AI Model Factory

渔见 AI 的长期数据与模型持续迭代工程。

## 目标

统一管理：

`数据采集 Batch -> 清洗/审核 -> 标注 -> Dataset Version -> Training Run -> Model Version -> Evaluation -> Error Pool -> 下一轮数据采集`

## 系统边界

- **GitHub**：代码、配置、Schema、Dataset/Model 版本索引、CI/CD。
- **Google Cloud Storage (GCS)**：原始图片、审核图片、CVAT 导出、冻结数据集、模型权重、评估产物。
- **Model Factory Console**：把日常数据操作从 Cloud Shell 命令转换成网页按钮。

## Model Factory Console V0.1

当前 V0.1 已覆盖主数据链路：

- **Overview**：Batch、图片、审核状态、鱼种分布、Dataset 数量。
- **Batches**：自动发现 `incoming/`；一键执行 `Audit -> Promote -> Registry Sync`。
- **Review**：看图审核；Approve / Reject / Needs Review / Hard Case；修正 Species Truth；保留 ReviewEvent 审计记录。
- **Dataset**：从 `approved` 样本选择 Batch，并冻结不可变 Dataset Version 到 GCS。

页面入口：

```text
/
/batches
/review
/datasets
```

Review 快捷键：

```text
A = Approve
R = Reject
H = Hard Case
N = Needs Review
→ = Skip
```

> 日常操作目标：只使用 Console。Cloud Shell 只用于一次性部署与故障排查。

## 核心 ID

- `batch_id`: 每一批新采集数据，例如 `BATCH_20260826_DB_001`
- `dataset_version`: 每次冻结后的训练数据版本，例如 `DS_M1_v0.1`
- `run_id`: 每次训练，例如 `RUN_M1_0001`
- `model_version`: 每个模型产物，例如 `MODEL_M1_0001`

## GCS 推荐结构

```text
gs://<bucket>/
├── incoming/<upload-folder>/
├── cleaning/<batch_id>/auto_v1/
├── raw/batches/<batch_id>/
├── reviewed/<batch_id>/
├── annotations/cvat/<batch_id>/
├── datasets/<dataset_version>/
├── gold/
├── models/<model_version>/
├── evaluations/<model_version>/
└── error_pool/
```

## Console 运行配置

必需环境变量：

```text
GCS_BUCKET=yujian-model-factory-571785698442
REGISTRY_DB_URL=...
APP_GIT_COMMIT=<deployed git sha>
```

本地开发可使用 SQLite；Cloud Run 正式环境必须使用持久化 PostgreSQL / Cloud SQL，不能依赖实例本地 SQLite。

## 当前阶段

1. 用 Pilot + 豆包跑通 Console 主流程。
2. 通过页面完成清洗、人工审核与 Species Truth。
3. 冻结 `DS_M1_v0.1`。
4. V0.2 接入 CVAT、Training Run、Evaluation 与 Error Pool。

详见 `docs/ARCHITECTURE.md`、`docs/GCS_BOOTSTRAP.md` 与 `docs/MODEL_FACTORY_CONSOLE_V0_1.md`。
