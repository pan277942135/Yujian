# YuJian AI Model Factory

渔见 AI 的长期数据与模型持续迭代工程。

## 目标

统一管理：

`数据采集 Batch -> 清洗/审核 -> 标注 -> Dataset Version -> Training Run -> Model Version -> Evaluation -> Error Pool -> 下一轮数据采集`

## 系统边界

- **GitHub**：代码、配置、Schema、Dataset/Model 版本索引、CI/CD。
- **Google Cloud Storage (GCS)**：原始图片、审核图片、CVAT 导出、冻结数据集、模型权重、评估产物。
- **Model Factory Console**：Overview / Batches / Review / Annotation / Datasets / Runs / Evaluation / Error Pool。

## 核心 ID

- `batch_id`: 每一批新采集数据，例如 `BATCH_20260826_WB_001`
- `dataset_version`: 每次冻结后的训练数据版本，例如 `DS_M1_v0.1`
- `run_id`: 每次训练，例如 `RUN_M1_0001`
- `model_version`: 每个模型产物，例如 `MODEL_M1_0001`

## GCS 推荐结构

```text
gs://<bucket>/
├── raw/batches/<batch_id>/
├── reviewed/<batch_id>/
├── annotations/cvat/<batch_id>/
├── datasets/<dataset_version>/
├── gold/
├── models/<model_version>/
├── evaluations/<model_version>/
└── error_pool/
```

## 当前阶段

1. 将已有 Pilot/WorkBuddy 数据作为独立 Batch 导入。
2. 建立 Species Truth + 去重 + 审核状态。
3. CVAT 产生 bbox 后冻结第一版 Dataset。
4. 启动分类 Baseline 与 YOLO Detection Baseline。
5. 每次新数据只通过 Batch -> Dataset Version 合并，不直接覆盖旧训练集。

详见 `docs/ARCHITECTURE.md` 与 `docs/GCS_BOOTSTRAP.md`。
