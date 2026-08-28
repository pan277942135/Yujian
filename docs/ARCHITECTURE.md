# YuJian AI Model Factory V1 架构

## 1. 生命周期

```text
Collect/Upload
  -> Batch Registry
  -> Raw GCS
  -> Dedup & Quality Gate
  -> Species Truth Review
  -> Annotation (CVAT)
  -> Dataset Freeze
  -> Training Run
  -> Evaluation
  -> Model Registry
  -> Error Pool
  -> Next Batch
```

## 2. 不可变原则

1. `raw/batches/<batch_id>` 只追加，不覆盖。
2. Dataset 一旦冻结，不修改；新数据必须生成新 `dataset_version`。
3. Training Run 必须绑定唯一 Dataset Version、代码 commit、训练参数和随机种子。
4. Gold/Test 不与 Train/Val 共享 `group_id/capture_event_id`。
5. 模型产物不可覆盖；每个模型使用独立 `model_version`。

## 3. GCS 对象布局

```text
gs://<bucket>/
├── raw/batches/<batch_id>/
│   ├── images/
│   ├── metadata/fish_manifest.csv
│   └── batch.json
├── reviewed/<batch_id>/
│   ├── approved/
│   ├── needs_review/
│   ├── rejected/
│   └── review_manifest.csv
├── annotations/cvat/<batch_id>/
├── datasets/<dataset_version>/
│   ├── manifest.csv
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset.json
├── gold/<gold_version>/
├── models/<model_version>/
│   ├── weights/
│   ├── run.json
│   └── metrics.json
├── evaluations/<model_version>/
│   ├── metrics.json
│   ├── confusion_matrix.csv
│   └── errors.csv
└── error_pool/<error_pool_version>/
```

## 4. 核心实体

### Batch
每次上传或采集形成一个 Batch。

必填：`batch_id`, `source`, `created_at`, `image_count`, `manifest_uri`, `status`。

状态：`INGESTED -> CLEANING -> REVIEW -> ANNOTATION -> READY_FOR_DATASET -> ARCHIVED`。

### Dataset Version
由一个或多个通过审核/标注的 Batch 冻结产生。

必填：`dataset_version`, `parent_version`, `batch_ids`, `train_count`, `val_count`, `test_count`, `manifest_uri`, `git_commit`。

### Training Run
每次训练都是独立 Run。

必填：`run_id`, `dataset_version`, `git_commit`, `model_family`, `params`, `seed`, `started_at`, `status`。

### Model Version
训练完成后的模型产物。

必填：`model_version`, `run_id`, `artifact_uri`, `metrics_uri`, `status`。

状态：`EXPERIMENT`, `CANDIDATE`, `PRODUCTION`, `REJECTED`, `ARCHIVED`。

## 5. 第一阶段页面

1. Overview
2. Data Batches
3. Review
4. Annotation
5. Dataset Versions
6. Training Runs
7. Evaluation
8. Error Pool

## 6. 第一阶段技术栈

- Object Storage: Google Cloud Storage
- Code/Version: GitHub
- Registry: SQLite（本地开发）→ 后续 Cloud SQL/PostgreSQL
- Annotation: CVAT
- Training: Ultralytics YOLO + PyTorch classifier
- Compute: 本地/Cloud GPU，Run metadata 统一写 Registry

## 7. 合并新数据的标准动作

```text
新 ZIP/采集结果
 -> 生成 BATCH_xxx
 -> 上传 raw
 -> 去重/质量筛选
 -> Species Truth
 -> CVAT 标注
 -> READY_FOR_DATASET
 -> 选择 Parent Dataset
 -> Freeze 新 Dataset Version
 -> Train 新 Run
 -> 与上一 Production/Candidate 对比
 -> 达标则升级模型
 -> 错误样本进入 Error Pool
```
