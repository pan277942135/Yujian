# YuJian Model Factory · Review / Registry V1

## 目标

把 `GCS raw batch -> Review -> Approved/Rejected/Hard Case -> Dataset Freeze` 变成可追踪的长期流程，而不是靠 ZIP/CSV 手工管理。

## 数据职责

- GCS：保存原始图片、manifest、标注、dataset、model 等不可变资产。
- Registry DB：保存 Batch、Image、Review 状态、Species Truth、审计历史、Dataset/Run/Model lineage。
- GitHub：保存代码、配置、Schema、CI 与部署定义。

## Review 状态

- `pending`
- `approved`
- `needs_review`
- `hard_case`
- `rejected`

## Species Truth 状态

- `LIKELY_CORRECT`
- `UNCERTAIN`
- `HARD_PAIR_REVIEW`
- `WRONG_LABEL_SUSPECTED`
- `EXPERT_HOLD`

每一次人工修改都会写入 `review_events`，保留 before/after JSON 与 reviewer，避免覆盖历史。

## 本地 / Cloud Shell 启动

```bash
cd ~/Yujian
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GCS_BUCKET=yujian-model-factory-571785698442
export REGISTRY_DB_URL=sqlite:///./var/yujian_registry.db

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

页面：

- `/`：Overview / Batch / Species 分布
- `/review`：Review Queue
- `/docs`：FastAPI API 文档

## 将 GCS Batch 同步到 Registry

先确保 Batch 已通过 `scripts/ingest_batch.py` 写入：

```text
gs://yujian-model-factory-571785698442/raw/batches/<BATCH_ID>/
```

再执行：

```bash
python scripts/sync_batch_registry.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_PILOT_001

python scripts/sync_batch_registry.py \
  --bucket yujian-model-factory-571785698442 \
  --batch-id BATCH_20260826_WB_001
```

同步规则：

1. 读取 `batch.json` 与 `fish_manifest.csv`。
2. 解析 GCS 中真实图片对象。
3. 新图片写入 Registry。
4. Manifest 中 `approved` 会初始化为 Approved；未知/待确认初始化为 Pending。
5. 再次同步只刷新来源/路径等采集元数据，不覆盖已经人工修改过的 Review / Truth。

## 生产数据库

开发/Cloud Shell 可使用 SQLite；正式 Cloud Run 不应把 SQLite 当持久化数据库。

生产环境通过同一 `REGISTRY_DB_URL` 切换 PostgreSQL/Cloud SQL：

```text
postgresql+psycopg://<user>:<password>@/<database>?host=/cloudsql/<instance-connection-name>
```

应用 ORM 同时兼容 SQLite 与 PostgreSQL。

## 下一阶段

1. Cloud SQL PostgreSQL + Secret Manager。
2. Cloud Run 部署 Review Console。
3. Review 批量操作 / Hard Pair 专用队列。
4. Approved 数据直接生成 Dataset Candidate。
5. Dataset Freeze 与 Registry DB 联动。
