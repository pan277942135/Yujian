# YuJian AI Model Factory

渔见 AI 的长期数据与模型持续迭代工程。

## 核心目标

不是维护“一次性训练集”，而是维护一个持续增长的数据飞轮：

```text
采集数据 / 真实用户鱼获 / 线上识别反馈
                  ↓
               Incoming
                  ↓
          自动清洗 / 去重 / QA
                  ↓
          人工审核 / Species Truth
                  ↓
        Approved Master Pool（永久累积）
                  ↓
      Dataset Snapshot v0.1 / v0.2 / ...
                  ↓
           Training / Evaluation
                  ↓
              Model Version
                  ↓
              用户继续使用
                  ↓
     预测正确 / 用户纠错 / 新鱼种候选
                  └──────────────→ 回到 Incoming / Review
```

目标是：**越用数据越全面、越用识别越准、越用支持的鱼种越多，同时每一版 Dataset / Model 都可以完整追溯。**

## 系统边界

- **GitHub**：代码、配置、Schema、Dataset/Model 版本索引、CI/CD。
- **Google Cloud Storage (GCS)**：原始图片、清洗结果、冻结 Dataset、模型、评估产物。
- **Registry DB**：Batch、图片审核状态、Species Catalog、反馈事件、Dataset/Model 版本关系。
- **Model Factory Console**：把日常数据操作从 Cloud Shell 命令转换成网页按钮。

## Model Factory Console V0.1

V0.1 覆盖主数据闭环：

- **Overview**：查看 Approved Master Pool、Active/Candidate Species、Feedback、Dataset 与审核进度。
- **Batches**：自动发现 `incoming/`，一键执行 `Audit -> Promote -> Registry Sync`。
- **Review**：看图审核；Approve / Reject / Needs Review / Hard Case；修正 Species Truth；保留 ReviewEvent 审计记录。
- **Species**：鱼种目录可扩展；新鱼种先 Candidate，确认后 Active。
- **Feedback**：接收线上模型识别正确/错误/未知反馈，并可一键重新形成 Incoming Batch。
- **Dataset**：默认冻结截至当前 **全部 Approved + Active Species** 的累计不可变快照。

页面入口：

```text
/
/batches
/review
/species
/feedback
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

## 三条长期不可破坏的规则

### 1. Approved Master Pool 是主资产

审核通过的数据不会属于某一个 Dataset；它进入长期 Master Pool。后续新 Approved 数据继续累积。

### 2. Dataset 是不可变快照

`DS_M1_v0.1`、`DS_M1_v0.2` 等都是 Master Pool 在不同时间点的快照，旧版本永不覆盖。

每个 Dataset 同时冻结（不复制训练图片，manifest 指向已有 immutable GCS object）：

```text
dataset_manifest.csv
class_map.json
dataset.json
```

### 3. Species ID 与模型 Class Index 解耦

Species Catalog 使用稳定 `species_key`。模型的数字 class index 只存在于具体 Dataset 的 `class_map.json` 中。

因此从 9 类扩展到 20 类、50 类时，不会破坏历史 Dataset / Model。

## 核心 ID

- `batch_id`: `BATCH_20260826_DB_001`
- `species_key`: `grass_carp`
- `dataset_version`: `DS_M1_v0.1`
- `run_id`: `RUN_M1_0001`
- `model_version`: `MODEL_M1_0001`
- `source_event_id`: 线上用户反馈事件的幂等 ID

## GCS 推荐结构

```text
gs://<bucket>/
├── incoming/<upload-folder>/
├── cleaning/<batch_id>/auto_v1/
├── raw/batches/<batch_id>/
├── feedback/
├── reviewed/batches/<batch_id>/
├── annotations/cvat/<batch_id>/
├── datasets/<dataset_version>/
│   ├── dataset_manifest.csv
│   ├── class_map.json
│   └── dataset.json
├── gold/
├── models/<model_version>/
├── evaluations/<model_version>/
└── error_pool/
```

## 线上 Feedback 接口

未来 App 可将真实用户鱼获与识别反馈写入：

```text
POST /api/feedback
X-YuJian-Ingest-Key: <FEEDBACK_INGEST_KEY>
```

支持：

```text
confirmed
corrected
unknown
new_species_candidate
```

未知的新鱼种纠错会自动生成 Species Candidate，但不会未经审核直接进入训练。

## Console 运行配置

```text
GCS_BUCKET=yujian-model-factory-571785698442
REGISTRY_DB_URL=<persistent PostgreSQL / Cloud SQL URL>
APP_GIT_COMMIT=<deployed git sha>
CONSOLE_ACCESS_KEY=<operator password>
FEEDBACK_INGEST_KEY=<mobile feedback API key>
```

本地开发可以使用 SQLite；Cloud Run 正式环境必须使用持久化 PostgreSQL / Cloud SQL。

## 当前阶段

1. 用 Pilot + 豆包跑通 `Incoming -> Review -> Approved Master Pool`。
2. 通过 Console 完成人工审核与 Species Truth。
3. 冻结累计 `DS_M1_v0.1`。
4. V0.2 接入 CVAT、Training Run、Evaluation、Error Pool 与自动再训练触发。

详见 `docs/ARCHITECTURE.md`、`docs/GCS_BOOTSTRAP.md` 与 `docs/MODEL_FACTORY_CONSOLE_V0_1.md`。
