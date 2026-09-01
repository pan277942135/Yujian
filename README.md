# YuJian AI Model Factory

> ## 新任务从这里开始
>
> YuJian 项目的项目级 Source of Truth：[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)
>
> 后续新开开发任务时，必须先读取该指导文档，再读取本 README、相关 `docs/`、最新 `main` HEAD 与 CI 状态。若历史对话与 GitHub 最新状态冲突，以 GitHub 为准。
>
> 项目明确分为两大主线：
>
> - **后台侧**：本仓库 `pan277942135/Yujian`，内部 AI Model Factory，负责数据、审核、Dataset、训练、评测、Detector / Classifier、模型发布与 Feedback Review。
> - **前端侧**：`pan277942135/Yujian_App`，面向真实钓友的 Android App，目前仍处于第一版 0→1 产品阶段。

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

当前阶段定位：**仅供内部使用的 AI 生产后台**。规模化以后再考虑多人权限、外部平台化等扩展。

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

每个 Dataset 同时冻结：

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

App 可将真实用户鱼获与识别反馈写入后台；具体当前接口与鉴权实现以最新代码和 `docs/PROJECT_GUIDE.md` 为准。

原则不变：

```text
confirmed
corrected
unknown / new_species_candidate
```

用户纠错不会未经人工 Review 直接成为 Ground Truth。

## Console 运行配置

```text
GCS_BUCKET=yujian-model-factory-571785698442
REGISTRY_DB_URL=<persistent PostgreSQL / Cloud SQL URL>
APP_GIT_COMMIT=<deployed git sha>
CONSOLE_ACCESS_KEY=<operator password>
FEEDBACK_INGEST_KEY=<mobile feedback API key>
```

本地开发可以使用 SQLite；Cloud Run 正式环境必须使用持久化 PostgreSQL / Cloud SQL。

## 当前项目方向

MVP 模型目标不是盲目扩大类别，而是：

> **先把中国钓友最常见约 20–50 类做深，达到真实场景可商用识别水平，并逐步接入 YOLO Fish Detector + Species Classifier。**

当前具体 P0、Runtime Parity、移动模型发布策略和后续 YOLO 架构详见：

- [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) — 项目级指导 / 新任务必读
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/FISH_KNOWLEDGE_V1.md`](docs/FISH_KNOWLEDGE_V1.md) — App 图鉴知识库与 Admin API
- [`docs/GCS_BOOTSTRAP.md`](docs/GCS_BOOTSTRAP.md)
- [`docs/MODEL_FACTORY_CONSOLE_V0_1.md`](docs/MODEL_FACTORY_CONSOLE_V0_1.md)
