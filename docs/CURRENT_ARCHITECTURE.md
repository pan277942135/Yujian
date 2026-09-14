# YuJian 当前控制台架构地图

版本：

```text
YuJian AI Model Factory Console
Audit baseline: origin/main @ fb6f8c33954ddd4ffdca6f38ec172e77d5855626
Audit date: 2026-09-14
```

本文件是 YuJian AI Platform V1 开发前的只读走查结果。它描述当前代码的真实边界，后续新增 Platform 必须以此为基线；历史文档与当前代码不一致时，以当前代码和 CI 为准。

## 1. 启动与应用结构

### 1.1 运行入口

生产容器由 `Dockerfile` 启动：

```text
uvicorn app.entry:app --host 0.0.0.0 --port ${PORT}
```

`app.entry:app` 不是重新创建的第二个 FastAPI 实例，而是从 `app.main` 导入的同一个 `app`。

### 1.2 FastAPI 结构

|层级|文件|职责|
|-|-|-|
|应用实例|`app/main.py`|创建 `FastAPI(title="YuJian AI Model Factory", version="0.1.0")`；创建根模板引擎；注册旧版页面和核心聚合 API；注册启动时 DB 初始化。|
|最终组装|`app/entry.py`|导入 `app.main.app`；注册所有业务 router；安装统一旧导航；安装反馈自动化 middleware；注册目标鱼种、分割模型和鱼鉴种子。|
|数据库|`app/db.py`|SQLite 本地开发 / Cloud SQL PostgreSQL 连接；`Base`、`SessionLocal`、`get_db`、`init_db`；启动时只做非破坏性 additive columns。|
|模型|`app/models.py`、`app/fish_knowledge/*.py`|SQLAlchemy Registry、数据审核、训练、模型、评估、鱼鉴和用户鱼获模型。|
|模板|各业务模块的 `Jinja2Templates(directory="app/templates")`|旧页面均直接使用共享目录，但没有旧版 `base.html` 继承体系。|
|媒体|`app/main.py`、`app/fish_knowledge/api.py`、`app/crop_qa.py` 等|通过受控 API 从 GCS 或本地路径返回图片；不应向前端暴露内部 GCS / Worker 地址。|

### 1.3 Router 注册顺序

`app/entry.py` 当前按以下顺序将 router 注册到同一个 FastAPI app：

```text
presence
dedupe
bulk_review
inspect
dataset_freeze
training
inference
feedback_ingest
inference_upload
auth
catches
batch_upload
automation
intelligence
crop_qa
crop_dataset
crop_review
dataset_crop_review
accepted_bbox
crop_audit
detector_parity
segmentation
fish_completion
fish_completion_auto
powerpaint_direct
powerpaint_shape_guided
fish_knowledge
fish_knowledge_admin
fish_knowledge_admin_compat
fish_asset_import
fish_asset_import_page
```

`app.main` 中的页面和核心 API 在 `include_router` 之前已经注册。新增 Platform 应作为 additive router 接入 `app.entry`，不能替换或重新注册旧 router。升级后，旧导航的模板源文件是 `app/templates/legacy/sidebar.html`，而 Platform 使用 `app/templates/platform/sidebar.html`。

### 1.4 Middleware 与启动事件

|类型|位置|当前行为|Platform 约束|
|-|-|-|-|
|控制台访问守卫|`app/secure.py:install_access_guard`，由 `app.main` 在创建 app 后安装|当配置 `CONSOLE_ACCESS_KEY` 时，控制台页面和 `/api/*` 默认需要 `yujian_console` cookie；`/health*`、登录、App Bearer API、公开鱼鉴 GET 例外。未登录页面重定向 `/login`，API 返回 401。|Platform 继续继承同一控制台鉴权，不得新增绕过口。|
|反馈自动化|`app/p0_automation.py:install_feedback_automation`，由 `app.entry` 安装|仅处理 `POST /api/feedback`：安全别名归一化、成功后尝试按阈值物化反馈批次；失败不会让原始反馈 POST 失败。|Platform 只读聚合已有状态，不重写反馈状态机。|
|旧模板导航|`app/unified_nav.py:install_unified_nav`，由 `app.entry` 对旧模板引擎逐个安装|使用 `UnifiedNavLoader` 替换页面首个 `<header>`，移除旧主导航链接后插入统一顶部导航；保留页面专属控制。|Platform 使用独立模板引擎，不安装 `UnifiedNavLoader`。|
|主启动事件|`app/main.py:startup`|`init_db()`，确保 Species Catalog。|新增表如果确有必要，只能 additive 且受本次任务允许的 3 张表约束。|
|扩展启动事件|`app/entry.py:seed_target_species_catalog`|初始化分割模型；目标鱼种、鱼鉴初始数据和内容种子。|不得改变旧启动顺序的语义。|

当前没有 `StaticFiles` mount、全局异常 handler 或独立前端打包产物；页面是服务端模板加页面内 JavaScript。

## 1.5 V1 增量边界

Platform V1 通过 `app/platform/routes/` 和 `app/platform/services/adapters.py` 增加聚合页面与 `/api/platform/*` 适配器。现有 Batch、Dataset、Review、Training、Model 和 Completion 仍是业务真值；只增加 `pipeline_run`、`fish_asset`、`platform_operation_log` 三张索引 / 审计表。新平台不把内部 GCS URI、Worker 地址或服务账号信息放进响应。

## 2. 当前页面 Route 清单

以下是从当前 `app.openapi()` 读取的 UI 入口及其实现文件。OpenAPI 在基线提交上共有 **187 个 paths**（包含页面、API 和媒体端点）。

|路径|对应文件|功能|
|-|-|-|
|`/`|`app/main.py` → `overview_page` → `overview.html`|旧版总览，展示批次、图片、审核和飞轮摘要。|
|`/batches`|`app/main.py` → `batches_page` → `batches.html`|Incoming 发现、Audit、Promote、Registry Sync、去重和鱼体检测。|
|`/batches/upload`|`app/batch_upload_api.py` → `batch_upload_page`|批量上传页面。|
|`/review/bulk`|`app/bulk_review.py` → `bulk_review_page`|批量快速审核。|
|`/review`|`app/main.py` → `review_page` → `review.html`|单张审核入口。|
|`/inspect`|`app/inspect.py` → `inspect_page` → `inspect.html`|带状态、鱼种、Presence 和搜索筛选的数据检查。|
|`/datasets`|`app/main.py` → `datasets_page` → `datasets.html`|Dataset Freeze 预览、冻结、Finalize、Audit。|
|`/datasets/accepted-bbox`|`app/accepted_bbox_review.py` → `accepted_bbox_page` → `accepted_bbox_review.html`|Frozen Dataset accepted bbox 补确认。|
|`/crop-datasets`|`app/main.py` → 307 重定向|兼容入口，重定向到 `/datasets/accepted-bbox`。|
|`/crop-review`|`app/crop_review.py` → `crop_review_page` → `crop_review.html`|普通 Batch 的 crop bbox 人工审核。|
|`/crop-qa`|`app/crop_qa.py` → `crop_qa_page` → `crop_qa.html`|Crop 质量检查。|
|`/training`|`app/training_api.py` → `training_page` → `training.html`|Classifier Training 任务、参数和产物。|
|`/inference`|`app/inference_api.py` → `inference_page` → `inference.html`|模型实测。|
|`/intelligence`|`app/intelligence_api.py` → `intelligence_page` → `intelligence.html`|混淆、数据缺口、错误驱动任务分析。|
|`/species`|`app/main.py` → `species_page` → `species.html`|旧版 Species Catalog 管理。|
|`/fish-knowledge`|`app/main.py` → `fish_knowledge_page` → `fish_knowledge.html`|Fish Knowledge CMS 内容工作台。|
|`/fish-knowledge/assets/import`|`app/fish_knowledge/import_batch.py` → `page_router`|鱼鉴资产批量导入。|
|`/feedback`|`app/main.py` → `feedback_page` → `feedback.html`|线上反馈 Review / Materialize。|
|`/debug/detector-parity`|`app/detector_parity_api.py` → `detector_parity_page`|Detector Golden / Runtime parity 调试。|
|`/debug/fish-segmentation`|`app/segmentation_api.py` → `fish_segmentation_page`|SAM 鱼体分割调试。|
|`/debug/fish-completion-lab`|`app/fish_completion_lab.py` → `fish_completion_lab_page`|Fish Completion Lab V0.1。|
|`/debug/fish-completion-lab-v02`|`app/fish_completion_lab.py` 和 `app/fish_completion_auto.py`|P2.5 / 自动鱼体资产和 Worker 调试；以当前实际注册顺序及 OpenAPI 为准。|
|`/debug/powerpaint-direct-lab`|`app/powerpaint_direct_lab.py` → `direct_lab_page`|PowerPaint Direct 实验。|
|`/debug/powerpaint-shape-guided-lab`|`app/powerpaint_shape_guided_lab.py` → `page`|PowerPaint Shape Guided 实验。|
|`/login`、`/logout`|`app/secure.py`|控制台口令登录和退出。|
|`/health`、`/health/deploy`、`/health/detector`|`app/main.py`、`app/entry.py`|运行、部署、Detector 健康检查。|

## 3. 当前 API Route 清单

为避免把 187 个路径压成无法使用的长表，下面按实际 router / prefix 展开。花括号是当前动态路径参数，不代表新增 API。

### 3.1 `app.main` 核心 API

|路径|方法|功能|
|-|-|-|
|`/api/overview`|GET|旧版总览聚合。|
|`/api/flywheel/summary`|GET|Master Pool / Feedback / Dataset 飞轮摘要。|
|`/api/incoming`|GET|列出 GCS Incoming 批次。|
|`/api/batches`|GET|读取 Registry Batch 及审核计数。|
|`/api/batches/audit`|POST|Manifest / Species Catalog 审核。|
|`/api/batches/promote`|POST|Incoming 批次入库。|
|`/api/batches/sync`|POST|批次登记到 Registry。|
|`/api/review`|GET|按状态、批次、鱼种、搜索词分页读取 ImageAsset。|
|`/api/review/stats`|GET|审核状态计数。|
|`/api/review/{batch_id}/{image_id}`|PATCH|旧审核更新；通过前要求人工确认 truth_species 和 accepted_bbox。|
|`/api/review/{batch_id}/{image_id}/reidentify-bbox`|POST|重新运行 Detector，仅更新 candidate bbox。|
|`/api/species`|GET、POST|读取 / 创建 Species Catalog。|
|`/api/species/{species_key}/status`|PATCH|更新 Species Catalog 状态。|
|`/api/feedback`|GET、POST|Feedback 读取与记录。|
|`/api/feedback/materialize`|POST|把反馈物化为 Incoming Batch。|
|`/api/datasets`|GET|读取 DatasetVersion 列表。|
|`/api/datasets/summary`|GET|读取 Dataset / Flywheel 摘要。|
|`/api/datasets/{dataset_version}/crop-readiness`|GET|读取 Crop readiness。|
|`/api/datasets/freeze`|POST|正式冻结累计 Dataset。|
|`/media/{batch_id}/{image_id}`|GET|受控制台鉴权保护的原图媒体网关。|

### 3.2 数据上传、Presence、去重和审核

|文件|Router prefix|实际路径族|职责|
|-|-|-|-|
|`app/batch_upload_api.py`|无 prefix|`POST /api/batches/upload-start`、`/upload-file`、`/upload-finalize`、`/upload`|上传分片 / 文件并完成 Batch 入库准备。|
|`app/presence.py`|`/api/presence`|`GET /batches`、`/batch/{batch_id}`、`/images`、`/image/{batch_id}/{image_id}`；`POST /scan`、`/reject-no-fish`|鱼体 Presence 扫描、计数和过滤。|
|`app/dedupe.py`|`/api/dedupe`|`GET /batches`、`/batch/{batch_id}`、`/groups/{batch_id}`；`POST /scan`、`/reject-duplicates`|图片 fingerprint、近重复分组和过滤。|
|`app/bulk_review.py`|无 prefix|`GET /api/bulk-review/species`、`/images`；`POST /api/bulk-review/apply`|批量审核和批量状态 / truth / bbox 更新。|
|`app/inspect.py`|无 prefix|`GET /api/inspect/images`；`PATCH /api/inspect/presence/{batch_id}/{image_id}`|数据检查列表和 Presence 人工覆盖。|
|`app/crop_review.py`|无 prefix|`GET /api/crop-review/{batch_id}/summary`、`/items`；`PATCH /api/crop-review/{batch_id}/{image_id}`|普通 Batch 的 accepted bbox 人工门。|
|`app/accepted_bbox_review.py`|无 prefix|`GET /api/dataset-accepted-bbox/summary`、`/items`；`PATCH /api/dataset-accepted-bbox/{batch_id}/{image_id}`；`POST .../reidentify`、`/bulk`|Frozen Dataset / accepted bbox 补确认兼容接口。|
|`app/crop_qa.py`|无 prefix|`GET /api/crop-qa`、`GET /media/inference/{image_id}/{kind}`|Crop QA 与 Inference 媒体。|

### 3.3 Dataset Freeze 和 Crop Dataset

|文件|Router prefix|实际路径族|职责|
|-|-|-|-|
|`app/dataset_api.py`|`/api/dataset-freeze`|`POST /preview`；`GET /{dataset_version}`、`/{dataset_version}/items`、`/{dataset_version}/audit`；`POST /{dataset_version}/finalize`|Whole-image Dataset Freeze、Lineage、Audit。|
|`app/crop_dataset_api.py`|`/api/crop-datasets`|`GET /sources`、`/{dataset_version}/validation`、`/{dataset_version}/status`；`POST /build`、`/{dataset_version}/freeze`|CROP_CLASSIFIER_V1 数据构建和冻结。|
|`app/dataset_crop_review.py`|`/api/dataset-crop-review`|`GET /{dataset_version}/summary`、`/items`、`/detector-audit`、`/{dataset_version}/{image_id}/image`、`/crop`；`PATCH /{dataset_version}/{image_id}`|Frozen Dataset bbox review、候选 Detector 信息和 crop preview。|
|`app/crop_audit_api.py`|`/api/crop-audit`|`GET /historical`|历史 crop 审计。|

### 3.4 Training、Inference、Intelligence、Automation

|文件|Router prefix|实际路径族|职责|
|-|-|-|-|
|`app/training_api.py`|无 prefix|`GET /training`；`GET /api/training/runs`、`/api/training/runs/{run_id}`、`/metrics`；`POST /api/training/runs`；`GET /api/models`|Cloud Run Training Job、TrainingRun、ModelVersion 列表和 metrics。|
|`app/inference_api.py`|无 prefix|`GET /inference`、`/api/inference/models`；`POST /api/inference/predict`、`/api/inference/batch`|模型实测。|
|`app/inference_upload_api.py`|无 prefix|`POST /api/v1/inference/upload`、`POST /api/v1/inference/{image_id}/review`|Android inference asset 接收和 App Review。|
|`app/intelligence_api.py`|无 prefix|`GET /intelligence`、`/api/intelligence`、`/confusion`、`/gaps`、`/tasks`；`POST /api/intelligence/analyze`、`/tasks`、`/tasks/{task_id}/batch`|错误分析、数据缺口和采集任务建议。|
|`app/p0_automation.py`|`/api/automation`|`GET /dataset-readiness`、`/feedback-status`|训练准备度和 Feedback 自动化状态。|

### 3.5 SAM、Completion、PowerPaint、Detector Debug

|文件|实际路径族|职责|
|-|-|-|
|`app/segmentation_api.py`|`GET /debug/fish-segmentation`；`POST /api/debug/fish-segmentation`、`/api/debug/fish-hero-review`|SAM 分割和 Hero Review。|
|`app/fish_completion_lab.py`|`GET /debug/fish-completion-lab`、`/debug/fish-completion-lab-v02`；`GET /api/debug/fish-completion-lab*/datasets`、`/images`、`/worker-status`、`/report/{test_id}`；`POST /api/debug/fish-completion-lab/masks`、`/prepare`、`/review`、`/run`|Completion Lab、Worker 调用和报告。|
|`app/fish_completion_auto.py`|`GET /debug/fish-completion-lab-v02`；`POST /api/debug/fish-completion-lab-v02/auto-run`|自动 Completion 决策和运行。|
|`app/powerpaint_direct_lab.py`|`GET /debug/powerpaint-direct-lab`；`GET /api/debug/powerpaint-direct-lab/datasets`、`/images`；`POST /.../prepare`、`/run`|PowerPaint Direct 实验。|
|`app/powerpaint_shape_guided_lab.py`|`GET /debug/powerpaint-shape-guided-lab`；`GET /api/debug/powerpaint-shape-guided-lab/datasets`、`/images`；`POST /.../prepare`、`/run`|PowerPaint Shape Guided / P2.5 实验。|
|`app/detector_parity_api.py`|`GET /debug/detector-parity`；`POST /api/debug/detector-parity`|Detector parity。|

### 3.6 Fish Knowledge、用户 App 和鉴权

|文件|Router prefix|实际路径族|职责|
|-|-|-|-|
|`app/fish_knowledge/api.py`|`/api/v1/fish`|`GET /species`、`/species/{species_id}`、`/detail`、`/gallery/{image_id}/media`、`/knowledge-media/...`|App 公开鱼鉴读取和媒体。|
|`app/fish_knowledge/admin.py`|`/api/v1/admin/fish`；兼容 `/api/admin/fish`|Species、Cover、Cards、Profile、Fishing、Gallery、Video、Similarity 的 CMS CRUD、发布和上传。|
|`app/fish_knowledge/import_batch.py`|`/api/v1/admin/fish/assets/import-batches`|鱼鉴资产批次创建、上传、扫描、执行、同步、版本激活和页面。|
|`app/auth_api.py`|`/api/v1/auth`|`POST /register`、`/login`|App 用户注册登录。|
|`app/catches_api.py`|`/api/v1/catches`|`GET/POST /`、`/statistics`、`/upload-image`、媒体端点|用户鱼获归档和统计。|
|`app/feedback_ingest_api.py`|无 prefix|`POST /api/feedback/ingest`|带 ingest key 的 App 反馈入口。|

## 4. Template 清单

所有旧业务模板均位于 `app/templates/`，当前没有旧版 `base.html` / `sidebar.html` 继承关系；除 `_mobile_ux.html` 外，大多数页面自带完整 `<!doctype html>`、`<header>` 和页面脚本。

|模板|用途|继承 / include|
|-|-|-|
|`overview.html`|旧版总览|include `_mobile_ux.html`|
|`batches.html`|批次发现与处理|include `_mobile_ux.html`|
|`batch_upload.html`|批量上传|include `_mobile_ux.html`|
|`bulk_review.html`|快速审核|include `_mobile_ux.html`|
|`review.html`|单张审核|include `_mobile_ux.html`|
|`inspect.html`|数据检查|include `_mobile_ux.html`|
|`datasets.html`|Whole-image Dataset Freeze|include `_mobile_ux.html`|
|`crop_datasets.html`|Crop Dataset 工作流|include `_mobile_ux.html`|
|`accepted_bbox_review.html`|Frozen accepted bbox 补确认|include `_mobile_ux.html`|
|`crop_review.html`|Batch crop review|include `_mobile_ux.html`|
|`crop_qa.html`|Crop QA|include `_mobile_ux.html`|
|`training.html`|训练任务|include `_mobile_ux.html`|
|`inference.html`|模型实测|include `_mobile_ux.html`|
|`intelligence.html`|模型智能分析|include `_mobile_ux.html`|
|`species.html`|Species Catalog|include `_mobile_ux.html`|
|`fish_knowledge.html`|鱼鉴 CMS|include `_mobile_ux.html`|
|`fish_asset_import.html`|鱼鉴资产导入|无 include；保留页面自身 header|
|`feedback.html`|用户反馈|include `_mobile_ux.html`|
|`detector_parity.html`|Detector parity|include `_mobile_ux.html`|
|`fish_completion_lab.html`|Fish Completion Lab|include `_mobile_ux.html`|
|`fish_completion_auto.html`|自动鱼体资产|include `_mobile_ux.html`|
|`powerpaint_direct_lab.html`|PowerPaint Direct|include `_mobile_ux.html`|
|`powerpaint_shape_guided_lab.html`|PowerPaint Shape Guided|include `_mobile_ux.html`|
|`_mobile_ux.html`|旧模板响应式样式和移动端导航增强|被业务模板 include|

Platform 新模板必须放在独立目录（推荐 `app/templates/platform/`），并由独立 `Jinja2Templates(directory="app/templates/platform")` 加载；不要让 `UnifiedNavLoader` 改写它们。

## 5. 当前 Sidebar / Navigation 位置

当前旧导航的真实实现不在 `app/templates/`：

```text
app/unified_nav.py
├── _NAV_ITEMS                 旧版主导航配置
├── _canonical_nav()           生成 <header class="app-nav"> HTML/Jinja
├── UnifiedNavLoader            读取模板源并替换第一个 header
└── install_unified_nav()      对各业务模板引擎安装一次
```

当前旧导航项包含：

```text
总览、数据批次、数据导入、快速审核、单张审核、鱼种管理、鱼鉴内容、
用户反馈、数据集、模型训练、模型实测、模型智能分析、Crop QA、
Detector Parity、Fish Completion Lab、Fish Asset Pipeline、
PowerPaint Direct、PowerPaint Shape Guided。
```

`app/templates/_mobile_ux.html` 通过 CSS / JS 将旧顶部导航在窄屏变成可展开导航，但它不是业务 Sidebar。

本次隔离要求：

- 旧工作台继续由 `app/unified_nav.py` 提供导航，旧页面 route 和模板保持可用。
- 新 Platform 使用 `app/platform/` 的 route/service/component 结构和 `app/templates/platform/sidebar.html`。
- 如新增 `app/templates/legacy/sidebar.html`，它只能承载旧版导航的可维护副本或兼容包装，不能改变旧页面的渲染结果。
- 不要在 `unified_nav.py` 中通过越来越长的条件把 Platform 菜单混入旧导航。

## 6. 当前数据模型与事实边界

### 6.1 已有 Registry 表（禁止重建）

|模型|物理表|用途|
|-|-|-|
|`Batch`|`batches`|批次登记、Incoming Promote 后的 Registry 入口。|
|`ImageAsset`|`image_assets`|单张图片、采集标签、truth、审核状态、场景质量。|
|`ReviewEvent`|`review_events`|审核前后变更审计。|
|`SpeciesCatalog`|`species_catalog`|稳定 species_key、中文名、Active / Candidate 状态。|
|`BatchCropReview`|`batch_crop_reviews`|普通 Batch 的 bbox 人工门。|
|`DatasetCropReview`|`dataset_crop_reviews`|Frozen Dataset 的 bbox 人工门。|
|`DatasetCropReviewEvent`|`dataset_crop_review_events`|Frozen bbox 审核历史。|
|`ImageFingerprint`|`image_fingerprints`|sha / pHash / dHash / 近重复分组。|
|`FishPresenceResult`|`fish_presence_results`|无鱼 / 单鱼 / 多鱼 / 不确定检测。|
|`DatasetVersion`|`datasets`|不可变 Dataset 快照；含 `pipeline_type`、`metadata_json`。|
|`DatasetItem`|`dataset_items`|冻结 Dataset 的 lineage item。|
|`TrainingRun`|`training_runs`|训练任务与 Cloud Run Job 记录。|
|`ModelVersion`|`models`|模型版本、产物、状态和所属 Dataset。|
|`Evaluation`|`evaluations`|评测产物 URI、混淆矩阵、错误池关系。|
|`ErrorCase`|`error_pool`|错误案例和 hard pair 信息。|
|`FeedbackEvent`|`feedback_events`|线上反馈，未人工 Review 前不能成为 Ground Truth。|
|`InferenceAsset`|`inference_assets`|App 推理资产和 accepted review 信息。|

### 6.2 鱼鉴与用户表

`app/fish_knowledge/` 现有 `FishSpecies`、`FishGalleryImage`、`FishProfile`、`FishFishing`、`FishVideo`、`FishSimilarity`、`FishRanking`、`FishSpeciesCover`、`FishCard` 以及资产导入批次 / 版本模型。用户侧已有 `AppUser`、`FishCatch`。

### 6.3 不可破坏的数据约束

```text
candidate_bbox      = Detector / Presence 候选，只是诊断证据
accepted_bbox       = 明确人工确认，才可以进入训练构建
采集标签             != 自动 Ground Truth
Approved Master Pool != 某一个 Dataset 私有数据
Dataset              = 不可变快照
Species ID           != 模型 class index
```

当前任务允许的新增 Registry 表最多为：

```text
pipeline_run
fish_asset
platform_operation_log
```

任何新 Platform API 必须优先读取已有表 / 调用已有 service；缺少能力时先加 adapter，再评估是否确有必要写入上述表。

## 7. 当前代码走查结论

1. 这是单体 FastAPI 服务，不是前后端分离 SPA；Platform 应继续使用服务端模板和页面内渐进增强，减少部署风险。
2. 旧导航是运行时模板 loader 注入，不能用“改一个公共 base 模板”的方式隔离；新 Platform 必须拥有自己的模板引擎和 Sidebar。
3. 旧审核逻辑已经包含 truth、accepted bbox、ReviewEvent 和反馈回写约束；新审核中心只能做适配和统一展示，不能复制出第二个审核状态机。
4. 旧 Training、Dataset Freeze、Evaluation 和 Fish Completion API 已存在；Platform 页面应通过 adapter 聚合和格式转换，不能创建 `Dataset2`、`TrainingJob2` 或 `Model2`。
5. 当前基线 `origin/main` 为 `fb6f8c3`。后续开发应继续从当前 `main` 前进，不使用强制回退或 `git reset --hard`。
