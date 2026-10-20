# DET_DS_v0.2 构建与训练门禁状态 V1

**任务：** `DET_DS_v0.2` 实际构建、标签核验与训练门禁解除
**核对日期：** 2026-10-20
**结果：** `IMPLEMENTATION_READY / CLOUD_BUILD_BLOCKED_NO_ADC`
**结论边界：** 仓库构建器和本地合成门禁已验证；本次没有读取云端数据、生成远端工件或核验线上 Job。

## 1. PR 与执行基线

| 项目 | 实际核对结果 |
|---|---|
| 仓库 | `pan277942135/Yujian` |
| PR | [#166](https://github.com/pan277942135/Yujian/pull/166) |
| PR 状态（修改前实时读取） | Open、Draft、Unmerged |
| PR 分支 | `feat/det-fish-training-and-bbox-console` |
| 修改前 PR head / 本地基线 | `257abd2280574098eb3a81a24624f5a62d2fc913` |
| 基线 CI | GitHub Actions run `39408136275` / run #1225，`success`；对应上述基线 SHA |
| 本地实现分支 | `work/det-ds-v02-builder`，从上述 SHA 开始 |

CI #1225 是本次改动前的基线结果，不代表之后提交的 CI 或云端数据验证通过。

## 2. 数据来源事实与验证状态

以下数量来自任务说明和先前项目记录；本次执行容器没有 Accepted Pool CSV、metadata、数据库快照、原图或云凭据，因此均不能标为独立核实：

| 数据事实 | 数量 | 本次状态 |
|---|---:|---|
| Accepted Pool 行数 / `accepted_bbox` 行数 | 6,308 | `USER_REPORTED / NOT_REVERIFIED` |
| 原图 URI / `source_image_exists=true` / `ACCEPTED` / `bbox_source=accepted_bbox` | 各 6,308 | `USER_REPORTED / NOT_REVERIFIED` |
| 旧划分 train / val / test | 4,406 / 956 / 946 | `USER_REPORTED / NOT_REVERIFIED`；v0.2 不沿用 |
| DET_DS_v0.1 | 594 张，Google Vision 来源 | `USER_REPORTED`；构建器明确不读取作 v0.2 标签 |
| 人工确认无鱼负样本 | 未知 | `BLOCKED`；必须从数据库的人为 override 和 ReviewEvent 实查 |

本次没有取得 Accepted Pool Manifest SHA、metadata SHA、原图 SHA 清单、唯一原图数量、真实 EXIF/尺寸、重复关系、标签冲突数、异常框数、train/val/test 实际划分或 COCO SHA。没有生成 `DET_DS_v0.2.tar.gz`、`bootstrap_manifest.json` 或 `bootstrap_report.json`。不得把这些未知值报告为 0。

## 3. 已实现的仓库流程

| 文件 | 实现内容 |
|---|---|
| `trainer/build_detector_dataset_v02.py` | 新增独立 v0.2 构建器；读取 Accepted Pool CSV/metadata，并将每条 `source_review_id` 与数据库人工审核记录、`accepted_bbox` 和原图 URI 对照；实际下载每个唯一原图 URI、计算原始字节 SHA、读取 EXIF 并校验归一化尺寸。数据检查失败时只生成 `BLOCKED` 诊断，不生成可训练工件。 |
| `trainer/build_detector_dataset_v02.py` | 仅用完整场景原图；拒绝把 Accepted Pool crop 当 detector 输入；`accepted_bbox` 是唯一正标签，`detector_bbox` 与 Google Vision 框只记录/忽略。分别核对 `bbox`、`pixel_bbox`、`source_size`、`crop_left/top/right/bottom`，并明确原图坐标与分类 crop 坐标语义。 |
| `trainer/build_detector_dataset_v02.py` | 按原始 SHA 与来源组连接成分确定性分配 80/10/10 split；按 SHA 合并相同原图、合并同图多个人工框；隔离已冻结困难样本；负样本只接受显式 `human_override=no_fish` 且有匹配 ReviewEvent 的记录。 |
| `trainer/build_detector_dataset_v02.py` | 写出 COCO train/val/test、逐图审核 provenance 和原/物料化 SHA；READY 前校验 Manifest、COCO、归档哈希和 split；本地输出原子写入、不覆盖；GCS 上传使用 generation precondition，先检查全部对象，最后写 report。相同归档/Manifest 的重试可复用语义相同的冻结报告，不覆盖既有对象。 |
| `trainer/detector_dataset_gate.py` | 逐行检查 SHA、审核记录与 BBox provenance、图像完整场景类型、分组/重复跨集合泄漏、冻结困难样本、实际源图字节审计、Manifest/归档 SHA、split 计数、COCO一致性及人工负样本。 |
| `app/det_fish_console.py` | 训练状态 API 校验真实 report、Manifest、归档 SHA；分别返回数据集门禁、v0.2 训练 Job 配置门禁和 execution-history 门禁；训练 Job 必须使用固定 Artifact Registry digest、v0.2 dataset/model 版本、完整代码 SHA 和 L4 资源。启动接口再次执行这些门禁。 |
| `app/platform/services/crop_dataset.py` | Accepted BBox pool 统计条件与来源查询一致，需 `ImageAsset.review_status=approved` 且 `training_eligible=true`。 |
| `scripts/build_det_ds_v02_cloud_shell.sh` | 提供项目方现有 Cloud Shell 的 CPU-only 构建入口。要求 checkout 与 PR #166 head 完全一致且工作树干净；先核验现有 Artifact Registry、Job 名称、Cloud Run 控制台服务配置、服务账号、Secret 与 Bucket；失败即停止，不创建 IAM、不读取 Secret 值、不启动 GPU 训练。 |

构建任务日志只输出计数、SHA、有限的 reconciliation 问题样例和错误类别，不打印原图内容或逐图 GCS URI。

Manifest 使用 `training_eligibility_basis` 记录数据库筛选条件，不宣称该字段是独立法律同意记录。`ImageAsset.training_eligible` 在 schema 中有默认值，因此报告不会把它包装为额外人工授权证据。

## 4. 本地测试与失败

| 检查 | 结果 |
|---|---|
| v0.2 构建器与严格数据门禁 | **15 passed** (`tests/test_detector_dataset_gate.py`、`tests/test_detector_dataset_v02_builder.py`) |
| DET_FISH 控制台门禁/API 测试 | **7 passed** (`tests/test_det_fish_console.py`) |
| Accepted BBox pool approved 条件回归 | **1 passed** (`tests/test_accepted_bbox_pool_count.py`) |
| Python compile、Cloud Shell / trainer deploy 脚本 `bash -n`、`git diff --check` | **通过** |
| Accepted Pool 两个既有测试模块的隔离运行 | **3 passed、4 failed**；4 项均在测试 fixture 的 `Base.metadata.create_all()` 阶段失败：`fish_cards.asset_version_id` 外键引用的 `fish_knowledge_asset_versions` 表未加载。失败发生在被测函数执行前。本次新增的独立 approved-count 回归测试通过。 |

以上是本地测试；合成 fixture 通过只验证代码合同，不代表 6,308 条真实数据、GCS 工件、Cloud SQL 或 Cloud Run Job 已通过。

## 5. 云端状态：BLOCKED / NOT_VERIFIED

本次实际执行环境：

- `gcloud` 命令不存在。
- `google.auth.default(scopes=[cloud-platform])` 返回 `DefaultCredentialsError: Your default credentials were not found`。
- 因此没有向 GCS、Cloud SQL 或 Cloud Run 发起真实数据读取或写入，也未运行构建 Job。
- 未读取本次实时 Console Revision/服务账号/Secret 引用、当前 DET_FISH 训练 Job 镜像 digest/配置、Job 执行历史或 `/api/det-fish/training/status`。
- 没有 GPU 训练、模型产物写入、生产切换、IAM 修改或 GCP API 启用。

先前项目记录曾指出 Accepted Pool metadata 自报 6,308 条、`DET_DS_v0.2/` 尚未存在，以及当时训练 Job 仍是 v0.1 配置；这些是先前状态，不作为本次实时 GCS/Job 核验。当前 v0.2 训练 Job 的项目、区域、镜像 digest、服务账号、GPU 类型/资源和 execution history 均为 `NOT_VERIFIED`。项目负责人在 Cloud Shell 执行后，必须以当时的真实输出更新这些状态。

## 6. 门禁状态

| 门禁 | 本次状态 | 原因 |
|---|---|---|
| Accepted Pool ↔ 数据库逐条标签核对 | `BLOCKED_NOT_RUN` | 无 CSV/metadata 和 Cloud SQL 数据快照 |
| 全量原图存在性、SHA、EXIF、尺寸与 BBox 核对 | `BLOCKED_NOT_RUN` | 无 GCS ADC；没有逐图证据 |
| 负样本与困难留出验证 | `BLOCKED_NOT_RUN` | 无数据库与原图 |
| v0.2 Manifest/COCO/归档冻结 | `BLOCKED_NOT_BUILT` | 必需输入未读取；没有冻结文件或 SHA |
| 训练页面真实数据集门禁 | `NOT_VERIFIED` | 未访问部署服务/API；仓库代码的合成测试通过 |
| Cloud Run 训练 Job 配置 | `NOT_VERIFIED` | 无 GCP 凭据；当前 Job 不可读取 |
| Cloud Run execution-history | `NOT_VERIFIED` | 无 GCP 凭据；当前执行历史不可读取 |
| GPU 训练 | `NOT_RUN` | 本任务明确不启动训练 |

## 7. 项目方 Cloud Shell 构建命令

在项目方已登录且有既有权限的 Cloud Shell 中，从 Yujian 仓库运行。该脚本会在权限缺失或资源身份不符时停止；不会自动开 API、增加 IAM 或运行 GPU 任务。

```bash
git fetch origin refs/pull/166/head:refs/remotes/origin/pr-166-live && \
git checkout --detach refs/remotes/origin/pr-166-live && \
PROJECT_ID=gemini-api-project-503706 \
DETECTOR_DATASET_BUILDER_REGION=asia-east1 \
DETECTOR_ARTIFACT_REGION=asia-southeast1 \
GCS_BUCKET=yujian-model-factory-571785698442 \
bash scripts/build_det_ds_v02_cloud_shell.sh
```

构建器只会在全部标签、原图、分组划分、负样本、COCO 和 SHA 门禁通过后，将以下三个不可静默覆盖的文件发布到目标前缀，并最后写报告作为冻结提交标记：

- `gs://yujian-model-factory-571785698442/detector-datasets/DET_DS_v0.2/DET_DS_v0.2.tar.gz`
- `gs://yujian-model-factory-571785698442/detector-datasets/DET_DS_v0.2/bootstrap_manifest.json`
- `gs://yujian-model-factory-571785698442/detector-datasets/DET_DS_v0.2/bootstrap_report.json`

本报告生成时，上述文件均**没有**在本执行环境实际创建或读取；未填造 SHA 或 PASS 状态。
