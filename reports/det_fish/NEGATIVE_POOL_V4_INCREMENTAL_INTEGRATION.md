# DET_FISH 负样本池与 DET_DS_v0.2 增量接入 V4

**状态：** `IMPLEMENTED / LOCAL_TESTS_PASS / LIVE_GCS_AUDIT_NOT_RUN / NOT_DEPLOYED`
**实现基线：** PR #166 HEAD `7448ddcc005ae31adb923f8ab889edd86f3f2b91`
**目标：** 在现有 DET_FISH 训练页和数据集构建流程内增加负样本池管理，不重写训练任务、Cloud Run Job、训练历史或模型登记。

## 基线与复用范围

PR #166 在本次改动前为 Open、Draft、Unmerged，分支 `feat/det-fish-training-and-bbox-console`，base `main` SHA `05554ce1b7104c7bce05610979dfc3478b6d24a5`。实时 PR HEAD 在实施和本地验证期间为 `7448ddcc005ae31adb923f8ab889edd86f3f2b91`。该 SHA 的最近 GitHub Actions PR CI run #1231 已成功；本次增量提交后的 CI 结果需以新提交运行状态为准。

继续使用现有 `/api/det-fish/training/status`、`/api/det-fish/training/start`、现有 `DETECTOR_TRAINING_JOB_NAME` Cloud Run Job 调用和同一执行历史读取、`trainer.train_detector` 入口、现有模型 GCS 路径及 `_model_summary` 模型登记/元数据读取。没有新增训练 Job、训练历史 API、GPU 服务、结果表或模型注册表。负样本池成员/审计事件仅记录 DET_FISH 专用成员资格，不代表训练运行历史。

正样本仍由 `trainer/build_detector_dataset_v02.py` 从审核通过的 `BatchCropReview.accepted_bbox_json` 和完整原图读取；v0.1 冻结工件仍只读。负样本构建输入来自活动池成员。当前数据库没有可复用的 DET_FISH 专用可撤销成员关系与逐次操作审计表，因此增加两张最小附加表：`det_fish_negative_pool_entries` 和追加式 `det_fish_negative_pool_events`。它们不改变原 Presence、审批或历史数据集记录。

## 负样本功能

- 训练页原有布局、门禁、参数、执行历史和验收卡片保留；仅把“确认无鱼负样本”数量卡片改为同一活动池 API 的真实可点击计数。
- 一级 Modal 展示当前/已移除成员、来源、原图链接、时间、操作者及构建资格；支持分页、筛选、批量软移除和重新加入。
- 二级 Modal 候选来自现有 `FishPresenceResult.status=no_fish` 和 `DET_DS_v0.1/bootstrap_manifest.json`，支持分页、批次/确认状态筛选、跨页选择及原图查看。机器判定只作为待确认候选；添加请求要求操作者输入名称并明确勾选无鱼确认。添加不会改写原 Presence 或审批事件。
- 添加前检查完整场景 GCS URI、实际对象读取及 SHA-256、训练资格、已知重复和正样本冲突。图片 ID 和活动源 SHA 唯一约束、幂等响应、软移除与 ADD/RE_ADD/REMOVE 追加审计防止重复成员并保留重加记录。
- 历史来源只读读取 v0.1 manifest 并按其中原始 `image_asset_id`/批次图像 ID 关联旧记录，不重建或修改 v0.1 工件。界面提供显式“核查历史记录”动作；代码核查接口会读取所有匹配原图字节并计算 SHA，不会在普通打开弹窗时自动遍历图片。

操作人字段使用当前控制台允许操作者提供的名称记录。由于该控制台使用既有共享访问机制，池事件中的名称不是独立账号身份认证凭证；这项身份限制没有被包装成更强的审计保证。

## 构建器和 YOLOX 空标注图核验

v0.2 构建器只读取活动池成员，确认最新 ADD/RE_ADD 操作具有操作者、时间、明确人工确认和合法来源，再下载完整原图并核验其当前 SHA 与入池 SHA。无鱼图片写入 COCO `images`，`annotations` 不产生该图片的鱼体框；来源组和原图 SHA 继续由现有连通组件逻辑统一划分。任何活动成员存在缺证、哈希不一致、重复或标签冲突都会写入构建错误并阻止冻结。移除只影响后续构建，不回写已冻结版本。

`Dockerfile.detector` 固定从 YOLOX `0.3.0` tag 构建训练镜像。该 tag 的官方 `COCODataset` 使用 COCO `getImgIds()` 建立样本索引，因此无 annotation 的 COCO image 仍进入样本清单；`load_anno_from_ids()` 对其取得零条 annotation，并创建形状为 `(0, 5)` 的空标签数组。训练代码仍调用同一个 `/opt/YOLOX/tools/train.py`。官方源文件：[YOLOX 0.3.0 `coco.py`](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/yolox/data/datasets/coco.py)。本仓库新增构建回归断言确认打包 COCO 每个负样本均保留为图像且没有关联 annotation；未启动 GPU 训练。

## 历史 154 条记录状态

用户在 V4 指令中提供的历史事实为：v0.1 `bootstrap_manifest.json` 包含 154 条无鱼记录及原图 GCS URI。该数值在本报告中作为用户提供的既有来源说明记录，不声称是本执行环境重新读取的实测结果。

当前 Work 环境没有 `gcloud`，没有可用 ADC，也没有 `GCS_BUCKET` 或 Cloud SQL/Registry DB 配置，故本次没有读取历史 Manifest/report 的字节、核对 154 个源对象或关联线上图片数据库。历史记录总数的实时复核、有效/已入池/待确认/重复/缺失/排除统计、旧图片 SHA 均为 `NOT_VERIFIED`。部署后可通过二级 Modal 的“核查历史记录”按钮调用 `GET /api/det-fish/negative-pool/historical-v01-audit` 得到实际明细；其返回值才是实时审核结果。本地实现不能替代该运行时数据核对。

## 数据库与审批页边界

- 新表通过现有 `Base.metadata.create_all` 启动路径创建；没有新增 DB、迁移服务或数据登记系统。
- 本次没有触碰批量审批页、`app/bulk_review.py` 或其模板、脚本、审批流程。
- 未更改模型、阈值、识别合同、GPU 门禁、训练任务配置、生产模型或冻结 v0.1。

## 本地验证

本次目标测试命令：

```bash
PYTHONPATH=. /tmp/yujian-neg-pool-venv/bin/pytest -q \
  tests/test_det_fish_negative_pool.py \
  tests/test_detector_dataset_v02_builder.py \
  tests/test_detector_dataset_gate.py \
  tests/test_det_fish_console.py
```

结果：`25 passed`。回归覆盖池 API 逻辑、池事件幂等/软删除/重加、原始 Presence 不变、既有人工无鱼审核 provenance、活动成员构建接入、原图 SHA 核验、manifest/COCO 门禁及负样本 COCO image 无 annotation。Python `compileall`、Jinja 模板解析、内嵌 JavaScript `node --check`、`git diff --check` 均通过；OpenAPI 中页面与新增 6 个负样本池路由检查通过（7 个路由断言）。PR 当前基线 SHA 的 Actions run #1231 成功；本次新提交后的仓库 CI 仍需以新运行结果为准。

这些是本地代码和 fixture 验证，不代表 Cloud Run 部署、历史 154 条真实数据核验、Dataset 冻结、训练 Job 启动或训练完成。

## 页面访问与当前状态

页面路由：`/platform/model/det-fish-training`。候选生产服务 URL 为 <https://yujian-model-factory-console-6254hdioya-de.a.run.app/platform/model/det-fish-training>，但本 V4 HEAD 尚未部署到 Cloud Run；不能把该地址报告成已经运行此实现的验收环境。PR #166 应继续保持 Draft / Open / Unmerged，项目负责人可在其获批部署环境前端验收。
