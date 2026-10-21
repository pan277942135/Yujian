# DET_FISH 无鱼负样本池 V3 实施与历史记录核查状态

**状态：** `IMPLEMENTED / HISTORICAL_SOURCE_AUDIT_BLOCKED_IN_WORK_ENVIRONMENT`
**PR 基线：** `7448ddcc005ae31adb923f8ab889edd86f3f2b91`
**本报告用途：** 记录本次代码实施、验证范围及未能从当前 Work 执行环境取得的真实历史工件证据。

## 历史 DET_DS_v0.1 无鱼候选

本次执行没有取得 Accepted Pool/生产数据库或 GCS 的访问凭据，不能独立读取 `DET_DS_v0.1/bootstrap_manifest.json`、其报告、历史图片记录或原始图片对象。因此本报告不填报历史 154 条的有效数、已入池数、待确认数、重复数、缺失数或排除数；这些值均为 `NOT_VERIFIED`。既有调查材料中出现过“154”这一历史计数，但它未在本次环境重新核对，不能作为当前实测值。

本地访问能力检查结果：

| 检查项 | 结果 |
| --- | --- |
| `gcloud` CLI | `NOT_INSTALLED` |
| Application Default Credentials | `UNAVAILABLE (DefaultCredentialsError)` |
| `GCS_BUCKET` | `NOT_SET` |
| Cloud SQL / Registry DB 配置 | `NOT_SET` |
| 实际历史 Manifest、报告、图片字节核查 | `NOT_RUN` |

代码新增了受控制台登录保护的只读接口 `GET /api/det-fish/negative-pool/historical-v01-audit`。它从配置的 GCS bucket 读取 v0.1 报告和逐图 Manifest，核对报告来源与 Manifest 无鱼行数，逐条关联数据库图片记录、读取原始 GCS 对象并重新计算 SHA-256，再统计有效、已入池、待确认、重复、缺失和排除记录。该接口执行后返回的结果才是历史数据核查值；本次未能调用它，故不把代码能力报告为真实数据审计通过。

历史 v0.1 的 Google Vision `no_fish` 输出仅作为待人工确认候选。它不会自动成为 DET_FISH 负样本；加入动作要求操作者逐张确认完整场景无鱼，并只记录到 DET_FISH 专用池。

## 实施范围

- 新增 DET_FISH 专用负样本成员表和追加式 ADD/RE_ADD/REMOVE 审计表；移除是软移除。
- 训练页负样本卡片读取同一活动池计数，并可打开池列表和第二层候选 Modal。
- 候选来自当前系统 no-fish 记录或可核验的 v0.1 历史 Manifest；AI 结论显示为待人工确认。
- 添加时从原 GCS 完整场景图流式计算 SHA-256，校验训练资格、已知重复和鱼体标签冲突；不改写原图片记录或审批记录。
- v0.2 builder 只读取活动池成员，并要求最新 ADD/RE_ADD 事件、操作者、时间、确认快照及入池 SHA；下载原图后重算 SHA。不匹配、正负冲突、重复或来源无效时构建保持 BLOCKED。
- COCO 保留无鱼图片的 `images` 项，不创建鱼体 `annotations`。按现有 SHA/来源组规则分组，移除成员只影响之后的数据集构建。
- 未修改批量审批页面、冻结 v0.1 工件、训练阈值或 GPU 训练门禁。

## 本次离线验证

在隔离的临时 Python 环境运行：

```bash
PYTHONPATH=. pytest -q tests/test_det_fish_negative_pool.py tests/test_detector_dataset_v02_builder.py tests/test_detector_dataset_gate.py
```

结果：`17 passed`。另对新增 Python 模块执行 `compileall`，无语法错误。测试覆盖 AI-only 候选必须明确确认、加入幂等、移除/重加审计、原始 Presence 记录未被修改、构建器仅读活动池成员、v0.2 原图 SHA 校验、Manifest/门禁与 COCO 无框负样本合同。

以上均为本地代码/fixture 验证，不是 GCS 历史 154 条实时核查，不是 Cloud Run 页面验收，也不是数据集冻结或训练 PASS。
