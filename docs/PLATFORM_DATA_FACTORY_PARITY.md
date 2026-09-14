# YuJian Platform 数据工厂 Legacy 功能等价矩阵

版本：YuJian AI Platform V1 P0

本矩阵用于防止 Platform 导航收敛时误删 Legacy 成熟能力。P0 只整理入口和审核基础设施；未达到 FULL PARITY 前，不隐藏对应 Legacy 页面。

| Legacy 能力 | 原路径 | Platform 状态 | 优先级 | 说明 |
|---|---|---|---|---|
| 数据导入 | `/batches/upload` | Platform 入口复用 | P0 | `/platform/data/import` 薄入口，继续使用成熟 Batch 上传 |
| Batch 查看 | `/batches` | 暂复用 Legacy | P1 | Batch 是采集/处理生命周期，不在 Dataset 列表伪装成 Dataset |
| 快速审核 | `/review/bulk` | 未完全迁移 | P1 | Platform 审核中心复用审核状态机；完整快速审核体验保留 |
| 单张审核 | `/review` | 部分迁移 | P1 | Platform 支持当前页详情与批量操作，复杂 Legacy 能力保留 |
| Crop Review | `/crop-review` | 部分迁移 | P1 | Platform“调整框”暂跳 Legacy 可视化编辑页 |
| accepted_bbox | `/datasets/accepted-bbox` | 先跳 Legacy | P0 | 不再要求输入 JSON；使用 Legacy 可视化框选逻辑 |
| Crop QA | `/crop-qa` | 尚未等价迁移 | P1 | 保留原入口，不从 Platform 菜单伪装为已迁移 |
| Inspect | `/inspect` | 尚未等价迁移 | P1 | 保留原入口和完整检查能力 |
| Dataset Freeze | `/datasets` | 后续 Dataset 详情迁移 | P1 | DatasetVersion 列表只展示真实数据集版本 |

## P0 边界

- Batch 表示原始采集上传及 AI 清洗生命周期。
- DatasetVersion 表示经过审核/冻结后可用于训练的数据版本。
- `/api/platform/datasets` 只返回 DatasetVersion；任何 `BATCH_*` 都不得进入该响应。
- `/platform/data/queue` 保留兼容路由并重定向到 `/platform/data/review`，队列统计能力合并在审核中心顶部。
- 只有矩阵中的状态更新为 `FULL PARITY` 后，才允许隐藏或移除对应 Legacy 页面。
