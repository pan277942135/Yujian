# 渔见 AI / YuJian AI｜项目开发指导文档

> **用途：项目级 Source of Truth。**
>
> 后续新开开发任务时，必须先读取本文件，再读取目标仓库 README、相关 `docs/`、最新 `main` HEAD 与 CI 状态；不要仅依赖历史对话摘要。
>
> 最后更新：2026-08-31

---

## 1. 项目定义

**渔见 AI / YuJian AI** 是面向中国钓友的鱼种识别与鱼获记录产品。

项目最终要形成持续数据飞轮：

```text
真实钓友鱼获图片
→ AI 识别
→ 用户确认 / 纠错
→ Feedback
→ 人工 Review
→ Accepted Label
→ Dataset Freeze
→ Training
→ Evaluation
→ Model Release
→ App 新版本
→ 新一轮真实用户数据
```

核心竞争力不是单纯的 App UI，也不是单个分类模型，而是：

> **真实中国钓友手机拍摄分布的数据资产 + 可持续训练/评测/发布闭环 + 可商用的移动端识鱼体验。**

---

## 2. 项目分为两大主线

### 2.1 后台侧：AI Model Factory

仓库：

```text
https://github.com/pan277942135/Yujian
```

定位：

> **当前仅供内部使用的 AI 生产后台。**

现阶段不为 SaaS、多租户、复杂 RBAC、计费或外部模型平台过度设计。等业务与团队规模化后再考虑扩展。

后台职责：

```text
数据导入
→ Batch Registry
→ 鱼体检测 / 数据清洗
→ 去重
→ 人工审核 / Species Truth
→ Approved Master Pool
→ Dataset Freeze
→ Training Run
→ Evaluation
→ Model Registry
→ 模型实测
→ Mobile Export
→ Runtime Parity
→ 发布可供 App 集成的模型
```

同时负责 App Feedback 回流：

```text
App confirmed / corrected / new species candidate
→ Feedback API
→ Feedback Pool
→ 人工 Review
→ Accepted Label
→ 下一版 Dataset
```

后台侧最重要的目标：

- 数据质量与可追溯性
- Dataset / Model 版本不可混淆
- 模型真实准确率
- Hard Negative / Hard Pair 能力
- 模型发布质量
- TorchScript / LiteRT / Android Runtime 一致性
- 反馈数据可进入下一轮训练闭环

### 2.2 前端侧：YuJian Android App

仓库：

```text
https://github.com/pan277942135/Yujian_App
```

定位：

> **面向真实钓友的消费级 Android 产品。**

当前 App 仍处于第一版、0→1 早期阶段。已有部分真实技术能力，但内容、信息架构、UI、UX、鱼获资产体系、图鉴与商用稳定性都需要持续优化。

必须明确：

> **Runtime Parity PASS 只代表“AI 能力可信地接入 App”，不代表 App 已达到商用 MVP。**

App 主要用户旅程：

```text
钓到鱼
→ 拍照 / 相册
→ AI 识鱼
→ 查看 Top-1 / Top-3 与把握度
→ 用户确认 / 修正
→ 保存鱼获
→ 查看个人鱼获资产
→ 图鉴 / 收藏 / 分享等后续体验
```

用户侧应避免暴露工程语言，例如 logits、tensor、dataset version、inference 等；内部可以记录完整技术元数据。

---

## 3. 两侧边界

### 后台侧负责

- 数据
- 审核
- Dataset
- Training / Evaluation
- Detector / Classifier
- Model Registry
- Mobile Export
- Model Contract
- Golden Set / Runtime Parity
- Feedback Review

### App 负责

- 拍照 / 相册
- 图片生命周期与本地预处理
- 调用已发布移动模型
- 识别结果与低置信度交互
- 用户确认 / 纠错
- 鱼获记录
- 图鉴与消费级产品体验
- Feedback 上传

### 核心边界原则

> **App 不应该“猜”模型如何使用。后台发布模型时必须明确 Model Contract。**

每个移动模型发布应逐步标准化包含：

```text
model file
model_version
model_sha256
model_family
input_shape
input_layout
input_dtype
color_order
resize_mode
interpolation
pixel_scale
mean / std
output_shape
output_type
class_map
class_map_sha256
preprocess_version
export_version
golden test / parity result
```

---

## 4. MVP 模型目标

MVP 阶段不是追求“鱼种越多越好”，而是：

> **先把中国钓友最常见的约 20–50 类鱼种做深，并在真实钓鱼拍摄分布下达到可商用识别水平。**

评价不能只看总体 Top-1 Accuracy，还应持续看：

- 真实鱼获场景准确率
- Top-1 / Top-3
- Confusion Matrix
- Hard Pair 准确率
- 低置信度 / 拒识能力
- 置信度校准
- 夜钓
- 手持
- 抄网 / 鱼护
- 草地 / 水边 / 泥地
- 鱼体弯曲
- 反光
- 遮挡
- 小鱼 / 大鱼
- 不同角度与生长阶段

重点 Hard Pair 示例：

```text
草鱼 ↔ 青鱼
鲫鱼 ↔ 鲤鱼
鲢鱼 ↔ 鳙鱼
黑鱼 ↔ 加州鲈
黄骨鱼 ↔ 鲶鱼类
```

---

## 5. MVP 识别架构：YOLO Detector + Species Classifier

MVP 需要逐步接入 **YOLO 鱼体检测模型**。

目标识别链路：

```text
用户原始照片
→ YOLO Fish Detector
→ 鱼体 bbox
→ bbox 合理扩边 / 保留完整鱼体
→ Letterbox / classifier preprocess
→ Species Classifier
→ Top-K / Confidence
→ 用户结果页
```

原因：真实照片中常同时存在人、手、鱼竿、抄网、鱼护、草地、水、桶、鞋等背景。只做整图分类容易学习背景 shortcut。

### YOLO 裁剪原则

不要简单使用紧贴 bbox 的裁剪：

```text
YOLO bbox
→ tight crop
→ 224×224
```

这可能裁掉尾鳍、背鳍、嘴、须、头部轮廓等关键分类特征。

推荐方向：

```text
bbox
→ 约 10–20% 合理扩边（后续通过实验确定）
→ 保留完整鱼体
→ 保持比例 Letterbox
→ Classifier
```

未来 Model Registry 应能够区分至少两类模型：

```text
DETECTOR
  └── Fish Detector / YOLO

CLASSIFIER
  └── Species Classifier
```

后续规模化可以演化为 Recognition Bundle，但当前不要为了 Bundle 管理系统阻塞 P0。

---

## 6. MVP 模型发布策略

MVP 阶段采用最简单、稳定的策略：

```text
训练新模型
→ 后台评测
→ Mobile Export
→ Python Runtime Parity
→ Android Runtime Parity
→ PASS
→ 集成 App
→ App 发版时一起发布模型
```

当前阶段 **不做**：

- App 远程动态下载新模型
- 灰度模型切换
- 在线 Model Registry 下发
- 自动 rollback

规模化以后再考虑：

```text
App 内置基础模型
+ 后台远程下载新模型
+ SHA 校验
+ 版本管理
+ rollback
```

即使 MVP 暂不做远程模型，也必须从现在开始保留完整 `model_version`、SHA、class map、preprocess/export contract，避免未来重做治理体系。

---

## 7. Runtime Parity 是正式 Model Release Gate

不能再把：

```text
成功导出 TFLite
=
移动模型完成
```

正确发布链路必须是：

```text
Training
→ Evaluation
→ Model Registry
→ Export TFLite / LiteRT
→ Python TFLite Parity
→ Android Runtime Parity
→ PASS
→ Publish Mobile Release
```

推荐至少建立：

```text
20–50 类扩展前：当前类别 × 多张 Golden Images
规模化后：每类至少 3 张以上代表性 Golden Images
```

同一 Golden 输入至少比较：

- 完整 logits / probability vector
- Top-1
- Top-3
- `max_abs_diff`
- `mean_abs_diff`
- cosine similarity

---

## 8. 当前技术检查点（2026-08-31）

当前主分类模型：

```text
MODEL_M1_v0.2
RUN_M1_v0.2_003
DS_M1_v0.2
```

当前 9 类：

```text
0 grass_carp         草鱼
1 bighead_carp       鳙鱼
2 silver_carp        白鲢
3 common_carp        鲤鱼
4 crucian_carp       鲫鱼
5 largemouth_bass    加州鲈
6 snakehead          黑鱼
7 yellow_catfish     黄骨鱼
8 black_carp          青鱼
```

当前移动模型文件：

```text
fish_classifier_v0_2.tflite
SHA256: 9575ede5c6c85b850647016d76e8e5175fa9ea6b609c47c83f54b4062e47d14e
```

当前已知实际输入 tensor：

```text
[1, 3, 224, 224]
```

输出：

```text
[1, 9]
```

当前 P0 阻塞：

> **同一 Golden Image 在 TorchScript 与 Android TFLite Runtime 下结果不一致。**

已验证常见 RGB/BGR、NCHW/NHWC、ImageNet/0..1 组合均不能解释差异，因此不要继续凭感觉修改 Android preprocess。

下一步判断顺序：

```text
同一 Golden Image
→ TorchScript 完整 9 维输出
→ 保存完全相同输入 tensor
→ Python LiteRT/TFLite 完整 9 维输出
→ 比较数值
→ Android 读取同一 tensor
→ 完整 9 维输出
```

决策树：

```text
TorchScript != Python TFLite
→ Export / LiteRT conversion 问题
→ 停止修改 Android preprocess
→ 修 Mobile Export
```

```text
TorchScript ≈ Python TFLite
但 Python TFLite != Android
→ Android Runtime / ByteBuffer / Interpreter / Delegate 问题
```

优先推荐的下一代 Mobile Export 方向：

```text
Android 输入：NHWC RGB
Wrapper 内：scale / normalize + NHWC→NCHW + classifier
```

尽量把预处理 contract 封装进模型，减少训练端与客户端漂移。

> 本节是时间敏感检查点。执行新任务时必须先查询 GitHub 最新 `main` HEAD、CI 与代码，若已经推进，以最新仓库状态为准。

---

## 9. Android App 当前阶段原则

Android App 当前仍是产品第一版，不应因为真实 TFLite 已接入就认为接近商用 MVP。

后续重点包括但不限于：

- 产品信息架构
- 首页与识鱼主路径
- 拍照 / 相册 UX
- 识别中状态
- 识别结果页
- Top-3 / 低置信度交互
- 用户纠错
- 图鉴内容与结构
- 我的鱼获资产体系
- 持久化
- 分享体验
- 图片生命周期
- 权限与异常状态
- Feedback 稳定上传
- Crash / 性能 / 真实设备测试
- Release signing / 正式 package

技术方向可在合适阶段逐步采用：

```text
Room              → 鱼获持久化
WorkManager       → Feedback 重试
Model metadata    → 版本/SHA/可追溯
```

但不要为了未来架构提前过度工程化。

---

## 10. 当前阶段不要优先做的事情

在 Runtime Parity 未解决前，不要把后台精力转向：

```text
重新训练 v0.3
INT8 / 量化
远程模型更新
复杂模型灰度
复杂 SaaS 权限
```

App 侧可以继续产品设计/研究，但不能用 UI 改动掩盖模型正确性问题。

模型 parity 修复后，再按数据与产品价值决定 20–50 类扩展、YOLO 接入、App UX 深化的节奏。

---

## 11. 代码与执行规则

两个 GitHub Repo 是唯一代码 Source of Truth：

```text
Backend / Model Factory:
pan277942135/Yujian

Android App:
pan277942135/Yujian_App
```

每次工程任务应遵守：

```text
读取 PROJECT_GUIDE.md
→ 读取目标仓库 README / 相关 docs
→ 核对最新 main HEAD
→ 核对最新 CI / 当前失败点
→ 修改
→ 测试
→ commit
→ push
→ CI
→ 返回真实 Commit SHA / CI 结果
```

禁止：

- 只修改临时本地文件而不 push
- 使用历史聊天中的旧 SHA 当作当前事实
- 未测试就宣称 PASS
- Runtime Parity FAIL 时把 APK “能运行”当作模型验收
- 用户纠错未经 Review 直接成为 Ground Truth

---

## 12. 新任务启动协议

以后任何新的 YuJian 开发会话，建议第一条执行指令直接采用：

```text
先读取：
1. pan277942135/Yujian/docs/PROJECT_GUIDE.md
2. 当前目标仓库 README.md
3. 与任务相关的 docs / source code
4. 最新 main HEAD 和 CI 状态

以 GitHub 最新状态为 Source of Truth。
不要从历史聊天重新猜项目架构。
如果 PROJECT_GUIDE 的“当前技术检查点”与最新代码/CI冲突，以最新代码/CI为准，并在完成阶段后更新文档。
```

---

## 13. 长期演进方向

### 后台规模化后

可考虑：

- 多角色协作 / RBAC
- 数据任务分配
- 更完整 Evaluation Dashboard
- Detector + Classifier Recognition Bundle
- 远程模型发布与灰度
- Model rollback
- 自动化 Hard Case Mining
- 主动学习 / 半自动标注

### App 规模化后

可考虑：

- 内置 Base Model + Remote Model
- SHA validation
- rollback
- 更丰富鱼获资产
- 社交 / 分享 / 成就体系
- 更大范围鱼种覆盖

但所有扩展都应服从一个原则：

> **先把真实钓友最常见 20–50 类做准、做稳、做到可商用，再扩大系统复杂度。**
