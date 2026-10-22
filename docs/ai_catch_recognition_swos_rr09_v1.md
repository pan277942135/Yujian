# 渔见 AI 渔获识别 Web V1：SWOS / RR09 连续性实现与验收记录

状态：`PARTIAL`。本分支交付了可审查的 Web 页面、真实推理适配端点、结果表单接线和局部自动化验证；目前没有服务端真实推理通过证据，也没有获得阶段 0 设计采纳、SAM 生产资格或公开照片授权，因此不得标记为正式上线就绪。

## 1. 仓库与控制面审计

| 证据 | 结果 |
|---|---|
| 仓库 | `pan277942135/Yujian` |
| 初次审计基线 `origin/main` | `53e84e77c302981fafe9f47818a67de420f0166a` |
| PR 前最新 `origin/main` | `04db31548a3773aa087d34d10bdee33ec8fcd1cc` |
| 功能分支 | `feat/ai-catch-recognition-web-swos-rr09-v1` |
| Yujian_App 最新 main | `888706fda16d6f343f9bb4e02463f3b7a3526c31` |
| App 阶段 0 PR #186 | Draft / Open / Unmerged；HEAD `3b0b5848206cdea949f4bf6be7eb7cb1d19b1e34` |
| App SWOS 15 帧 PR #187 | Draft / Open / Unmerged；HEAD `ee3bb45858183144598bc28c5bae53ae55b324e0` |
| Yujian 模型平台 PR #176 | 已合并进入 `04db31548a3773aa087d34d10bdee33ec8fcd1cc`；其 CI/Chromium 展示检查通过，但标注为 `UI_SIMULATION_ONLY`；记录的真实推理 UAT 仍为 `Failed to fetch`，Cloud Run Revision SHA 仍存在不一致 |
| Yujian 控制面 | 本仓库没有 `AGENTS.md`、`CONTROL_PLANE_REGISTRY.yaml` 或产品级 Admission 注册入口；不可把 App 设计仓库的 admission 结果转用为本仓库产品 Admission |
| App 控制面审计 | 在 App 的设计候选分支运行验证器得 `CONTROL_PLANE_ADMISSION=PASS`，但那只是该候选分支的静态控制面结果，不是本产品功能准入；#186/#187 未采纳 |

阶段 0 与本功能必须保持独立。此 PR 不复制 App 中的 SWOS/RR09 合同，不修改历史 RR09 V2，也不将未采纳提案声明为生产权威。

## 2. 当前已接线行为

### Web 页面

- 新页面：`/platform/model/catch-recognition`，按现行 `PLATFORM_PAGES` 注册，沿用 Platform V1 外壳和导航。
- 页面明确说明本次识别不保存照片；只有用户点击保存并完成鱼获账户认证后，才调用既有 catch 图片上传和保存接口。
- 没有随仓库发布的示例鱼获照片，因为已检查素材未找到可核验的公开使用授权。未上传或复制 `208758.jpg`、私有截图或模型测试照片。
- 原图浏览、推理、真实检测框调试开关和结果照片复用浏览器本地同一 Blob URL。精确鱼体轮廓保持关闭。
- 识别结果从实际 `/api/product/catch-recognition/infer` 响应更新。MEDIUM / LOW 需要人工选择目录鱼种；无鱼、质量问题或技术错误不播放成功揭晓。
- “继续记录记忆”保持禁用，因为仓库没有已登记的 Web 记忆目标页面。

### 推理端点

`POST /api/product/catch-recognition/infer`：

1. 只接收单张 JPEG、PNG、WEBP；校验编码、大小及像素数。
2. 从模型注册表中仅选取唯一同时满足 `is_production=true` 与 `status=PRODUCTION` 的分类器。缺失或多条均返回可恢复 503，不回退到候选版本。
3. 调用既有 `_predict_bytes(..., include_intermediates=False)`，复用 Detector → 质量门 → Crop → Classifier；不复制模型逻辑，也不触发现有推理历史图片存储路径。
4. 返回真实 Detector / Classifier 版本、BBox、Top-3、质量门、阶段耗时和请求 ID。中间图、base64、GCS URI 不进入产品响应。
5. SAM 状态固定标记 `UNAVAILABLE / SAM_NOT_QUALIFIED`，不输出 Mask。

应用现有 `CONSOLE_ACCESS_KEY` 中间件会保护该 `/api/product/...` 端点和 `/platform/...` 页面；此实现沿用当前平台访问边界，不把其误报为匿名消费者 API。

## 3. 3+2 决策和阈值差异

页面适配器读取真实流水线字段，并按已审阅的 Yujian_App `main` 上 3+2 观察合同映射状态：未分类且 `no_fish` → `ERROR_NO_FISH`；其他未分类 → `ERROR_IMAGE_QUALITY`；分类 Top-1 `<0.45` → `RESULT_LOW`；否则 Top-1 与 Top-2 差 `<0.12` → `RESULT_MEDIUM`；其余 → `RESULT_HIGH`。

**需要在产品准入前解决的合同差异：** Yujian 当前 `app/inference_api.py` 的 `LOW_CONFIDENCE_THRESHOLD` 默认是 `0.55`，其运行字段 `low_confidence` 由该值产生；App 观察合同使用 `0.45` 并增加 `0.12` margin。本适配器不改模型、旧质量门或运行配置，只为 UI 映射观察到的 3+2 状态。这一跨仓库阈值映射尚无本仓库的正式产品 Admission，不得将其描述成已统一或生产验证。后续要么由总体决策流程采纳版本化共享合同，要么在准入前停止使用其中一套值；禁止静默覆盖阈值。

## 4. SWOS / RR09 时序与连续性边界

客户端以实际 HIGH 识别响应作为成功揭晓入口。Processing 名义帧为 P0–P7（0、350、950、1200、1550、2050、2600、2800ms）；RR09 为 R0、R0.5、R1、R2、R3、R4、R5（0、100、200、380、550、680、850ms）。实现总时长常量为 3650ms，`R0.5=100ms` 只是静态采样位置，不额外延长时序。推理超过 2800ms 时页面继续等真实结果，然后再启动结果揭晓；不以动画计时制造识别完成。

共享照片阶段复用原始 Blob URL，在 Processing 与 Result DOM 的实测内容矩形之间以单个临时图层变换 200ms；照片和其中鱼体共用一个变换。R0.5 几何在该 200ms 线性过渡的中点。R1 后固定到 Result 区域。现有页面没有向 API 流式发送子阶段事件，所以 Processing 列表仅在服务响应后用返回耗时回填；动态时间轴尚未获得端到端事件流证据。

当前实现刻意不显示 R2–R3 的金光移入文字：SAM ViT-B 轮廓/Mask 尚未过真实照片精度门，检测 BBox 不能冒充鱼体轮廓或字形来源。HIGH 名称在 R4（680ms）才可见；R5 静止。Medium、Low、错误路径不播放成功揭晓。

## 5. Detector 与 SAM 资格证据

- 真实流水线实现已经存在于仓库：Detector 运行时、质量门、标准 Crop 和注册表分类器；本 PR 复用其调用链。
- PR #176 的 UAT 记录含 `Failed to fetch`；本次对部署健康页的只读浏览器访问超时。没有可用的已部署推理响应可以证明 API 从 UI 到生产模型全链路成功。
- SAM 的代码位于实验/调试分割路径，需通过 `SEGMENTATION_CHECKPOINT_PATH/URI` 提供权重；初始化可能 fail-open。未找到已登记的生产 SAM 权重、60–100 张授权真实照片及人工参考 Mask，也没有 IoU / Dice 或边界误差评估。
- 因此 G2（在线真实推理）与 G3（生产 SAM ViT-B 资格）为 `BLOCKED`。不在照片上人工画鱼体轮廓，也不把单帧人工结果冒充模型 Mask。

## 6. 动态高保真下一阶段实施方案

### 推荐架构

保留现有 DOM/CSS 结果与 Processing 页面；以一个 `SWOSMotionController` 和单一 `elapsedMs` 驱动所有光层、照片几何和文字状态。UI 端口至少定义 `RenderFrame(elapsedMs, photo, geometry, opticalConfig, recognitionState)`，相同输入应能确定性重绘。

| 层 | 首选实现 | 资格/降级要求 |
|---|---|---|
| 水光焦散、波纹、外围折射 | WebGL 2 Shader | Chromium 中报告 GPU renderer、帧时间与失效状态；context loss / WebGL 不可用时降级为低密度 Canvas 2D/DOM 焦散，禁用大面积 blur |
| 照片、Result 页面、表单 | 现有 HTML/CSS | 原 DOM 是 3+2 表单和 Hero 的布局权威；不得截图重造 |
| 共享照片转换 | Photo Geometry Engine + 单 DOM/Canvas 图像层 | source/target rect、object-fit 和裁切元数据由同一变换链共享；R0.5 实测插值，不生成背景，不做双图交叉淡入 |
| 鱼体金色 Core / Soft Glow / Halo | 可信 Mask 的 Canvas/SVG 或 GPU mask pass | 三层复用同一可信几何；Mask 缺失、低可信或坐标未校准则关闭三层金光 |
| 金光转移路径 | 基于鱼体锚点的 SVG Path / Canvas | 来源必须为可信轮廓/锚点，目标为冻结文字区域；不得在无来源情况下独立画亮标题 |
| 中文笔画 | Canvas glyph mask | 使用已确认字体逐字测量；文字在 R4 才完整可读；字体加载失败时改为无光静态揭晓 |
| 时间 | Unified Timeline | 唯一 requestAnimationFrame 时钟；支持显式 seek；浏览器性能降级不能改变业务识别状态 |

### 照片几何与数据字段

未来接口必须带稳定照片 ID / SHA-256、EXIF 归一方向、源宽高、Processing 内容框、Result Hero 框、`object-fit` / 裁剪策略、鱼体 BBox/Contour 及可信状态、识别 Session ID、单一 transition elapsed time、gold-trace 状态。照片与鱼体几何必须共用同一个矩阵；不得分别插值。纵图进横向 Hero 时先尝试现有 BBox 安全裁剪，安全裁剪不成立则保留完整同源前景并用低对比同源模糊背景，不拉伸或生成背景。

### 性能、Reduce Motion 与 API 28 迁移边界

- Chromium 目标设备记录 3650ms render、JS long task、WebGL draw/帧时间和内存。预先约定性能预算后，再用真实最低档设备采样；本次没有取得性能结论。
- `prefers-reduced-motion` 应展示同一业务状态的静态终帧，保留用户可手动步进/检查的能力。
- 视觉配置以 JSON 参数导出，时间采样由确定性毫秒驱动，避免刷新率差异造成漂移。
- Android API 28 仅迁移合同与参数：共享照片矩阵、采样时钟、输入 Mask 几何、强度曲线；后端识别/质量门不在 UI 动画时钟内改写。Android renderer 选型、性能和 APK 不属于此 PR。

### 下一阶段工作拆分与验收

1. 总体流程确认 Stage 0/#186/#187 的采纳关系、3+2 版本来源与视觉合同。
2. 获得授权多照片测试集和对应参考轮廓；先通过 detector/mask 可信度、IoU/Dice 与坐标校准门。
3. 把当前同步推理适配端点升级为可取消、带 Session ID 和只读阶段事件的流式/轮询契约；阶段事件来自真实后端工作，不由前端计时伪造。
4. 实现 `PhotoGeometryEngine`、确定性 `UnifiedTimeline` 和单 `RenderFrame` 入口。
5. 实现低密度 SWOS 光学 renderer、同源鱼体三层金光、R0/R0.5/R1 变换与可信几何条件下的 RR09 转译。
6. 加入暂停、逐帧、拖动、0.25×/0.5×/1×、15 帧定位、图层开关、强度滑杆和 JSON 配置导入导出。
7. 以多张授权照片跑 Chromium Desktop/Mobile、Reduce Motion、错误/无鱼/Medium/Low、高延迟/取消/WebGL context loss 测试；产出截图、录屏、动态性能统计。
8. Android API 28 只依据冻结的几何与时间合同开展独立适配，不在本 PR 启动。

## 7. 当前验收状态

| Gate | 当前状态 | 依据 |
|---|---|---|
| G0 产品 Control Plane Admission | `BLOCKED` | 本仓库无已登记 Admission；App #186/#187 尚未采纳；不能互借 App 候选分支 PASS |
| G1 Web 页面 / 导航接线 | `IMPLEMENTED`，待浏览器预览 | 页面注册、侧栏入口和单页真实推理工作区已加入 |
| G2 真实在线模型推理 | `BLOCKED` | 已有推理流水线被代码复用，但最新 PR #176 报告 UAT `Failed to fetch` / Revision SHA 不一致，本次没有真实服务响应 |
| G3 SAM ViT-B 生产资格 | `BLOCKED` | 无生产权重与授权参考 Mask 评测集，无分割质量报告 |
| G4 15 帧时序静态契约 | `CONTRACT_PASS`（代码常量检查） | Processing 8 帧 + RR09 7 帧；总时长 3650ms；R0.5=100ms、R1=200ms |
| G5 结果保存路径 | `IMPLEMENTED`，未做云端写入测试 | 仅显式保存后使用 App Bearer API；没有登录凭据或 GCS 写入验证 |
| G6 Chromium / 多真实照片视觉验收 | `NOT_RUN` | 没有公开许可测试照，本次也未部署 PR preview；无 Chromium 截图/录屏证据 |
| G7 Draft PR CI | 待本地检查后执行 | PR 尚未创建 |
| G8 动态方案可执行性 | `DESIGNED` | 见第 6 节；WebGL/Canvas 性能尚未测，不能声称选型已验证 |

### 明确未运行

- `ANDROID_NOT_RUN`：未改 Android 生产代码、未构建 APK。
- `DYNAMIC_DEMO_NOT_RUN`：没有自动播放 Demo，页面的 CSS 环境光不代表动态原型。
- `LIVE_REAL_PHOTO_INFERENCE_NOT_RUN`：没有将私有或未授权照片发送到线上服务。
- `PUBLIC_PREVIEW_NOT_RUN`：没有覆盖或部署生产 Pages。

正式状态只能为 `PARTIAL`，不得输出 `AI_CATCH_RECOGNITION_WEB_READY_FOR_REVIEW`，直至 Admission、真实部署推理、SAM 资格及浏览器门禁解决。
