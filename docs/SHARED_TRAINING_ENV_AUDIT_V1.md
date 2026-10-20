# DET_FISH 与鱼种分类训练环境统一审计 V1

**Task ID:** `YUJIAN_SHARED_TRAINING_ENV_AUDIT_V1`  
**审计日期:** 2026-10-20  
**仓库:** `pan277942135/Yujian`  
**审计基线:** PR #166 初始 HEAD `190038c2622965597af82d1debd85f316c241e59`  
**云端状态:** `NOT_VERIFIED`

## 1. 范围与证据规则

本报告核对仓库中的真实训练入口、容器、依赖、数据路径、评估、模型产物和部署脚本。Cloud Run 服务、GCP 项目、Revision、Job 实际配置、GPU 配额、计费、权限和 GCS 对象均由项目负责人在 GCP 核查；仓库声明不等于云端已经部署或可运行。

审计未启动 Cloud Run Job、GPU 训练、部署或计费资源。PR #166 在审计开始时为 Draft / Open / Unmerged；其基线 CI 运行 `39403380830` 的 36 项检查通过。该 CI 只证明仓库测试和构建步骤通过，不证明 GCP 数据、训练任务、真实上传推理或线上服务。

## 2. DET_FISH 检测器

| 项目 | 仓库实现 |
|---|---|
| 入口 | `trainer/train_detector.py:main`，Cloud Run 容器默认执行 `python -m trainer.train_detector` |
| 网络与数据 | `trainer/yolox_fish_exp.py`：YOLOX-Nano 单类 fish、416×416、YOLOX COCO 数据集格式；v0.2 有 90°/180°/270° 训练增强 |
| 容器 | `Dockerfile.detector`：`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`，torch 2.7.1，torchvision 0.22.1 cu128 |
| 依赖 | `requirements-detector.txt`：NumPy 固定 1.26.4；OpenCV-headless、pycocotools、ONNX、ONNX Runtime 等多数只有版本范围 |
| 外部代码/权重 | Dockerfile 浅克隆 YOLOX `0.3.0` 标签，没有固定 Git commit；`PRETRAIN_URL` 下载 YOLOX 预训练权重，没有 SHA-256 校验 |
| 仓库 Job 配置 | `scripts/deploy_detector_training_gcp.sh`：`yujian-detector-trainer`，默认区域 `asia-southeast1`，4 vCPU / 16 GiB / 1 张 NVIDIA L4，1 task、0 retry、3600 秒 |
| 数据位置 | `detector-datasets/{version}/`；运行时 bucket 必须配置 `GCS_BUCKET`。模型在 `models/{version}/` |
| 产物 | `best_ckpt.pth`、ONNX、`detector_metadata.json`、`recognition_pipeline_v1.json` |
| 当前评估 | `evaluate_checkpoint()` 使用 YOLOX COCO evaluator 计算验证集 AP50 与 AP50:95。虽定义 `instances_test2017.json`，训练代码没有显式运行独立 test evaluator；没有输出独立 test Precision / Recall |

L4 资源数值仅来自仓库部署脚本，实际 Job、配额、区域可用性、磁盘和预算均为 `NOT_VERIFIED`。训练器把数据、检查点和下载的预训练权重写入 `/tmp`；Job 没有为本地临时磁盘空间声明容量或做数据规模预检。训练 Job 一小时超时也尚无实测训练时长证据。

### DET_FISH 数据门禁与实际可用性

- `data/detector/DET_DS_v0.2/README.md` 明确为 `INTAKE_OPEN / NOT TRAINABLE`，只含采集要求，没有图片或标签。
- 仓库及当前工作区未发现 v0.2 的 `bootstrap_report.json`、`bootstrap_manifest.json`、冻结归档或 COCO 图片/标注。此结果只说明本次可读文件中未发现；GCS 是否有对象为 `NOT_VERIFIED`。
- `trainer/build_detector_dataset.py` 是旧 v0.1 构建路径，标签来自 `FishPresenceResult/google-vision-presence-v0.4` 候选框；不能把它当成人工 accepted BBox 数据源。
- `trainer/build_reviewed_datasets.py` 能读取人工 `accepted_bbox`，但检测输出为 YOLO 文本标签和 CSV Manifest；它没有生成 YOLOX 要求的 `annotations/instances_{train,val,test}2017.json`，也没有覆盖人工无鱼负样本、来源组拆分及冻结包登记。
- 因此仓库无法证明“6000+ 人工确认 BBox”进入了 DET_FISH 训练；Accepted Pool 查询计数也不证明它进入已冻结、实际使用的检测训练集。数量和转换链路在 GCP/DB 未核实前均为 `NOT_VERIFIED`。
- 本次增加的 `trainer/detector_dataset_gate.py` 同时核对逐图人工记录、哈希、完整场景标志、正负样本、留出集、来源组切分和数据包内 COCO 与审核框一致性。控制台与训练 Worker 共用该门禁；目前没有符合该合同的真实 v0.2 包，故实际启动仍应被阻断。

## 3. 鱼种分类器

| 项目 | 仓库实现 |
|---|---|
| 入口 | `trainer/train_classifier.py`；通过 `DATASET_VERSION`、`MODEL_VERSION`、`MODEL_FAMILY`、训练参数环境变量运行 |
| 训练数据 | DB `DatasetVersion` 的 GCS `manifest_uri` 与 `class_map_uri`；Crop 路径要求 `CROP_CLASSIFIER_V1`、`input_type=crop_image`、可信人工物种标签及 accepted bbox 验证 |
| 网络 | Torchvision `mobilenet_v3_small`，使用 ImageNet DEFAULT 权重初始化 |
| 损失/优化 | 带类别权重与 label smoothing 的 CrossEntropyLoss；AdamW；以验证 Macro F1 做 early stopping |
| 容器 | `Dockerfile.trainer`：`python:3.11-slim`、CPU-only torch 2.7.1、torchvision 0.22.1 |
| 依赖 | `requirements-training.txt`：NumPy `>=2,<3`，Pillow、SQLAlchemy、psycopg、GCS、FastAPI 等范围依赖 |
| 仓库 Job 配置 | `scripts/deploy_training_gcp.sh`：`yujian-classifier-trainer`，默认区域 `asia-east1`，2 vCPU / 4 GiB、1 task、0 retry、3600 秒；未配置 GPU |
| 产物/登记 | `models/{version}/model_state_dict.pt`、`model_torchscript.pt`、`metrics.json`、class map、评估材料；写入 `TrainingRun`、`ModelVersion`、`Evaluation` |
| 指标 | 训练代码实际输出 Top-1 accuracy、Top-3 accuracy、Macro Precision、Macro Recall、Macro F1、逐类统计与独立 test metrics（前提是 test split 确实独立） |

用户要求的输入是可信鱼种标签的鱼体 Crop；仓库存在 `build_reviewed_datasets.py` 及 `crop_dataset_validator.py` 支持该数据路径。当前实际数据库、冻结 Crop Manifest、class map、Crop 图片、划分和训练记录未从本地仓库取得；远端 GCS/DB 状态均为 `NOT_VERIFIED`。

### 分类器数据划分风险

`build_reviewed_datasets.py` 默认用单个 `image_id` 哈希切 Train / Validation / Test，而不按 `source_batch` 或来源组分组。Crop Manifest 保留 `source_batch`、`source_image_id`，但 `crop_dataset_validator.py` 未执行跨集合来源组泄漏核验。另，`trainer/train_classifier.py:train_model()` 在 Validation 或 Test 为空时分别回退到 Test / Train 或 Validation / Train，可能把非独立数据报告为验证或测试结果。启动分类器前需要核对真实 Manifest；应补充按源图/来源组去重与划分检查，并在报告中拒绝将回退结果称作独立测试。

## 4. 可共享项与必须隔离项

| 可共享的基础能力 | 必须保持隔离的项目 |
|---|---|
| Python 3.11 运行时基线、通用 GCS 命名与访问库、镜像构建流程、日志字段、作业状态字段、模型工件描述协议 | DET_FISH GPU/CUDA 镜像与分类器 CPU 镜像；训练入口和参数；依赖文件/锁文件；数据集版本、Manifest、划分；评估指标和验收门禁；模型命名空间与发布流程 |
| 仓库中 `TrainingRun`、`ModelVersion`、`Evaluation` 数据模型可作为统一登记方向 | 当前分类器实际写入上述 DB 登记；DET_FISH 目前只写 GCS metadata/failure artifact，并由 Cloud Run Execution 提供状态，没有写统一 DB Registry |
| 两套 Job 都以 GCS 保存数据或产物 | 当前两个 deploy 脚本指定的 Artifact Registry 仓库与区域不同，必须分别核实；共享 bucket 只是脚本/环境配置，不证明线上对象或权限实际共享 |

容器数值依赖不可直接统一：分类器要求 NumPy 2.x，YOLOX 检测器固定 NumPy 1.26.4；分类器安装 CPU torch，检测器依赖 CUDA 12.8 runtime。强行把二者塞进同一训练镜像会引入兼容和资源风险。推荐共用构建约定、基础 Python 版本、身份/日志字段与工件注册协议，保留两个镜像、两个 Job、两个锁定依赖集和两套评估。

目前 `requirements-training.txt`、`requirements-detector.txt` 均不是完整锁文件；torch/torchvision 是固定版本，但其他传递依赖浮动。YOLOX 仅按标签浅克隆，预训练权重无内容哈希。正式训练前应为两镜像分别生成可复现的完整锁文件，并固定 YOLOX commit、预训练权重 SHA 和容器镜像 digest。不要共用同一份 NumPy / CUDA 锁文件。

## 5. 仓库中需要补齐的文件与配置

1. **DET_DS_v0.2 数据**：经授权的完整场景图、审核框和人工确认无鱼图；逐图 SHA、来源 URI/许可、reviewer/date、source group、拆分、EXIF 校正尺寸、hard-case 标签。
2. **DET_DS_v0.2 冻结工件**：仓库 README 所列清单、与 Worker/Console 门禁一致的 JSON Manifest、COCO train/val/test 标注、冻结归档、报告、Manifest SHA 和归档 SHA；当前缺少“人工审核记录 → YOLOX COCO 包”的构建/转换入口。
3. **独立测试评估**：Detector 的 COCO test AP50/AP50:95、Precision、Recall 与无鱼误检率；固定检测阈值和评估合同。当前仅明确计算 val AP。
4. **分类器分组拆分门禁**：以来源图/来源批次分组，检查重复 SHA 与跨 split；空 Validation/Test 必须阻止训练或明确标记为非独立结果。
5. **依赖锁定**：两个任务各自的完整 Python 依赖锁定文件；固定 YOLOX Git commit 和预训练权重哈希。
6. **统一训练登记**：Detector Job ID、数据集 SHA、代码/镜像 SHA、ONNX SHA、指标和失败状态尚未写入共享训练登记。Classifier 当前 DB 登记可复用，但 DET_DS_v0.2 还没有可核验的注册记录。
7. **实际资源与访问**：真实 Job 镜像/命令/服务账号、实际 GCS bucket/prefix、Artifact Registry、Cloud SQL/secret、Cloud Run 权限及 GPU 配额由项目负责人核查，均 `NOT_VERIFIED`。

## 6. 项目负责人 GCP 核查清单

请在确认的目标项目中核对并记录实际值。仓库默认值不能作为当前线上事实：

- 当前项目 ID、Model Factory Cloud Run 服务的区域、Revision、镜像 digest、代码来源和服务配置。
- Classifier Job `yujian-classifier-trainer`（脚本默认 `asia-east1`）及 Detector Job `yujian-detector-trainer`（脚本默认 `asia-southeast1`）是否存在；实际镜像 digest、入口、环境变量、CPU、内存、GPU、超时、重试和服务账号。
- Cloud Run GPU/L4 区域支持与当前配额；GPU Job 是否启用、调度与运行权限；由负责人确认当前计费状态和是否允许未来付费训练。本报告没有启动或请求启动任何云资源。
- 实际 GCS bucket 名称、DET_DS_v0.2 / DET_FISH_v0.2 对象列表、对象 generation/metadata、下载权限与校验 SHA；分类器冻结 Manifest 与 class map 的对象路径及实际可读性。
- Artifact Registry `yujian-training` 与 `yujian-detector-training` 的区域、镜像 digest、保留策略。
- Cloud SQL `yujian-registry`、数据库 schema、`TrainingRun`/`ModelVersion`/`Evaluation` 可写权限；Detector 运行身份是否有适当且最小化的写入权限。
- Secret Manager DB 密钥、Cloud Logging/Monitoring、失败诊断对象可读写权限与日志保留策略。
- 两个 Job 临时盘空间和一小时执行超时是否足以容纳实际训练数据、检查点、ONNX 导出和评估。

## 7. 推荐目标架构

1. 共用受控容器构建与发布约定（Python 3.11、非 root、可复现构建、镜像 digest、通用 GCS/日志字段），不强行共用 CUDA 与 NumPy 运行栈。
2. 维持 `classifier-trainer` CPU 镜像/Job 与 `detector-trainer` CUDA L4 镜像/Job 独立；每个有独立依赖锁、参数 Schema、数据版本和启动门禁。
3. 用统一工件清单记录 `task_type`、run ID、dataset version + SHA、git SHA、image digest、seed/参数、真实指标、产物 SHA、状态与时间。Classifier 继续现有 DB 登记；Detector 在确认 v0.2 DatasetVersion/Registry 合同后再接入，不能只从 GCS 文件名推断状态。
4. Detector 用人工 accepted BBox 完成 COCO 适配和逐样本冻结验证；Classifier 使用鱼体 Crop + 可信 species class map。两者的样本 ID、来源组和测试集保持隔离。Case A、Case B、04-181200 等困难回归样本留在训练集之外。
5. 训练完成只产生 candidate artifact。两个模型分别运行自己的独立评估与发布门禁；本次不自动切换生产模型。

## 8. 本次代码审计后的状态

- 两份训练部署脚本已改为必须显式提供 `PROJECT_ID` 或 `GOOGLE_CLOUD_PROJECT`，不再静默回退到历史项目 ID；未执行部署脚本。Detector Worker 与部署脚本默认目标已改为 `DET_DS_v0.2` / `DET_FISH_v0.2`，并在缺少符合门禁的数据时失败关闭。
- DET_FISH Console 使用实际 ONNX Runtime loader 校验模型是否可选；训练按钮合并数据集、Job 配置、活动任务和 v0.2 目标工件覆盖检查。
- Detector Console 与 GPU Worker 共用逐图 v0.2 门禁；Worker 在下载/解压后额外对比包内 Manifest、COCO 图片清单及 COCO 框与人工审核框。没有通过门禁的 v0.2 数据不会进入模型训练。
- 这些检查属于仓库代码，不验证线上配置或远端数据。真实训练、真实模型上传推理、部署和线上截图尚未完成。
