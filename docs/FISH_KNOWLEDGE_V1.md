# Fish Knowledge Database V1

Fish Knowledge 是面向 YuJian App 的结构化鱼种知识数据层，与训练数据管理、模型 class index 保持解耦。

## 稳定身份

```text
fish_species.id
        =
species_catalog.species_key
```

模型每个 Dataset 的数字 class index 仍只存在于该 Dataset 的 `class_map.json`。图鉴、识别结果和审核数据通过稳定的 `species_key` 关联，扩展鱼种不会改变历史模型的类别映射。

## 数据表

- `fish_species`：名称、别名、分类、学名、简介与发布状态。
- `fish_gallery`：最多 5 张有序轮播图，可登记 HTTPS 图片或上传到 YuJian GCS。
- `fish_profile`：体型、视觉特征、栖息水域、食性、活跃季节。
- `fish_fishing`：水层、季节、饵料、钓法和短说明。
- `fish_video`：介绍、钓法、真实钓获和装备视频。
- `fish_similarity`：相似鱼及结构化短差异说明。
- `fish_ranking`：仅预留表结构，V1 不开放接口。

生产数据库迁移：`schemas/0014_fish_knowledge_v1.sql`。

## App 只读 API

```text
GET /api/v1/fish/species
GET /api/v1/fish/species/{species_id}
GET /api/v1/fish/gallery/{image_id}/media
```

列表仅返回 `ACTIVE` 鱼种，`cover_image` 从 Gallery 的第一张图片动态派生。

详情固定返回：

```json
{
  "species": {},
  "gallery": {"species_id": "grass_carp", "images": []},
  "profile": {},
  "fishing": {},
  "videos": [],
  "similarity": []
}
```

尚未维护的模块返回稳定空数组或 `null` 字段；非法、未知或 `DRAFT` 的 `species_id` 返回 `404`。App 不需要通过缺字段判断数据状态。

这些 GET 接口是公开只读接口。所有 Admin 写接口仍受 Model Factory Console 登录保护。

## Admin API

```text
GET    /api/v1/admin/fish/species
GET    /api/v1/admin/fish/species/{species_id}
POST   /api/v1/admin/fish/species
PATCH  /api/v1/admin/fish/species/{species_id}
PUT    /api/v1/admin/fish/species/{species_id}/profile
PUT    /api/v1/admin/fish/species/{species_id}/fishing

POST   /api/v1/admin/fish/species/{species_id}/gallery
POST   /api/v1/admin/fish/species/{species_id}/gallery/upload
PATCH  /api/v1/admin/fish/species/{species_id}/gallery/{image_id}
DELETE /api/v1/admin/fish/species/{species_id}/gallery/{image_id}

POST   /api/v1/admin/fish/species/{species_id}/videos
PATCH  /api/v1/admin/fish/species/{species_id}/videos/{video_id}
DELETE /api/v1/admin/fish/species/{species_id}/videos/{video_id}

PUT    /api/v1/admin/fish/species/{species_id}/similarity/{similar_species_id}
DELETE /api/v1/admin/fish/species/{species_id}/similarity/{similar_species_id}
```

### 图片上传合同

- Multipart 字段：`file`, `type`, `order`, `title`。
- 格式：JPEG / PNG / WEBP。
- 单图上限：15 MB；像素上限：4000 万。
- 每鱼种最多 5 张，`order` 为 `0..4` 且不可重复。
- 存储路径：`gs://<GCS_BUCKET>/fish_knowledge/<species_id>/gallery/<sha256>.<ext>`。
- 同一图片重试按 SHA-256 幂等返回，不重复写 GCS 或数据库。
- 删除 Gallery 元数据时保留其 GCS 对象，避免不可恢复的资产删除。

## 发布状态

- `DRAFT`：Admin 可维护，App 查询不可见。
- `ACTIVE`：App 列表和详情可见。

Fish Knowledge 的发布状态不自动改变 `species_catalog` 的训练状态。新建鱼种会先进入 Catalog `candidate`；是否进入模型训练仍需按原有审核流程独立确认。
