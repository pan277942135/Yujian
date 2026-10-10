from app.entry import app
from app.main import templates


def test_fish_asset_batch_import_api_and_page_are_registered():
    paths = app.openapi()["paths"]
    assert "/api/v1/admin/fish/assets/import-batches" in paths
    assert "/api/v1/admin/fish/assets/import-batches/local" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/upload" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/scan" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/execute" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/retry" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/sync-content" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/versions/{version_id}/preview" in paths
    assert "/fish-knowledge/assets/import" in paths
    assert "/api/v1/admin/fish/assets/species/{species_id}" in paths
    assert "/api/v1/admin/fish/assets/species/{species_id}/roles/{asset_role}" in paths
    assert "/api/v1/admin/fish/assets/single-upload" in paths
    assert "/api/v1/admin/fish/assets/versions/{version_id}/preview" in paths
    assert "/api/v1/admin/fish/assets/versions/{version_id}/review" in paths
    assert "/api/v1/admin/fish/assets/versions/{version_id}/activate" in paths
    assert "/api/v1/admin/fish/assets/versions/{version_id}/public-api-check" in paths
    assert "/api/v1/admin/fish/assets/versions/{version_id}/client-check" in paths
    assert "/api/v1/admin/fish/assets/batches/{batch_id}/freeze" in paths
    assert "/api/v1/admin/fish/assets/batches/{batch_id}/freeze-manifest" in paths
    assert "/api/v1/admin/fish/assets/batches/{batch_id}/freeze-manifest.csv" in paths
    detail_schema = app.openapi()["components"]["schemas"]["SpeciesFullDetailOut"]["properties"]
    assert {"cover_hero_image", "cover_assets", "knowledge_assets"} <= set(detail_schema)
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_asset_import.html")
    for marker in (
        "Fish Knowledge Asset Batch Import V1",
        "扫描并预检",
        "选择文件夹",
        "上传文件夹",
        "上传文件夹格式要求（更新和新增）",
        "webkitdirectory",
        "relative_path",
        "Batch Preview",
        "VALID",
        "WARNING",
        "INVALID",
        "所有图片将进入 DRAFT",
        "Retry Failed",
        "同步到鱼鉴内容",
        "history-batch",
        "data-batch-id",
        "closeModal()",
        "closeBtn",
        "closeModalOnBackdrop",
        ".modal[hidden]{display:none!important}",
    ):
        assert marker in source
    templates.env.from_string(source)
    assert "save-stage-qa" in source
    assert "单独保存" in source


def test_fish_knowledge_asset_v13_controls_show_roles_preview_and_freeze_evidence():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_knowledge.html")
    for role in (
        "COVER_LIST", "COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT",
        "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL",
    ):
        assert role in source
    assert "single-upload" in source
    assert "checkerboard" in source
    assert "FROZEN_DRAFT" in source
    assert "upload.progress" in source or "xhr.upload.onprogress" in source
    batch, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_asset_import.html")
    assert "structured_content" in batch
    assert "BLOCKED_CONTENT_MISMATCH" in batch
    assert "BLOCKED_VISUAL_QA" in batch
    assert "manifestCsv" in batch
    assert "base+'.csv'" in batch
    assert "冻结已审核版本" in batch
    templates.env.from_string(source)
    templates.env.from_string(batch)


def test_fish_asset_import_template_has_valid_control_bindings():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_asset_import.html")
    assert "data-batch-id" in source
    assert "history-batch" in source
    history_block = source.split("async function history", 1)[1].split("\n$('historyRows')", 1)[0]
    assert "onclick" not in history_block
