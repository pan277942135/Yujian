from app.entry import app
from app.main import templates


def test_fish_asset_batch_import_api_and_page_are_registered():
    paths = app.openapi()["paths"]
    assert "/api/v1/admin/fish/assets/import-batches" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/scan" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/execute" in paths
    assert "/api/v1/admin/fish/assets/import-batches/{batch_id}/retry" in paths
    assert "/fish-knowledge/assets/import" in paths
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_asset_import.html")
    for marker in (
        "Fish Knowledge Asset Batch Import V1",
        "扫描并预检",
        "Batch Preview",
        "VALID",
        "WARNING",
        "INVALID",
        "所有图片将进入 DRAFT",
        "Retry Failed",
        "history-batch",
        "data-batch-id",
        "closeModal()",
    ):
        assert marker in source
    templates.env.from_string(source)


def test_fish_asset_import_template_has_valid_control_bindings():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_asset_import.html")
    assert "data-batch-id" in source
    assert "history-batch" in source
    assert "onclick=\\"loadBatch(\\\\'" not in source
