import shutil
import subprocess
from pathlib import Path

from app.entry import app
from app.main import templates, fish_knowledge_batch_import_page, fish_knowledge_page


def test_fish_knowledge_workspace_route_and_template_are_registered():
    assert "/fish-knowledge" in app.openapi()["paths"]
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_knowledge.html")
    for marker in (
        "Fish Knowledge CMS v1.3",
        "鱼种资产包",
        "列表 Cover Card",
        "五张黑金鱼鉴卡",
        "结构化知识",
        "真实 Gallery",
        "Fishing Video",
        "/api/v1/admin/fish/species",
        "/api/v1/admin/fish/cards/",
        "/api/admin/fish/assets/upload",
        "form.append('species_id'",
        "/api/v1/fish/species/",
    ):
        assert marker in source

    # Compile the source after the unified navigation wrapper has been applied.
    templates.env.from_string(source)


def test_fish_knowledge_nav_is_present_in_shared_template_source():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "overview.html")
    assert 'href="/fish-knowledge"' in source
    assert "鱼鉴内容" in source


def test_fish_knowledge_workspace_javascript_parses():
    node = shutil.which("node")
    if node is None:
        return
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_knowledge.html")
    script = source.split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        [node, "--check"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_v14_fish_workspace_is_the_rendered_route_and_has_independent_sections():
    assert "fish_knowledge_v14.html" in fish_knowledge_page.__code__.co_consts
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_knowledge_v14.html")
    for label in ("鱼种概览", "基本信息", "图片资产", "五张知识卡", "批量导入", "发布中心", "扩展内容", "历史与审计"):
        assert label in source
    assert "fish_knowledge_v14/" in source

    for section in ("basic", "batch-import"):
        partial, _filename, _uptodate = templates.env.loader.get_source(
            templates.env,
            f"fish_knowledge_v14/{section}.html",
        )
        templates.env.from_string(partial)
        assert Path(f"app/static/fish_knowledge_v14/{section}.js").is_file()
        assert partial.strip()
    assert fish_knowledge_batch_import_page.__code__.co_consts


def test_fish_knowledge_upload_ui_binds_url_and_reports_persistence_state():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_knowledge.html")
    for marker in (
        "function applyUploadedAsset",
        "document.getElementById('coverUrl').value=url",
        "cards[index].image_url=url",
        "function updateCardImage",
        "return uploadedUrl",
        "上传中...",
        "图片已上传，但绑定保存失败",
        "上传成功，已保存并回显",
        "error.code=data&&typeof data==='object'?data.error:null",
        "await refreshSelected()",
    ):
        assert marker in source
