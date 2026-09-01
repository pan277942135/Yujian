from app.entry import app
from app.main import templates


def test_fish_knowledge_workspace_route_and_template_are_registered():
    assert "/fish-knowledge" in app.openapi()["paths"]
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "fish_knowledge.html")
    for marker in (
        "鱼鉴内容工作台",
        "鱼种资产包",
        "列表 Cover Card",
        "五张黑金鱼鉴卡",
        "结构化知识",
        "真实 Gallery",
        "Fishing Video",
        "/api/v1/admin/fish/species",
        "/api/v1/admin/fish/cards/",
        "/api/v1/fish/species/",
    ):
        assert marker in source

    # Compile the source after the unified navigation wrapper has been applied.
    templates.env.from_string(source)


def test_fish_knowledge_nav_is_present_in_shared_template_source():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "overview.html")
    assert 'href="/fish-knowledge"' in source
    assert "鱼鉴内容" in source
