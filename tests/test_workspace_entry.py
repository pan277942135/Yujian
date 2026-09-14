from app.entry import app
from app.main import templates


def test_workspace_entry_routes_are_registered_without_removing_legacy_overview():
    paths = app.openapi()["paths"]
    assert "/" in paths
    assert "/legacy" in paths
    assert paths["/"]["get"]["operationId"].startswith("workspace_page")
    assert paths["/legacy"]["get"]["operationId"].startswith("legacy_overview_page")


def test_workspace_template_offers_both_workspaces():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "workspace.html")
    assert "旧版模型工作台" in source
    assert "AI 智能生产平台" in source
    assert 'href="/legacy"' in source
    assert 'href="/platform"' in source


def test_legacy_navigation_points_to_legacy_overview():
    source, _filename, _uptodate = templates.env.loader.get_source(templates.env, "overview.html")
    assert 'href="/legacy"' in source
    assert "旧版总览" in source
