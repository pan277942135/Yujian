from __future__ import annotations

from starlette.requests import Request

from app.entry import app
from app.main import templates as legacy_templates
from app.platform.routes.pages import PLATFORM_PAGES, templates as platform_templates
from app.unified_nav import CANONICAL_NAV


def test_legacy_sidebar_is_the_legacy_nav_source():
    source = legacy_templates.env.loader.delegate.get_source(legacy_templates.env, "legacy/sidebar.html")[0]
    assert 'class="app-nav"' in source
    assert "旧版总览" in source
    assert 'class="platform-sidebar"' not in source
    assert 'class="app-nav"' in CANONICAL_NAV


def test_legacy_and_platform_renderers_do_not_cross_inject_navigation():
    request = Request({"type": "http", "method": "GET", "path": "/legacy", "query_string": b"", "headers": []})
    legacy = legacy_templates.env.get_template("overview.html").render(request=request)
    platform_page = next(page for page in PLATFORM_PAGES if page.path == "/platform")
    platform = platform_templates.env.get_template(platform_page.template).render(
        request=request, page=platform_page, page_title=platform_page.title, platform_pages=PLATFORM_PAGES
    )
    assert 'class="app-nav"' in legacy
    assert 'class="platform-sidebar"' in platform
    assert 'class="app-nav"' not in platform
