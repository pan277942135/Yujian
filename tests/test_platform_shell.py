from starlette.requests import Request

from app.entry import app
from app.platform.routes.pages import PLATFORM_PAGES, templates


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []})


def test_platform_shell_registers_all_workspace_pages():
    paths = app.openapi()["paths"]
    expected = {page.path for page in PLATFORM_PAGES}
    assert expected <= set(paths)
    assert all(paths[path]["get"]["responses"]["200"] for path in expected)


def test_platform_shell_templates_compile_and_render_without_legacy_nav_injection():
    for page in PLATFORM_PAGES:
        template = templates.env.get_template(page.template)
        rendered = template.render(request=_request(page.path), page=page, page_title=page.title, platform_pages=PLATFORM_PAGES)
        assert page.title in rendered
        assert 'class="platform-sidebar"' in rendered
        assert 'class="app-nav"' not in rendered
