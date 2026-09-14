from __future__ import annotations

from starlette.requests import Request

from app.platform.routes.pages import PLATFORM_PAGES, templates


def test_pipeline_center_renders_trace_stages_and_failure_area():
    page = next(item for item in PLATFORM_PAGES if item.path == "/platform/pipeline")
    rendered = templates.env.get_template(page.template).render(
        request=Request({"type": "http", "method": "GET", "path": page.path, "query_string": b"", "headers": []}),
        page=page,
        page_title=page.title,
        platform_pages=PLATFORM_PAGES,
    )
    for marker in ("Detector", "Classifier", "SAM", "Completion", "Asset", "失败"):
        assert marker in rendered
