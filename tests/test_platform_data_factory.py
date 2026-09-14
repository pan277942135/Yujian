from __future__ import annotations

from starlette.requests import Request

from app.platform.routes.pages import PLATFORM_PAGES, templates


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []})


def test_data_factory_pages_use_platform_templates_and_empty_states():
    pages = {page.path: page for page in PLATFORM_PAGES}
    assert pages["/platform/data/datasets"].template == "platform/data_datasets.html"
    assert pages["/platform/data/review"].template == "platform/data_review.html"
    assert pages["/platform/data/queue"].template == "platform/data_queue.html"
    for path in ("/platform/data/datasets", "/platform/data/review", "/platform/data/queue"):
        page = pages[path]
        rendered = templates.env.get_template(page.template).render(
            request=_request(path), page=page, page_title=page.title, platform_pages=PLATFORM_PAGES
        )
        assert page.title in rendered
        assert "暂无" in rendered


def test_platform_component_partials_are_present():
    for name in (
        "platform/components/metric_card.html",
        "platform/components/status_tag.html",
    ):
        assert templates.env.get_template(name) is not None
