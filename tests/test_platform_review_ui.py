from __future__ import annotations

from starlette.requests import Request

from app.platform.routes.pages import PLATFORM_PAGES, templates


def test_review_center_has_three_column_workflow_and_required_actions():
    page = next(item for item in PLATFORM_PAGES if item.path == "/platform/data/review")
    rendered = templates.env.get_template(page.template).render(
        request=Request({"type": "http", "method": "GET", "path": page.path, "query_string": b"", "headers": []}),
        page=page,
        page_title=page.title,
        platform_pages=PLATFORM_PAGES,
    )
    for text in ("筛选", "AI 结果", "确认正确", "修改鱼种", "调整框", "批量确认", "批量删除"):
        assert text in rendered
    assert "platform-sidebar" in rendered
