from __future__ import annotations

from starlette.requests import Request

from app.platform.routes.pages import PLATFORM_PAGES, templates


def test_asset_knowledge_habitat_and_system_pages_are_real_platform_views():
    pages = {page.path: page for page in PLATFORM_PAGES}
    expected = {
        "/platform/assets": ("原图", "Mask", "透明鱼", "Sticker"),
        "/platform/knowledge": ("基础信息", "识别特征", "钓法", "相似鱼"),
        "/platform/habitat": ("场景等级", "容量", "生态规则"),
        "/platform/system/tasks": ("训练任务", "Pipeline 任务", "资产任务"),
        "/platform/system/logs": ("运行记录", "失败", "消息"),
    }
    for path, markers in expected.items():
        page = pages[path]
        rendered = templates.env.get_template(page.template).render(
            request=Request({"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []}),
            page=page,
            page_title=page.title,
            platform_pages=PLATFORM_PAGES,
        )
        for marker in markers:
            assert marker in rendered
