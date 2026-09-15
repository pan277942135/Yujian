from __future__ import annotations

from starlette.requests import Request

from app.platform.routes.pages import PLATFORM_PAGES, templates


def test_model_factory_pages_expose_simple_and_advanced_workflows():
    pages = {page.path: page for page in PLATFORM_PAGES}
    assert pages["/platform/model/training"].template == "platform/model_training.html"
    assert pages["/platform/model/registry"].template == "platform/model_registry.html"
    assert pages["/platform/model/evaluation"].template == "platform/model_evaluation.html"
    for path, markers in {
        "/platform/model/training": ("开始训练", "高级参数", "Epoch", "Batch Size", "发布模型", "fish_classifier_v0_2.tflite"),
        "/platform/model/registry": ("模型版本", "发布状态", "不展示内部存储路径"),
        "/platform/model/evaluation": ("Accuracy", "混淆矩阵", "错误案例", "模型智能分析", "Hard Case", "生成采集Batch"),
    }.items():
        page = pages[path]
        rendered = templates.env.get_template(page.template).render(
            request=Request({"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []}),
            page=page,
            page_title=page.title,
            platform_pages=PLATFORM_PAGES,
        )
        for marker in markers:
            assert marker in rendered
