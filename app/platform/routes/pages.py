from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request


router = APIRouter(tags=["platform-pages"])
templates = Jinja2Templates(directory="app/templates")


@dataclass(frozen=True)
class PlatformPage:
    path: str
    template: str
    title: str
    section: str
    description: str
    api_path: str | None = None


PLATFORM_PAGES = (
    PlatformPage("/platform", "platform/dashboard.html", "首页总览", "总览", "查看数据、模型、流水线和鱼体资产的生产状态。", "/api/platform/dashboard"),
    PlatformPage("/platform/data/import", "platform/data_import.html", "导入采集数据", "AI 数据工厂", "复用现有 Batch 上传流程导入真实鱼获采集数据。", "/batches/upload"),
    PlatformPage("/platform/data/datasets", "platform/data_datasets.html", "数据集管理", "AI 数据工厂", "用数据集视角管理上传、AI 清洗和训练准备状态。", "/api/platform/datasets"),
    PlatformPage("/platform/data/review", "platform/data_review.html", "数据审核中心", "AI 数据工厂", "集中处理低置信、BBox 异常和质量异常样本。", "/api/platform/review/items"),
    PlatformPage("/platform/model/training", "platform/model_training.html", "模型训练", "模型工厂", "从已冻结数据集创建训练任务并追踪结果。", "/api/platform/training/jobs"),
    PlatformPage("/platform/model/registry", "platform/model_registry.html", "模型仓库", "模型工厂", "查看模型版本、指标和发布状态。", "/api/platform/models"),
    PlatformPage("/platform/model/evaluation", "platform/model_evaluation.html", "模型评估", "模型工厂", "查看指标、混淆关系和错误案例。", "/api/platform/models/{model_id}/evaluation"),
    PlatformPage("/platform/pipeline", "platform/pipeline.html", "智能流水线", "智能流水线", "追踪一张鱼照片如何经过 AI 节点生成资产。", "/api/platform/pipelines"),
    PlatformPage("/platform/lab/fish-portrait", "platform/lab/fish_portrait.html", "Fish Portrait POC", "智能流水线", "基于真实钓获照片和标准鱼体资产，验证 AI 鱼体数字化生成效果。", "/api/platform/portrait/jobs"),
    PlatformPage("/platform/assets", "platform/assets.html", "数字资产工厂", "数字资产工厂", "查看原图、Mask、透明鱼和 Sticker 资产。", "/api/platform/assets"),
    PlatformPage("/platform/knowledge", "platform/knowledge.html", "鱼类知识库", "鱼类知识库", "复用现有 Fish Knowledge CMS 数据。", "/api/platform/knowledge"),
    PlatformPage("/platform/habitat", "platform/habitat.html", "渔境生态", "渔境生态", "管理鱼缸、鱼塘、湖湾等渔境生态配置。", "/api/platform/habitat"),
    PlatformPage("/platform/system/tasks", "platform/system_tasks.html", "任务中心", "系统管理", "统一查看训练、流水线和资产任务。", "/api/platform/system/tasks"),
    PlatformPage("/platform/system/logs", "platform/system_logs.html", "日志中心", "系统管理", "查看 Platform 操作、警告和失败记录。", "/api/platform/system/logs"),
)


def _render(request: Request, page: PlatformPage) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name=page.template,
        context={
            "page": page,
            "page_title": page.title,
            "platform_pages": PLATFORM_PAGES,
        },
    )


def _page_route(page: PlatformPage):
    def handler(request: Request) -> HTMLResponse:
        return _render(request, page)

    handler.__name__ = f"platform_{page.title.replace(' ', '_')}_page"
    handler.__doc__ = page.description
    return handler


for _page in PLATFORM_PAGES:
    router.add_api_route(
        _page.path,
        _page_route(_page),
        methods=["GET"],
        response_class=HTMLResponse,
        name=f"platform_{_page.title.replace(' ', '_')}",
    )


@router.get("/platform/data/datasets/{dataset_id}", response_class=HTMLResponse, name="platform_dataset_detail")
def platform_dataset_detail_page(request: Request, dataset_id: str) -> HTMLResponse:
    """Render one DatasetVersion without adding it to the sidebar menu."""

    page = PlatformPage(
        "/platform/data/datasets/{dataset_id}",
        "platform/data_dataset_detail.html",
        "数据集详情",
        "AI 数据工厂",
        "查看 DatasetVersion 的规模、来源、版本链和 AI 清洗情况。",
        f"/api/platform/datasets/{dataset_id}",
    )
    return _render(request, page)


@router.get("/platform/data/queue", include_in_schema=False)
def platform_queue_compatibility_redirect() -> RedirectResponse:
    """Keep old bookmarks working while the queue is part of Review Center."""

    return RedirectResponse(url="/platform/data/review", status_code=307)


__all__ = ["PLATFORM_PAGES", "router", "templates"]
