from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.image_studio_worker_client import IMAGE_STUDIO_MODE, MAX_REFERENCES
from app.platform.models import ImageStudioRun
from fastapi import HTTPException

import app.platform.routes.image_studio as image_studio_route
from app.platform.routes.image_studio import STORAGE_TYPE, _mask_composite, _require_worker_ready
from app.platform.services.image_studio_prompt import compile_image_studio_prompt


def _png(size=(4, 4), color=(0, 0, 0), mode="RGB") -> bytes:
    image = Image.new(mode, size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_image_studio_contract_is_separate_from_fish_lab():
    assert STORAGE_TYPE == "IMAGE_STUDIO_V1"
    assert ImageStudioRun.__tablename__ == "image_studio_run"
    assert IMAGE_STUDIO_MODE == "image_studio_v1"
    assert MAX_REFERENCES == 2


def test_identity_lock_compiler_assigns_reference_authority():
    compiled = compile_image_studio_prompt(
        "Change the outfit to a dark coat.",
        mode="IDENTITY_LOCK",
        preservation="MAX",
        reference_roles=["IDENTITY", "OUTFIT"],
    )
    assert compiled.mode == "IDENTITY_LOCK"
    assert compiled.preservation == "MAX"
    assert compiled.reference_roles == ("IDENTITY", "OUTFIT")
    assert "Do not blend, average" in compiled.prompt
    assert "Picture 2 role = IDENTITY" in compiled.prompt
    assert "Picture 3 role = OUTFIT" in compiled.prompt
    assert "face averaging" in compiled.negative_prompt


def test_prompt_compiler_rejects_more_than_two_references():
    with pytest.raises(ValueError, match="at most two"):
        compile_image_studio_prompt(
            "test",
            reference_roles=["IDENTITY", "OUTFIT", "SCENE"],
        )


def test_mask_composite_restores_base_pixels_outside_mask():
    base = Image.new("RGB", (4, 4), (10, 20, 30))
    generated = Image.new("RGB", (4, 4), (200, 210, 220))
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)

    def encode(image):
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    result = Image.open(
        io.BytesIO(_mask_composite(encode(base), encode(generated), encode(mask)))
    ).convert("RGB")

    assert result.getpixel((0, 0)) == (10, 20, 30)
    assert result.getpixel((3, 3)) == (10, 20, 30)
    assert result.getpixel((1, 1)) == (200, 210, 220)


def test_worker_source_keeps_legacy_mode_and_adds_multi_reference_inputs():
    worker = (
        Path(__file__).resolve().parents[1]
        / "workers"
        / "fish-qwen-refine-worker"
        / "worker.py"
    ).read_text(encoding="utf-8")

    assert 'QWEN_MODE = "fish_preserve_refine_qwen_v1"' in worker
    assert 'IMAGE_STUDIO_MODE = "image_studio_v1"' in worker
    assert "SUPPORTED_MODES = {QWEN_MODE, IMAGE_STUDIO_MODE}" in worker
    assert "references: list[UploadFile] | None" in worker
    assert 'f"image{offset}"' in worker
    assert '"reference_count": len(reference_inputs)' in worker


def test_image_studio_uses_independent_menu_and_storage_namespace():
    sidebar = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "sidebar.html"
    ).read_text(encoding="utf-8")
    route = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "platform"
        / "routes"
        / "image_studio.py"
    ).read_text(encoding="utf-8")
    pages = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "platform"
        / "routes"
        / "pages.py"
    ).read_text(encoding="utf-8")

    assert '<div class="nav-group-title">Image Studio</div>' in sidebar
    assert 'href="/platform/image-studio"' in sidebar
    assert 'image_studio/v1/runs/' in route
    assert 'select(ImageStudioRun)' in route
    assert 'PipelineRun' not in route
    assert 'PlatformPage("/platform/image-studio"' in pages


def test_long_qwen_calls_do_not_block_fastapi_event_loop():
    root = Path(__file__).resolve().parents[1]
    studio = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    queue = (root / "app" / "services" / "image_studio_jobs.py").read_text(encoding="utf-8")
    qwen_lab = (root / "app" / "platform" / "routes" / "qwen_image_edit_lab.py").read_text(encoding="utf-8")

    assert 'status="QUEUED"' in studio
    assert "enqueue_image_studio_queue()" in studio
    assert "invoke_image_studio_worker(" in studio
    assert "ThreadPoolExecutor(max_workers=1" in queue
    assert "pg_try_advisory_lock" in queue
    assert "await run_in_threadpool(\n            invoke_qwen_refine_worker," in qwen_lab


def test_worker_readiness_gate_blocks_warmup_before_run_creation(monkeypatch):
    monkeypatch.setattr(
        image_studio_route,
        "check_qwen_refine_worker",
        lambda: {
            "health": {
                "status": "loading",
                "model_loaded": False,
            }
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        _require_worker_ready()
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "QWEN_WORKER_NOT_READY"
    assert exc_info.value.detail["status"] == "loading"
    assert exc_info.value.detail["model_loaded"] is False


def test_worker_readiness_gate_accepts_ready_loaded_model(monkeypatch):
    health = {"status": "ready", "model_loaded": True, "model": "Qwen-Image-Edit-2511"}
    monkeypatch.setattr(
        image_studio_route,
        "check_qwen_refine_worker",
        lambda: {"health": health},
    )
    assert _require_worker_ready() == health


def test_image_studio_ui_disables_generate_until_gpu_ready():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio.html"
    ).read_text(encoding="utf-8")

    assert 'id="studioGenerate" class="studio-primary" type="button" disabled' in template
    assert 'id="studioGpuStart"' in template
    assert '"/api/qwen-lab/gpu/status?studio_ts="' in template
    assert '"/api/qwen-lab/gpu/start"' in template
    assert '(state === "READY" || state === "BUSY") && payload?.model_loaded === true' in template
    assert '"BUSY · GPU 正在生成；仍可继续提交"' in template
    assert '"LOADING · Qwen 尚未 Ready，生成未提交"' in template
    assert '"/api/image-studio/v1/queue?ts="' in template



def test_image_studio_common_presets_are_primary_ui():
    root = Path(__file__).resolve().parents[1]
    template = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio.html"
    ).read_text(encoding="utf-8")

    for preset in ["HEAD_SWAP", "CHARACTER_FUSION", "HEAD_SCENE", "OUTFIT", "HD"]:
        assert f'data-preset="{preset}"' in template
    assert "角色换头" in template
    assert "角色融合" in template
    assert "换头 + 换背景" in template
    assert "只换穿搭" in template
    assert "高清增强" in template
    assert "<summary>高级设置</summary>" in template


def test_qwen_runtime_has_no_automatic_vm_shutdown_policy():
    root = Path(__file__).resolve().parents[1]
    uat = (root / "scripts" / "qwen_gpu_manual_uat.sh").read_text(encoding="utf-8")
    boot = (root / "scripts" / "deploy_qwen_worker_boot.sh").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "uat-deploy.yml").read_text(encoding="utf-8")

    assert 'gcloud compute instances stop "$GPU_INSTANCE"' not in uat
    assert 'gcloud compute instances stop "$GPU_INSTANCE"' not in boot
    assert 'gcloud compute instances stop "$GPU_INSTANCE"' not in workflow
    assert "FINAL_STOP_JSON" not in uat
    assert "SECOND_STOP_JSON" not in uat
    assert "expected RUNNING" in uat
    assert "Temporary GPU cleanup after failed Qwen UAT" not in workflow



def test_image_studio_durable_fifo_queue_contract():
    root = Path(__file__).resolve().parents[1]
    route = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    queue = (root / "app" / "services" / "image_studio_jobs.py").read_text(encoding="utf-8")
    entry = (root / "app" / "entry.py").read_text(encoding="utf-8")
    gpu = (root / "app" / "platform" / "routes" / "qwen_gpu.py").read_text(encoding="utf-8")

    assert '"queue_policy": "FIFO_SINGLE_L4"' in route
    assert '@router.get("/queue")' in route
    assert '"queue_position"' in route
    assert 'run.status = "QUEUED"' in route
    assert "ThreadPoolExecutor(max_workers=1" in queue
    assert ".order_by(ImageStudioRun.created_at.asc(), ImageStudioRun.run_id.asc())" in queue
    assert "pg_try_advisory_lock" in queue
    assert "IMAGE_STUDIO_PROCESSING_LEASE_SECONDS" in queue
    assert "recover_pending_image_studio_jobs()" in entry
    assert 'ImageStudioRun.status.in_(["QUEUED", "RUNNING"])' in gpu


def test_image_studio_queue_keeps_cloud_run_background_cpu_active():
    root = Path(__file__).resolve().parents[1]
    deploy = (root / "scripts" / "deploy_console_runtime.sh").read_text(encoding="utf-8")

    assert "--min 1" in deploy
    assert "--no-cpu-throttling" in deploy


def test_image_studio_runtime_uat_polls_queued_job_to_success():
    root = Path(__file__).resolve().parents[1]
    uat = (root / "scripts" / "qwen_gpu_manual_uat.sh").read_text(encoding="utf-8")

    assert 'payload.get("status") in {"QUEUED", "RUNNING", "SUCCESS"}' in uat
    assert 'queue.get("concurrency") == 1' in uat
    assert 'queue.get("policy") == "FIFO_SINGLE_L4"' in uat
    assert 'IMAGE_STUDIO_FINAL_STATUS' in uat
    assert 'test "$IMAGE_STUDIO_FINAL_STATUS" = "SUCCESS"' in uat



def test_image_studio_workbench_has_no_inline_result_gallery():
    root = Path(__file__).resolve().parents[1]
    template = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio.html"
    ).read_text(encoding="utf-8")

    assert ">Result<" not in template
    assert "studioResult" not in template
    assert "studioHistory" not in template
    assert "studioUseAsBase" not in template
    assert "/platform/image-studio/tasks" in template
    assert "refreshCurrentRun" in template


def test_image_studio_task_list_menu_pages_and_download_contract():
    root = Path(__file__).resolve().parents[1]
    sidebar = (root / "app" / "templates" / "platform" / "sidebar.html").read_text(encoding="utf-8")
    pages = (root / "app" / "platform" / "routes" / "pages.py").read_text(encoding="utf-8")
    route = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    task_list = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio_tasks.html"
    ).read_text(encoding="utf-8")
    task_detail = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio_task_detail.html"
    ).read_text(encoding="utf-8")

    assert "任务处理列表" in sidebar
    assert 'href="/platform/image-studio/tasks"' in sidebar
    assert "current_path == '/platform/image-studio'" in sidebar
    assert "current_path.startswith('/platform/image-studio/tasks')" in sidebar

    assert 'PlatformPage("/platform/image-studio/tasks"' in pages
    assert '"/platform/image-studio/tasks/{run_id}"' in pages

    assert '@router.get("/runs/{run_id}/download/{kind}")' in route
    assert '"Content-Disposition"' in route
    assert '"stages": request.get("stages") or []' in route
    assert '"started_at": run.started_at.isoformat()' in route

    assert "QUEUED" in task_list
    assert "RUNNING" in task_list
    assert "SUCCESS" in task_list
    assert "FAILED" in task_list
    assert "查看详情" in task_list
    assert "下载结果" in task_list

    assert "处理阶段" in task_detail
    assert "图片资产" in task_detail
    assert "Base 原图" in task_detail
    assert "生成结果" in task_detail
    assert "/download/" in task_detail
