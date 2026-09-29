from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.image_studio_worker_client import IMAGE_STUDIO_IDENTITY_V2_MODE, IMAGE_STUDIO_MODE, MAX_REFERENCES
from app.platform.models import ImageStudioRun
from fastapi import HTTPException

import app.platform.routes.image_studio as image_studio_route
from app.platform.routes.image_studio import STORAGE_TYPE, _delete_failed_run, _mask_composite, _require_worker_ready
from app.platform.services.image_studio_prompt import (
    compile_clean_frame_prompt,
    compile_image_studio_prompt,
    compile_scene_transfer_stage_prompt,
    compile_strict_head_swap_prompt,
)
from app.services.image_studio_identity import (
    Box,
    IdentityPreprocessError,
    composite_head_roi,
    measure_strict_composite,
    prepare_strict_identity_assets,
)


def _png(size=(4, 4), color=(0, 0, 0), mode="RGB") -> bytes:
    image = Image.new(mode, size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_image_studio_contract_is_separate_from_fish_lab():
    assert STORAGE_TYPE == "IMAGE_STUDIO_V1"
    assert ImageStudioRun.__tablename__ == "image_studio_run"
    assert IMAGE_STUDIO_MODE == "image_studio_v1"
    assert IMAGE_STUDIO_IDENTITY_V2_MODE == "image_studio_identity_v2"
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
    assert "SOLE identity authority" in compiled.prompt
    assert "original Base person" in compiled.prompt
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
    assert "SUPPORTED_MODES = {QWEN_MODE, IMAGE_STUDIO_MODE, IMAGE_STUDIO_IDENTITY_V2_MODE}" in worker
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

    assert 'id="studioGenerate" class="studio-generate" type="button" disabled' in template
    assert '"/api/qwen-lab/gpu/status?studio_ts="' in template
    assert '"/api/qwen-lab/gpu/start"' in template
    assert '(gpuState === "READY" || gpuState === "BUSY") && payload.model_loaded === true' in template
    assert '"运行中，可继续提交"' in template
    assert '"正在启动 / 加载"' in template
    assert '"/api/image-studio/v1/queue?ts="' in template



def test_image_studio_workbench_is_chat_style_multi_image_composer():
    root = Path(__file__).resolve().parents[1]
    template = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio.html"
    ).read_text(encoding="utf-8")

    assert 'id="studioFiles"' in template
    assert 'multiple' in template
    assert 'id="studioPrompt"' in template
    assert 'id="studioGenerate"' in template
    assert 'form.append("mode", "NATURAL_EDIT")' in template
    assert 'form.append("reference_roles", "[]")' in template
    assert 'files.slice(1).forEach((file) => form.append("references", file))' in template
    assert "最多 3 张图片" in template
    assert "图片顺序就是“图1、图2、图3”" in template
    assert "studioPresets" not in template
    assert "studioRole1" not in template
    assert "studioMask" not in template
    assert "studioSteps" not in template
    assert "<summary>高级设置</summary>" not in template


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
    assert "_requeue_orphaned_running_jobs()" in queue
    assert "ImageStudioRun.status.in_([QUEUED, RUNNING])" in queue
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



def test_delete_failed_run_removes_assets_before_db_record(monkeypatch):
    events = []

    class FakeDb:
        def delete(self, run):
            events.append(("db_delete", run.run_id))

        def commit(self):
            events.append(("db_commit", None))

    run = SimpleNamespace(run_id="IMAGE_STUDIO_20260928_120000_deadbeef", status="FAILED")
    monkeypatch.setattr(
        image_studio_route,
        "_delete_run_assets",
        lambda run_id: events.append(("assets", run_id)) or 4,
    )

    payload = _delete_failed_run(FakeDb(), run)

    assert payload == {
        "deleted": True,
        "run_id": run.run_id,
        "deleted_assets": 4,
    }
    assert events == [
        ("assets", run.run_id),
        ("db_delete", run.run_id),
        ("db_commit", None),
    ]


def test_delete_failed_run_rejects_non_failed_status(monkeypatch):
    called = []
    run = SimpleNamespace(run_id="IMAGE_STUDIO_20260928_120001_feedface", status="SUCCESS")
    monkeypatch.setattr(
        image_studio_route,
        "_delete_run_assets",
        lambda run_id: called.append(run_id) or 0,
    )

    with pytest.raises(HTTPException) as exc_info:
        _delete_failed_run(SimpleNamespace(), run)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["code"] == "IMAGE_STUDIO_DELETE_REQUIRES_FAILED"
    assert called == []


def test_image_studio_failed_cleanup_deletes_managed_run_namespace():
    root = Path(__file__).resolve().parents[1]
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

    assert 'prefix = f"image_studio/v1/runs/{safe_run_id}/"' in route
    assert "client.list_blobs(bucket, prefix=prefix)" in route
    assert "shutil.rmtree(run_dir)" in route
    assert '@router.delete("/runs/{run_id}")' in route
    assert '@router.delete("/runs/failed")' in route
    assert "仅 FAILED 任务允许使用失败任务清理" in route

    assert "清理全部失败任务" in task_list
    assert 'data-delete-run="' in task_list
    assert 'method: "DELETE"' in task_list
    assert "Reference、Mask、生成结果" in task_list

    assert "删除失败任务" in task_detail
    assert 'method: "DELETE"' in task_detail
    assert "Reference、Mask、生成结果" in task_detail



def test_image_studio_reclaims_orphaned_running_only_under_global_lock():
    root = Path(__file__).resolve().parents[1]
    queue = (root / "app" / "services" / "image_studio_jobs.py").read_text(encoding="utf-8")

    acquire = queue.index("token = _acquire_cross_instance_lock()")
    reclaim = queue.index("_requeue_orphaned_running_jobs()", acquire)
    claim = queue.index("run_id = _claim_next_job()", reclaim)

    assert acquire < reclaim < claim
    assert "any RUNNING row" in queue
    assert "lease timeout" in queue
    assert "timedelta" not in queue



def test_identity_reference_overrides_base_face_and_negative_does_not_block_swap():
    compiled = compile_image_studio_prompt(
        "Replace the person identity while preserving pose and scene.",
        mode="IDENTITY_LOCK",
        preservation="MAX",
        reference_roles=["IDENTITY", "FACE_ANGLE"],
    )

    assert "Picture 2 is the SOLE identity authority" in compiled.prompt
    assert "Picture 1 / Base is authoritative only for camera, crop, body pose" in compiled.prompt
    assert "It is NOT authoritative for the person's face or identity" in compiled.prompt
    assert "Do not preserve the original Base person's facial identity" in compiled.prompt
    assert "IDENTITY wins" in compiled.prompt
    assert "unrequested identity change" not in compiled.negative_prompt
    assert "preserving the original Base face identity" in compiled.negative_prompt
    assert "identity drift away from Picture 2" in compiled.negative_prompt


def test_identity_authority_also_applies_to_multi_reference_scene_workflow():
    compiled = compile_image_studio_prompt(
        "Replace the subject identity and move the subject into the reference scene.",
        mode="MULTI_REFERENCE",
        preservation="STRONG",
        reference_roles=["IDENTITY", "SCENE"],
    )

    assert "Picture 2 is the SOLE identity authority" in compiled.prompt
    assert "Picture 3 role = SCENE" in compiled.prompt
    assert "original Base face must not be preserved" in compiled.prompt
    assert "hybrid face between Base and identity reference" in compiled.negative_prompt


def test_image_studio_detail_images_do_not_reload_every_poll():
    root = Path(__file__).resolve().parents[1]
    detail = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio_task_detail.html"
    ).read_text(encoding="utf-8")

    assert "lastAssetSignature" in detail
    assert "assetSignature !== lastAssetSignature" in detail
    assert "preview + '?t=' + Date.now()" not in detail
    assert 'run.status === "SUCCESS" || run.status === "FAILED"' in detail
    assert "window.clearInterval(pollTimer)" in detail


def test_natural_edit_delegates_reference_roles_to_qwen():
    compiled = compile_image_studio_prompt(
        "把图2的人物替换到图1里，保持图1的姿势和背景。",
        mode="NATURAL_EDIT",
        preservation="NORMAL",
        reference_roles=["REFERENCE"],
    )

    assert compiled.mode == "NATURAL_EDIT"
    assert compiled.preservation == "NORMAL"
    assert compiled.reference_roles == ("REFERENCE",)
    assert "Infer the role of each picture from the user's instruction" in compiled.prompt
    assert "Picture 1 is the first uploaded image." in compiled.prompt
    assert "Picture 2 is the 2 uploaded image." in compiled.prompt
    assert "IDENTITY REPLACEMENT AUTHORITY" not in compiled.prompt
    assert "BASE IMAGE AUTHORITY" not in compiled.prompt



def test_qwen_uat_accepts_busy_loaded_worker_after_studio_success():
    root = Path(__file__).resolve().parents[1]
    uat = (root / "scripts" / "qwen_gpu_manual_uat.sh").read_text(encoding="utf-8")

    assert "wait_for_worker_healthy()" in uat
    assert '"$current" == "READY" || "$current" == "BUSY"' in uat
    assert '"$model_loaded" == "true"' in uat
    assert 'wait_for_worker_healthy 24 "post-image-studio"' in uat



def test_qwen_deploy_is_noninvasive_for_live_image_studio_queue():
    root = Path(__file__).resolve().parents[1]
    boot = (root / "scripts" / "deploy_qwen_worker_boot.sh").read_text(encoding="utf-8")
    uat = (root / "scripts" / "qwen_gpu_manual_uat.sh").read_text(encoding="utf-8")

    assert "worker_changed=$WORKER_CHANGED" in boot
    assert "cmp -s" in boot
    assert r"src=\${pair%%:*}" in boot
    assert r"dst=\${pair#*:}" in boot
    assert r'if [[ ! -f "\$dst" ]] || ! cmp -s "\$src" "\$dst"; then' in boot
    assert r'if [[ "\$same" == "true" ]]' in boot
    assert "Skipping ComfyUI/Qwen restart to preserve live generation" in boot
    assert "Waiting for live Image Studio/Qwen workload to drain before restart" in boot
    assert "wait_for_idle_before_restart" in boot
    assert "sudo systemctl restart fish-qwen-comfyui.service" in boot
    assert 'wait_for_worker_healthy 24 "post-generate"' in uat
    assert 'wait_for_worker_healthy 24 "post-image-studio"' in uat
    assert 'wait_for_display READY 24 "post-generate"' not in uat



class _FakeVisionClient:
    def __init__(self, boxes):
        self._boxes = list(boxes)

    def face_detection(self, *, image, max_results=5):
        annotations = []
        for left, top, right, bottom in self._boxes:
            vertices = [
                SimpleNamespace(x=left, y=top),
                SimpleNamespace(x=right, y=top),
                SimpleNamespace(x=right, y=bottom),
                SimpleNamespace(x=left, y=bottom),
            ]
            annotations.append(
                SimpleNamespace(
                    fd_bounding_poly=SimpleNamespace(vertices=vertices),
                    bounding_poly=None,
                )
            )
        return SimpleNamespace(
            face_annotations=annotations,
            error=SimpleNamespace(message=""),
        )


def test_strict_head_swap_prompt_is_local_and_identity_authoritative():
    compiled = compile_strict_head_swap_prompt(
        "Replace only the head identity.",
        has_angle_reference=True,
        identity_strength="HIGH",
    )

    assert compiled.mode == "STRICT_HEAD_SWAP"
    assert compiled.reference_roles == ("IDENTITY", "FACE_ANGLE")
    assert "LOCAL HEAD ROI" in compiled.prompt
    assert "Picture 2 is a tightly cropped IDENTITY head reference" in compiled.prompt
    assert "Do not invent shoulders, torso, outfit" in compiled.prompt
    assert "feather-composited back into the untouched Base image" in compiled.prompt
    assert "copying identity-reference clothing" in compiled.negative_prompt


def test_scene_stage_prompt_does_not_ask_scene_reference_for_identity():
    compiled = compile_scene_transfer_stage_prompt("Move the subject into the target room.")

    assert compiled.mode == "SCENE_TRANSFER"
    assert compiled.reference_roles == ("SCENE",)
    assert "Picture 2 / SCENE is authoritative only for the environment" in compiled.prompt
    assert "Do not copy any person identity" in compiled.prompt
    assert "separate strict head-swap stage" in compiled.prompt


def test_blend_and_full_rebuild_have_distinct_authority_semantics():
    blend = compile_image_studio_prompt(
        "Blend lightly.",
        mode="IDENTITY_BLEND",
        preservation="STRONG",
        reference_roles=["IDENTITY"],
    )
    rebuild = compile_image_studio_prompt(
        "Rebuild character.",
        mode="FULL_CHARACTER_REBUILD",
        preservation="NORMAL",
        reference_roles=["IDENTITY"],
    )

    assert "soft identity influence, not a replacement authority" in blend.prompt
    assert "Base remains the primary person identity" in blend.prompt
    assert "The Base person remains identity authority" in blend.prompt
    assert "sole authoritative source" not in blend.prompt
    assert "FULL CHARACTER REBUILD" in rebuild.prompt
    assert "original person appearance, face, hair, clothing, and body styling may be rebuilt" in rebuild.prompt


def test_strict_identity_preprocess_builds_head_crops_and_mask(monkeypatch):
    base = _png(size=(200, 300), color=(20, 30, 40))
    identity = _png(size=(240, 320), color=(180, 160, 140))
    clients = iter([
        _FakeVisionClient([(70, 80, 130, 150)]),
    ])
    # One detector object is reused for both images, so return base then identity.
    class SequenceClient:
        def __init__(self):
            self.calls = 0

        def face_detection(self, *, image, max_results=5):
            self.calls += 1
            box = (70, 80, 130, 150) if self.calls == 1 else (80, 70, 160, 165)
            return _FakeVisionClient([box]).face_detection(image=image, max_results=max_results)

    prepared = prepare_strict_identity_assets(
        base_bytes=base,
        identity_bytes=identity,
        tightness="MEDIUM",
        client=SequenceClient(),
    )

    assert prepared.base_face_box == Box(70, 80, 130, 150)
    assert prepared.base_head_box.width > prepared.base_face_box.width
    assert prepared.base_head_box.height > prepared.base_face_box.height
    assert prepared.identity_head_box.width > prepared.identity_face_box.width
    assert prepared.base_head_crop.startswith(b"\x89PNG")
    assert prepared.identity_face_crop.startswith(b"\x89PNG")
    assert prepared.identity_head_crop.startswith(b"\x89PNG")
    assert prepared.head_mask.startswith(b"\x89PNG")
    assert prepared.head_mask_preview.startswith(b"\x89PNG")


def test_strict_composite_preserves_every_pixel_outside_head_roi():
    base = _png(size=(100, 120), color=(10, 20, 30))
    edited = _png(size=(40, 50), color=(220, 120, 80))
    head_box = Box(30, 20, 70, 70)

    mask_image = Image.new("L", (40, 50), 255)
    mask_output = io.BytesIO()
    mask_image.save(mask_output, format="PNG")

    final_bytes = composite_head_roi(
        base_bytes=base,
        edited_head_bytes=edited,
        head_box=head_box,
        roi_mask_bytes=mask_output.getvalue(),
    )
    metrics = measure_strict_composite(
        base_bytes=base,
        result_bytes=final_bytes,
        head_box=head_box,
    )

    assert metrics["outside_roi_preserved"] is True
    assert metrics["outside_roi_changed_pixels"] == 0
    assert metrics["head_roi_mean_abs_diff"] > 0


def test_strict_identity_rejects_ambiguous_multiple_faces():
    base = _png(size=(200, 300), color=(20, 30, 40))
    client = _FakeVisionClient([
        (20, 40, 90, 120),
        (105, 45, 175, 125),
    ])

    with pytest.raises(IdentityPreprocessError) as exc_info:
        prepare_strict_identity_assets(
            base_bytes=base,
            identity_bytes=base,
            client=client,
        )

    assert exc_info.value.code == "MULTIPLE_FACES_AMBIGUOUS"


def test_image_studio_v2_route_and_worker_contracts_are_explicit():
    root = Path(__file__).resolve().parents[1]
    route = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    worker = (
        root
        / "workers"
        / "fish-qwen-refine-worker"
        / "worker.py"
    ).read_text(encoding="utf-8")
    client = (root / "app" / "image_studio_worker_client.py").read_text(encoding="utf-8")

    assert '"STRICT_HEAD_SWAP"' in route
    assert '"HEAD_SWAP_SCENE_TRANSFER"' in route
    assert '"identity_transfer_v2"' in route
    assert 'pipeline_stage="STRICT_HEAD_SWAP"' in route
    assert 'pipeline_stage="SCENE_TRANSFER"' in route
    assert '"AUTO_CROP"' in route
    assert '"AUTO_MASK"' in route
    assert '"COMPOSITE"' in route

    assert 'IMAGE_STUDIO_IDENTITY_V2_MODE = "image_studio_identity_v2"' in worker
    assert "pipeline_stage" in worker
    assert 'IMAGE_STUDIO_IDENTITY_V2_MODE = "image_studio_identity_v2"' in client


def test_image_studio_advanced_modes_are_not_exposed_in_chat_workbench():
    root = Path(__file__).resolve().parents[1]
    template = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio.html"
    ).read_text(encoding="utf-8")

    assert 'value="STRICT_HEAD_SWAP"' not in template
    assert 'value="HEAD_SWAP_SCENE_TRANSFER"' not in template
    assert 'value="IDENTITY_BLEND"' not in template
    assert 'value="FULL_CHARACTER_REBUILD"' not in template
    assert 'id="studioRef3"' not in template
    assert 'id="studioIdentityStrength"' not in template
    assert 'id="studioHeadTightness"' not in template
    assert 'id="studioKeepHairColor"' not in template
    assert 'id="studioKeepBaseHairShape"' not in template
    assert 'form.append("mode", "NATURAL_EDIT")' in template


def test_task_detail_exposes_strict_identity_intermediates():
    root = Path(__file__).resolve().parents[1]
    route = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    detail = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio_task_detail.html"
    ).read_text(encoding="utf-8")

    for kind in [
        "base_face_crop",
        "base_head_crop",
        "identity_face_crop",
        "identity_head_crop",
        "mask_binary",
        "mask_preview",
        "edited_head_roi",
    ]:
        assert kind in route
    assert "intermediate_assets" in route
    assert "run.intermediate_assets" in detail
    assert "身份强度" in detail
    assert "头部范围" in detail


def test_busy_loaded_worker_is_accepted_for_fifo_submission(monkeypatch):
    monkeypatch.setattr(
        image_studio_route,
        "check_qwen_refine_worker",
        lambda: {"health": {"status": "busy", "model_loaded": True}},
    )
    assert _require_worker_ready()["status"] == "busy"



def test_clean_frame_prompt_removes_only_overlay_ui_by_default():
    compiled = compile_clean_frame_prompt("Keep the photographed subject unchanged.")

    assert compiled.mode == "BASE_EDIT"
    assert compiled.preservation == "MAX"
    assert "watermark text" in compiled.prompt
    assert "status bars" in compiled.prompt
    assert "toolbars" in compiled.prompt
    assert "screenshot controls" in compiled.prompt
    assert "Preserve genuine in-scene signage" in compiled.prompt
    assert "printed clothing graphics" in compiled.prompt
    assert "genuine product logos" in compiled.negative_prompt


def test_image_studio_chat_workbench_keeps_output_defaults_hidden():
    root = Path(__file__).resolve().parents[1]
    template = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio.html"
    ).read_text(encoding="utf-8")

    assert 'id="studioOutputLongEdge"' not in template
    assert 'id="studioCleanOutput"' not in template
    assert 'form.append("output_long_edge", "1024")' in template
    assert 'form.append("clean_output", "true")' in template
    assert 'form.append("resolution_mode", "target_long_edge")' in template


def test_image_studio_route_has_clean_frame_and_exact_output_resize():
    root = Path(__file__).resolve().parents[1]
    route = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    detail = (
        root
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "image_studio_task_detail.html"
    ).read_text(encoding="utf-8")

    assert 'output_long_edge: int = Form(default=1024)' in route
    assert 'clean_output: bool = Form(default=True)' in route
    assert '"CLEAN_FRAME"' in route
    assert 'compile_clean_frame_prompt(' in route
    assert 'pipeline_stage="CLEAN_FRAME"' in route
    assert '_resize_png_to_long_edge(' in route
    assert '"output_size"' in route
    assert '"clean_frame_result": "Clean Frame Result"' in route
    assert "输出长边" in detail
    assert "最终尺寸" in detail
    assert "清理叠加 UI" in detail


def test_worker_supports_target_long_edge_generation():
    root = Path(__file__).resolve().parents[1]
    worker = (
        root
        / "workers"
        / "fish-qwen-refine-worker"
        / "worker.py"
    ).read_text(encoding="utf-8")
    client = (root / "app" / "image_studio_worker_client.py").read_text(encoding="utf-8")

    assert '"target_long_edge"' in worker
    assert "_aspect_dimensions" in worker
    assert "{768, 1024, 1536, 2048}" in worker
    assert "target_long_edge: int | None = None" in client
    assert '"target_long_edge": params["target_long_edge"]' in client



def test_natural_edit_route_uses_generic_reference_roles():
    root = Path(__file__).resolve().parents[1]
    route = (root / "app" / "platform" / "routes" / "image_studio.py").read_text(encoding="utf-8")
    prompt_service = (
        root / "app" / "platform" / "services" / "image_studio_prompt.py"
    ).read_text(encoding="utf-8")

    assert 'if mode_value == "NATURAL_EDIT":' in route
    assert 'roles = ["REFERENCE"] * reference_count' in route
    assert '"NATURAL_EDIT",' in prompt_service
    assert '"REFERENCE",' in prompt_service
