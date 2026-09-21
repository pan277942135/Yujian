from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_worker_boot_units_restore_comfyui_dependency():
    comfy = (ROOT / "workers/fish-qwen-refine-worker/fish-qwen-comfyui.service").read_text(encoding="utf-8")
    worker = (ROOT / "workers/fish-qwen-refine-worker/fish-qwen-refine-worker.service").read_text(encoding="utf-8")
    wait_script = (ROOT / "workers/fish-qwen-refine-worker/wait-for-comfyui.sh").read_text(encoding="utf-8")
    deploy = (ROOT / "scripts/deploy_qwen_worker_boot.sh").read_text(encoding="utf-8")

    assert "WorkingDirectory=/opt/comfyui-qwen/ComfyUI" in comfy
    assert "main.py --listen 127.0.0.1 --port 8188" in comfy
    assert "Requires=fish-qwen-comfyui.service" in worker
    assert "After=network-online.target fish-qwen-comfyui.service" in worker
    assert "ExecStartPre=/opt/fish-qwen-refine-worker/wait-for-comfyui.sh" in worker
    assert "system_stats" in wait_script
    assert "systemctl enable fish-qwen-comfyui.service fish-qwen-refine-worker.service" in deploy
    assert "api/qwen-lab/gpu/start" in deploy
    assert "gcloud compute instances start" not in deploy


def test_qwen_worker_health_contract_surfaces_boot_stage_and_error_code():
    worker = (ROOT / "workers/fish-qwen-refine-worker/worker.py").read_text(encoding="utf-8")

    assert 'stage="waiting_comfyui"' in worker
    assert 'stage="qwen_warmup"' in worker
    assert '"COMFYUI_UNAVAILABLE"' in worker
    assert 'payload["error_code"]' in worker
