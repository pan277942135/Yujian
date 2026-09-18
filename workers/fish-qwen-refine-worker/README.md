# fish-qwen-refine-worker

独立的 Qwen-Image-Edit-2511 GPU worker。Cloud Run 只调用 `POST /refine`，不暴露
ComfyUI 节点图；输入必须是 Fish Completion Lab 产出的 SAM Visible。

## VM 部署

将本目录复制到 GPU VM 的 `/opt/fish-qwen-refine-worker`，并确认已有 ComfyUI
在 `127.0.0.1:8188` 运行、Qwen 模型和自定义节点已经按已验证 workflow 安装。

```bash
cd /opt/fish-qwen-refine-worker
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p output
install -m 0644 qwen2511_api.json /opt/fish-qwen-refine-worker/qwen2511_api.json
install -m 0644 fish-qwen-refine-worker.service /etc/systemd/system/fish-qwen-refine-worker.service
mkdir -p /etc/yujian
cat >/etc/yujian/fish-qwen-refine-worker.env <<'EOF'
COMFYUI_URL=http://127.0.0.1:8188
QWEN_WORKFLOW_PATH=/opt/fish-qwen-refine-worker/qwen2511_api.json
QWEN_OUTPUT_DIR=/opt/fish-qwen-refine-worker/output
QWEN_REQUEST_TIMEOUT_SECONDS=1200
# Optional: set the same value in Cloud Run FISH_QWEN_REFINE_WORKER_TOKEN.
# QWEN_REFINE_WORKER_TOKEN=replace-me
EOF
systemctl daemon-reload
systemctl enable --now fish-qwen-refine-worker
curl http://127.0.0.1:8002/health
```

The systemd `User` is the known VM account `pan277942135`; change it in the unit
if the actual GPU VM account differs. If the worker is called from Cloud Run, allow
TCP 8002 only from the intended egress path or put it behind the existing protected
network boundary. Never commit the bearer token.

## API

```bash
curl http://127.0.0.1:8002/health
curl -X POST http://127.0.0.1:8002/refine \
  -F image=@sam-visible.png \
  -F 'params={"mode":"fish_preserve_refine_qwen_v1","source_run_id":"FCL_demo","steps":20,"seed":12345,"auto_straighten":false,"prompt":"Restore and refine this fish.","negative_prompt":"human hand"}'
```

The response returns `refine_result_uri` and `final_asset_uri` under the worker's
static `/output` mount. Auto Straighten is currently a transparent parameter;
the response explicitly reports `auto_straighten_applied=false`.
