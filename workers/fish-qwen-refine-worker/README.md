# fish-qwen-refine-worker

独立的 Qwen-Image-Edit-2511 GPU worker。Cloud Run 继续调用既有 \`POST /refine\` 协议；本次 P0 只把首次请求加载改为服务启动 Warmup，并增加运行状态、配置和性能观测。

Worker 仍由本机 ComfyUI 唯一持有 Qwen 模型。启动 Warmup 会提交一张最小测试图，等待 Qwen 工作流真实完成；ComfyUI 成功完成后模型已进入 GPU 缓存，Worker 才返回 \`model_loaded=true\`。不会在 Worker 内再创建第二套 Diffusers 模型。

## VM 部署

将本目录同步到 GPU VM 的 \`/opt/fish-qwen-refine-worker\`，并确认已有 ComfyUI 在 \`127.0.0.1:8188\` 运行、Qwen 模型和自定义节点已经按已验证 workflow 安装。

\`\`\`bash
cd /opt/fish-qwen-refine-worker
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p output
install -m 0644 config.yaml /opt/fish-qwen-refine-worker/config.yaml
install -m 0644 qwen2511_api.json /opt/fish-qwen-refine-worker/qwen2511_api.json
install -m 0644 fish-qwen-refine-worker.service /etc/systemd/system/fish-qwen-refine-worker.service
mkdir -p /etc/yujian
cat >/etc/yujian/fish-qwen-refine-worker.env <<'EOF'
COMFYUI_URL=http://127.0.0.1:8188
QWEN_CONFIG_PATH=/opt/fish-qwen-refine-worker/config.yaml
QWEN_WORKFLOW_PATH=/opt/fish-qwen-refine-worker/qwen2511_api.json
QWEN_OUTPUT_DIR=/opt/fish-qwen-refine-worker/output
QWEN_REQUEST_TIMEOUT_SECONDS=1200
# Optional: set the same value in Cloud Run FISH_QWEN_REFINE_WORKER_TOKEN.
# QWEN_REFINE_WORKER_TOKEN=replace-me
EOF
systemctl daemon-reload
systemctl enable --now fish-qwen-refine-worker
curl http://127.0.0.1:8002/health
\`\`\`

The service starts in \`status=starting\`. During Warmup, \`/refine\` returns 503 with \`QWEN_WORKER_NOT_READY\`. After the first successful Qwen workflow it returns \`status=ready\` and \`model_loaded=true\`. If Warmup fails, the state is \`status=error\` and the error is exposed in the health payload.

The verified L4 workflow uses \`qwen_image_edit_2511_int8_convrot.safetensors\` so the model fits in 24 GiB. The runtime compute precision is configured as \`torch.float16\`; health exposes both \`dtype=float16\` and \`precision_mode=fp16_compute_int8_weights\`. A full fp16 20B weight file must not be substituted on this VM.

## Health

\`\`\`json
{
  "status": "ready",
  "service": "fish-qwen-refine-worker",
  "model": "Qwen-Image-Edit-2511",
  "gpu": "NVIDIA L4",
  "device": "cuda",
  "dtype": "float16",
  "model_loaded": true
}
\`\`\`

## API

\`\`\`bash
curl http://127.0.0.1:8002/health
curl -X POST http://127.0.0.1:8002/refine \\
  -F image=@sam-visible.png \\
  -F 'params={"mode":"fish_preserve_refine_qwen_v1","source_run_id":"FCL_demo","steps":25,"seed":12345,"auto_straighten":false,"prompt":"Restore and refine this fish.","negative_prompt":"human hand"}'
\`\`\`

The response remains compatible with the existing Backend client and adds \`request_id\`, \`performance.inference_time\`, \`performance.total_time\`, GPU/runtime metadata and the selected configuration.

Never commit the bearer token or the VM environment file.
