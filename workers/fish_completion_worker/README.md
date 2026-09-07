# Fish Completion GPU Worker

This service is a strict PowerPaint adapter. It has no mock path and never returns the input image on failure.

Required:
- POWERPAINT_CHECKPOINT_DIR: official v1 checkpoint directory containing unet/unet.safetensors and text_encoder/text_encoder.safetensors.
- COMPLETION_OUTPUT_BUCKET: GCS bucket for generated PNGs.
- ADC with read access to input objects and write access to the output bucket.
- NVIDIA container runtime with a compatible L4 driver.

Optional: WORKER_AUTH_TOKEN, POWERPAINT_STEPS (default 30), POWERPAINT_SEED (default 20260907), POWERPAINT_LOCAL_FILES_ONLY (default true).

The checkpoint is not baked into the image. Mount or stage it on the GPU worker before starting the container.

GET /health returns 503 until CUDA and the checkpoint are loaded.
POST /completion accepts GCS URIs for an RGB image and a grayscale mask and runs official PowerPaint v1 shape-guided inference once.

Build verification is defined in .github/workflows/fish-completion-worker-build.yml.
