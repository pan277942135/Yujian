# Fish Segmentation Demo

This is an internal experimental demo. It is not part of the production recognition or catch APIs.

## Flow

```text
uploaded image
  -> normalize_android_source
  -> detector_runtime.detect (DET_FISH_v0.1)
  -> assess_detections (confidence x sqrt(area) primary)
  -> SAM bbox prompt
  -> mask quality gate
  -> RGBA PNG transparent fish
```

The demo endpoints are:

- `GET /debug/fish-segmentation`
- `POST /api/debug/fish-segmentation` with multipart field `image`

## SAM configuration

The runtime uses the open-source Segment Anything package with a `vit_b` checkpoint by default. Configure one of:

```text
SEGMENTATION_CHECKPOINT_PATH=/absolute/path/sam_vit_b.pth
SEGMENTATION_CHECKPOINT_URI=gs://<bucket>/<object>/sam_vit_b.pth
SEGMENTATION_MODEL_TYPE=vit_b
```

The checkpoint is deliberately not stored in GitHub or the production data directories. If it is not configured, the API returns an explicit `503` configuration error.

## Scope

This demo does not modify `app/recognition_pipeline.py`, the detector artifact, classifier, Android app, recognition API, or catch schema. It does not write generated demo assets into the training or production data pools.
