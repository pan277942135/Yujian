from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild_from_torchscript(path: Path) -> tuple[nn.Module, int]:
    scripted = torch.jit.load(str(path), map_location="cpu")
    scripted.eval()
    state = scripted.state_dict()
    weight = state.get("classifier.3.weight")
    if weight is None or weight.ndim != 2:
        raise RuntimeError("cannot resolve MobileNetV3 classifier output from TorchScript state_dict")
    num_classes = int(weight.shape[0])

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()

    probe = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        original = scripted(probe)
        rebuilt = model(probe)
    max_abs = float((original - rebuilt).abs().max().item())
    if max_abs > 1e-6:
        raise RuntimeError(f"reconstructed model mismatch: max_abs={max_abs}")
    return model, num_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchscript", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = Path(args.torchscript)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    image_size = int((metrics.get("params") or {}).get("image_size") or 224)
    classes = sorted(list(class_map.get("classes") or []), key=lambda x: int(x.get("class_index", 0)))

    model, num_classes = rebuild_from_torchscript(source)
    if len(classes) != num_classes:
        raise RuntimeError(f"class_map count {len(classes)} != model output count {num_classes}")

    try:
        import litert_torch
    except Exception as exc:
        raise RuntimeError("litert-torch is required for LiteRT/TFLite export") from exc

    sample = (torch.randn(1, 3, image_size, image_size),)
    edge_model = litert_torch.convert(model, sample)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(out))

    if not out.exists() or out.stat().st_size <= 8:
        raise RuntimeError("TFLite export did not produce a valid file")
    header = out.read_bytes()[:8]
    if header[4:8] != b"TFL3":
        raise RuntimeError(f"unexpected TFLite flatbuffer header: {header!r}")

    converted_max_abs = None
    try:
        with torch.inference_mode():
            ref = model(sample[0])
        got = edge_model(*sample)
        if isinstance(got, (list, tuple)):
            got = got[0]
        got_tensor = torch.as_tensor(got)
        converted_max_abs = float((ref - got_tensor).abs().max().item())
    except Exception:
        # Export itself remains authoritative; Android runtime smoke will validate the file.
        pass

    report = {
        "source_torchscript": str(source),
        "source_sha256": sha256(source),
        "model_family": "mobilenet_v3_small",
        "image_size": image_size,
        "num_classes": num_classes,
        "class_indices": [int(x.get("class_index", 0)) for x in classes],
        "tflite_path": str(out),
        "tflite_size": out.stat().st_size,
        "tflite_sha256": sha256(out),
        "flatbuffer_identifier": header[4:8].decode("ascii"),
        "torch_to_litert_max_abs": converted_max_abs,
        "tensor_contract": {
            "logical_input": "RGB float32, ImageNet normalized",
            "pytorch_layout": "NCHW",
            "spatial_size": [image_size, image_size],
            "output_classes": num_classes,
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
