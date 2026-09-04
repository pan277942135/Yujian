from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_RGB = tuple(round(x * 255) for x in IMAGENET_MEAN)


class WholeImageLetterbox:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        contained = ImageOps.contain(source, (self.size, self.size), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), IMAGENET_MEAN_RGB)
        x = (self.size - contained.width) // 2
        y = (self.size - contained.height) // 2
        canvas.paste(contained, (x, y))
        return canvas


def center_crop_image(image: Image.Image, size: int) -> Image.Image:
    resize = transforms.Resize(int(size * 1.15))
    crop = transforms.CenterCrop(size)
    return crop(resize(image.convert("RGB")))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    return transform(image).unsqueeze(0)


def tensor_sha(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)


def ranked(logits: np.ndarray, classes: list[dict], k: int = 3) -> list[dict]:
    probs = softmax_np(logits)
    order = np.argsort(-probs)[:k]
    by_index = {int(row["class_index"]): row for row in classes}
    result: list[dict] = []
    for index in order:
        row = by_index.get(int(index), {})
        result.append(
            {
                "class_index": int(index),
                "species_key": row.get("species_key"),
                "display_name_zh": row.get("display_name_zh") or row.get("name_zh"),
                "probability": float(probs[index]),
                "logit": float(logits[index]),
            }
        )
    return result


def run_torch(model: torch.jit.ScriptModule, tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        output = model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
    return output.detach().cpu().numpy().reshape(-1).astype(np.float32)


def run_tflite(model_path: str, tensor: torch.Tensor) -> np.ndarray:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception as exc:
        raise RuntimeError("Install ai-edge-litert/litert-torch to use --tflite") from exc

    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    actual_shape = [int(v) for v in input_detail["shape"]]
    if actual_shape != list(array.shape):
        raise RuntimeError(f"TFLite input shape {actual_shape} != tensor {list(array.shape)}")
    interpreter.set_tensor(input_detail["index"], array)
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"]).reshape(-1).astype(np.float32)


def diff_report(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom else (1.0 if np.array_equal(a, b) else 0.0)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": cosine,
        "top1_match": int(np.argmax(a)) == int(np.argmax(b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Model Factory center-crop vs App whole-image letterbox on one exact source image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--torchscript", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tflite", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    classes = sorted(list(class_map.get("classes") or []), key=lambda x: int(x.get("class_index", 0)))
    if not classes:
        raise RuntimeError("class_map.json has no classes")

    with Image.open(args.image) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
        original_size = [original.width, original.height]
        center_image = center_crop_image(original, args.image_size)
        letterbox_image = WholeImageLetterbox(args.image_size)(original)

    center_image.save(out_dir / "01_model_factory_center_crop.png")
    letterbox_image.save(out_dir / "02_app_letterbox.png")

    center_tensor = image_to_tensor(center_image)
    letterbox_tensor = image_to_tensor(letterbox_image)
    center_tensor.detach().cpu().numpy().astype("<f4", copy=False).tofile(out_dir / "01_center_crop_tensor_f32le.bin")
    letterbox_tensor.detach().cpu().numpy().astype("<f4", copy=False).tofile(out_dir / "02_letterbox_tensor_f32le.bin")

    model = torch.jit.load(args.torchscript, map_location="cpu")
    model.eval()
    center_torch = run_torch(model, center_tensor)
    letterbox_torch = run_torch(model, letterbox_tensor)

    result = {
        "source_image": str(Path(args.image)),
        "source_size": original_size,
        "image_size": args.image_size,
        "model_factory_contract": "Resize(int(224*1.15)) -> CenterCrop(224) -> RGB -> ImageNet normalize",
        "app_contract": "Whole-image fit -> ImageNet-mean padding -> RGB -> ImageNet normalize",
        "center_crop": {
            "preview": "01_model_factory_center_crop.png",
            "tensor": "01_center_crop_tensor_f32le.bin",
            "tensor_sha256": tensor_sha(center_tensor),
            "torch_logits": [float(v) for v in center_torch],
            "torch_top3": ranked(center_torch, classes),
        },
        "letterbox": {
            "preview": "02_app_letterbox.png",
            "tensor": "02_letterbox_tensor_f32le.bin",
            "tensor_sha256": tensor_sha(letterbox_tensor),
            "torch_logits": [float(v) for v in letterbox_torch],
            "torch_top3": ranked(letterbox_torch, classes),
        },
        "preprocess_effect": {
            "top1_match": int(np.argmax(center_torch)) == int(np.argmax(letterbox_torch)),
            "center_top1": int(np.argmax(center_torch)),
            "letterbox_top1": int(np.argmax(letterbox_torch)),
        },
    }

    if args.tflite:
        center_tflite = run_tflite(args.tflite, center_tensor)
        letterbox_tflite = run_tflite(args.tflite, letterbox_tensor)
        result["center_crop"]["tflite_logits"] = [float(v) for v in center_tflite]
        result["center_crop"]["tflite_top3"] = ranked(center_tflite, classes)
        result["center_crop"]["torch_vs_tflite"] = diff_report(center_torch, center_tflite)
        result["letterbox"]["tflite_logits"] = [float(v) for v in letterbox_tflite]
        result["letterbox"]["tflite_top3"] = ranked(letterbox_tflite, classes)
        result["letterbox"]["torch_vs_tflite"] = diff_report(letterbox_torch, letterbox_tflite)

    report_path = out_dir / "diagnosis.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
