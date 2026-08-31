from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

# Android BitmapFactory accepts this golden JPEG even though its stream is truncated.
# Pillow is strict by default, so opt into the same tolerant behavior for the parity fixture.
ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return (exp / np.sum(exp)).astype(np.float32)


def topk(values: np.ndarray, k: int = 3) -> list[dict]:
    probs = softmax(values)
    order = np.argsort(-probs)[:k]
    return [
        {"index": int(i), "probability": float(probs[i]), "logit": float(values[i])}
        for i in order
    ]


def image_to_nchw(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (224, 224):
            raise RuntimeError(f"golden image must be 224x224, got {image.size}")
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchscript", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tensor-out", default=None)
    parser.add_argument("--expected-index", type=int, default=None)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    tensor = image_to_nchw(Path(args.image))
    tensor_bytes = tensor.astype("<f4", copy=False).tobytes(order="C")
    tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
    if args.tensor_out:
        Path(args.tensor_out).write_bytes(tensor_bytes)

    model = torch.jit.load(args.torchscript, map_location="cpu")
    model.eval()
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(tensor))
        if isinstance(torch_output, (tuple, list)):
            torch_output = torch_output[0]
    torch_logits = torch_output.detach().cpu().numpy().reshape(-1).astype(np.float32)

    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception as exc:
        raise RuntimeError("ai_edge_litert interpreter is required for parity verification") from exc

    interpreter = Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"expected one input/output tensor, got {len(inputs)}/{len(outputs)}")

    input_detail = inputs[0]
    output_detail = outputs[0]
    input_shape = [int(v) for v in input_detail["shape"]]
    if input_shape != list(tensor.shape):
        raise RuntimeError(f"TFLite input shape {input_shape} != golden tensor shape {list(tensor.shape)}")
    if input_detail["dtype"] != np.float32:
        raise RuntimeError(f"TFLite input dtype must be float32, got {input_detail['dtype']}")

    interpreter.set_tensor(input_detail["index"], tensor)
    interpreter.invoke()
    tflite_logits = interpreter.get_tensor(output_detail["index"]).reshape(-1).astype(np.float32)

    if torch_logits.shape != tflite_logits.shape:
        raise RuntimeError(f"output shape mismatch: torch={torch_logits.shape} tflite={tflite_logits.shape}")

    diff = np.abs(torch_logits - tflite_logits)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    cosine = cosine_similarity(torch_logits, tflite_logits)
    torch_top1 = int(np.argmax(torch_logits))
    tflite_top1 = int(np.argmax(tflite_logits))

    report = {
        "input_shape": list(tensor.shape),
        "input_dtype": "float32_le",
        "input_tensor_sha256": tensor_sha256,
        "input_tensor_bytes": len(tensor_bytes),
        "torch_logits": [float(v) for v in torch_logits],
        "tflite_logits": [float(v) for v in tflite_logits],
        "torch_top3": topk(torch_logits),
        "tflite_top3": topk(tflite_logits),
        "torch_top1": torch_top1,
        "tflite_top1": tflite_top1,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cosine_similarity": cosine,
        "expected_index": args.expected_index,
        "max_abs_tolerance": args.max_abs_tolerance,
        "min_cosine": args.min_cosine,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failures: list[str] = []
    if args.expected_index is not None and torch_top1 != args.expected_index:
        failures.append(f"TorchScript golden Top1={torch_top1}, expected={args.expected_index}")
    if torch_top1 != tflite_top1:
        failures.append(f"Top1 mismatch: TorchScript={torch_top1}, TFLite={tflite_top1}")
    if max_abs > args.max_abs_tolerance:
        failures.append(f"max_abs_diff={max_abs:.8f} > {args.max_abs_tolerance}")
    if cosine < args.min_cosine:
        failures.append(f"cosine_similarity={cosine:.8f} < {args.min_cosine}")
    if failures:
        raise RuntimeError("TFLITE_PARITY_FAIL: " + "; ".join(failures))

    print("TFLITE_PARITY_PASS")


if __name__ == "__main__":
    main()
