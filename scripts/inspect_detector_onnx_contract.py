#!/usr/bin/env python3
"""Inspect a detector ONNX graph and compare its declared model contract.

This is a read-only offline audit. It hashes the supplied ONNX bytes, checks
the graph with ONNX's checker, and reports input/output tensors plus detector
head activation nodes. It does not download artifacts or run inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx


def _shape(value_info: Any) -> list[int | str | None]:
    dimensions = value_info.type.tensor_type.shape.dim
    return [
        int(dim.dim_value) if dim.dim_value else (dim.dim_param or None)
        for dim in dimensions
    ]


def _tensor_info(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    try:
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    except ValueError:
        dtype = str(tensor_type.elem_type)
    return {"name": value_info.name, "dtype": dtype, "shape": _shape(value_info)}


def analyze_model(model: Any, onnx_sha256: str, onnx_bytes: int, metadata: dict[str, Any] | None) -> dict[str, Any]:
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    head_sigmoids = []
    for index, node in enumerate(graph.node):
        if node.op_type != "Sigmoid" or not node.name.startswith("/head/"):
            continue
        source = node.input[0] if node.input else ""
        if "obj_preds" in source:
            semantic = "objectness_probability"
        elif "cls_preds" in source:
            semantic = "class_probability"
        else:
            semantic = "other_head_activation"
        head_sigmoids.append({
            "node_index": index,
            "name": node.name,
            "semantic_from_source_name": semantic,
            "input": source,
            "output": node.output[0] if node.output else None,
        })

    objectness_sigmoids = [item for item in head_sigmoids if item["semantic_from_source_name"] == "objectness_probability"]
    class_sigmoids = [item for item in head_sigmoids if item["semantic_from_source_name"] == "class_probability"]
    declared_contract = ((metadata or {}).get("onnx") or {}).get("output_contract")
    actual_input = _tensor_info(graph.input[0]) if graph.input else None
    actual_output = _tensor_info(graph.output[0]) if graph.output else None
    metadata_onnx = (metadata or {}).get("onnx") or {}
    metadata_sha = (metadata or {}).get("onnx_sha256")
    metadata_bytes = (metadata or {}).get("onnx_bytes")
    semantic_supported = bool(objectness_sigmoids and class_sigmoids)

    return {
        "schema_version": "DETECTOR_ONNX_OUTPUT_CONTRACT_AUDIT_V1",
        "artifact": {
            "sha256": onnx_sha256,
            "bytes": onnx_bytes,
            "metadata_sha256": metadata_sha,
            "metadata_bytes": metadata_bytes,
            "matches_metadata_sha256": metadata_sha == onnx_sha256 if metadata else None,
            "matches_metadata_bytes": int(metadata_bytes) == onnx_bytes if metadata and metadata_bytes is not None else None,
        },
        "onnx": {
            "ir_version": model.ir_version,
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "graph_input_count": len(graph.input),
            "graph_output_count": len(graph.output),
            "input": actual_input,
            "output": actual_output,
            "node_count": len(graph.node),
            "op_counts": dict(sorted(op_counts.items())),
            "sigmoid_node_count": int(op_counts.get("Sigmoid", 0)),
            "softmax_node_count": int(op_counts.get("Softmax", 0) + op_counts.get("LogSoftmax", 0)),
            "exp_node_count": int(op_counts.get("Exp", 0)),
            "detector_head_sigmoids": head_sigmoids,
            "objectness_head_sigmoid_count": len(objectness_sigmoids),
            "class_head_sigmoid_count": len(class_sigmoids),
            "onnx_checker": "PASS",
        },
        "declared_contract": {
            "output_contract": declared_contract,
            "input_name": metadata_onnx.get("input_name"),
            "input_shape": metadata_onnx.get("input_shape"),
            "output_shape": metadata_onnx.get("output_shape"),
        },
        "interpretation": {
            "head_probability_activations_observed": semantic_supported,
            "head_probability_activation_evidence": "Sigmoid nodes consume objectness and class prediction head tensors before the multi-scale head output is assembled.",
            "softmax_observed": bool(op_counts.get("Softmax", 0) or op_counts.get("LogSoftmax", 0)),
            "scope_limit": "This graph is the supplied artifact only. It does not establish the identity or graph of an unavailable current production GCS object.",
        },
    }


def inspect_onnx(onnx_path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    raw = onnx_path.read_bytes()
    model = onnx.load_model_from_string(raw)
    onnx.checker.check_model(model)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else None
    result = analyze_model(model, hashlib.sha256(raw).hexdigest(), len(raw), metadata)
    result["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["toolchain"] = {"onnx": onnx.__version__}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_onnx(args.onnx, args.metadata)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
