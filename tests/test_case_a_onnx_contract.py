from __future__ import annotations

import unittest

import onnx
from onnx import TensorProto, helper

from scripts.inspect_detector_onnx_contract import analyze_model


class DetectorOnnxContractAuditTests(unittest.TestCase):
    def test_reports_head_sigmoids_for_objectness_and_class(self) -> None:
        obj_input = "/head/obj_preds.0/Conv_output_0"
        cls_input = "/head/cls_preds.0/Conv_output_0"
        obj_output = "/head/Sigmoid_output_0"
        cls_output = "/head/Sigmoid_1_output_0"
        inputs = [
            helper.make_tensor_value_info(obj_input, TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info(cls_input, TensorProto.FLOAT, [1, 1]),
        ]
        outputs = [
            helper.make_tensor_value_info(obj_output, TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info(cls_output, TensorProto.FLOAT, [1, 1]),
        ]
        nodes = [
            helper.make_node("Sigmoid", [obj_input], [obj_output], name="/head/Sigmoid"),
            helper.make_node("Sigmoid", [cls_input], [cls_output], name="/head/Sigmoid_1"),
        ]
        graph = helper.make_graph(nodes, "audit-test", inputs, outputs)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)

        report = analyze_model(model, "a" * 64, 123, {"onnx": {"output_contract": "decoded_xywh_objectness_class_probability"}})

        self.assertEqual(report["onnx"]["objectness_head_sigmoid_count"], 1)
        self.assertEqual(report["onnx"]["class_head_sigmoid_count"], 1)
        self.assertTrue(report["interpretation"]["head_probability_activations_observed"])

    def test_does_not_infer_probability_activations_when_head_has_none(self) -> None:
        value = "raw"
        inputs = [helper.make_tensor_value_info(value, TensorProto.FLOAT, [1, 1])]
        outputs = [helper.make_tensor_value_info(value, TensorProto.FLOAT, [1, 1])]
        graph = helper.make_graph([], "audit-test-no-activation", inputs, outputs)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        report = analyze_model(model, "b" * 64, 12, None)

        self.assertFalse(report["interpretation"]["head_probability_activations_observed"])


if __name__ == "__main__":
    unittest.main()
