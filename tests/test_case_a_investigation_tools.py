from __future__ import annotations

import hashlib
import json
import zipfile
import unittest

from scripts.audit_case_a_artifacts import inspect_bundle, inspect_case_a_source
from scripts.audit_detector_dataset_manifest import audit_manifest


class CaseAInvestigationToolTests(unittest.TestCase):
    def test_bundle_audit_checks_zip_model_and_golden_identity(self):
        onnx = b"fake onnx bytes for deterministic audit fixture"
        onnx_sha = hashlib.sha256(onnx).hexdigest()
        metadata = {
            "model_version": "DET_FISH_v0.1",
            "model_family": "YOLOX_NANO",
            "dataset_version": "DET_DS_v0.1",
            "onnx_sha256": onnx_sha,
            "onnx_bytes": len(onnx),
            "onnx": {"input_shape": [1, 3, 416, 416]},
        }
        golden = {
            "model_version": "DET_FISH_v0.1",
            "dataset_version": "DET_DS_v0.1",
            "onnx_sha256": onnx_sha,
            "onnx_bytes": len(onnx),
            "input_shape": [1, 3, 416, 416],
            "cases": [{"id": "fixture"}],
        }
        bundle = self.root / "fixture.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("fish_detector_yolox_nano_v0_1.onnx", onnx)
            archive.writestr("detector_metadata.json", json.dumps(metadata))
            archive.writestr("recognition_pipeline_v1.json", "{}")
            archive.writestr("golden_cases.json", json.dumps(golden))

        result = inspect_bundle(bundle)

        self.assertTrue(result["onnx"]["matches_metadata"])
        self.assertTrue(result["onnx"]["matches_metadata_bytes"])
        self.assertTrue(result["golden_manifest"]["matches_bundle_onnx"])
        self.assertEqual(result["golden_manifest"]["case_count"], 1)

    def test_case_a_source_guard_rejects_any_nonmatching_bytes(self):
        source = self.root / "wrong.jpg"
        source.write_bytes(b"not the authorized Case A source")
        with self.assertRaisesRegex(ValueError, "source SHA mismatch"):
            inspect_case_a_source(source)

    def test_source_audit_checks_dimensions_and_exif_after_sha(self):
        from PIL import Image

        source = self.root / "synthetic_1440x1920.jpg"
        Image.new("RGB", (1440, 1920), (22, 99, 150)).save(source, format="JPEG", quality=95)
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

        result = inspect_case_a_source(source, expected_sha256=source_sha)

        self.assertEqual(result["dimensions"], [1440, 1920])
        self.assertEqual(result["exif_orientation"], 1)

    def test_source_audit_rejects_nonidentity_exif_orientation(self):
        from PIL import Image

        source = self.root / "synthetic_exif_rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (1440, 1920), (22, 99, 150)).save(
            source, format="JPEG", quality=95, exif=exif
        )
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "EXIF orientation mismatch"):
            inspect_case_a_source(source, expected_sha256=source_sha)

    def test_dataset_manifest_audit_reports_cross_split_batch_and_missing_provenance(self):
        entries = [
            {
                "batch_id": "same-session",
                "image_id": "image-a",
                "split": "train",
                "presence_status": "single_fish",
                "negative": False,
                "boxes": [{"confidence": 0.35}],
            },
            {
                "batch_id": "same-session",
                "image_id": "image-b",
                "split": "test",
                "presence_status": "no_fish",
                "negative": True,
                "boxes": [],
            },
        ]

        result = audit_manifest(entries)

        self.assertEqual(result["split_images"], {"test": 1, "train": 1})
        self.assertEqual(result["split_annotations"], {"test": 0, "train": 1})
        self.assertEqual(result["cross_split_batch_ids"], ["same-session"])
        self.assertFalse(result["source_identity_audit_possible"])
        self.assertFalse(result["hard_scene_coverage_audit_possible"])
        self.assertFalse(result["manual_bbox_provenance_audit_possible"])

    def setUp(self):
        import tempfile

        self._temporary_directory = tempfile.TemporaryDirectory()
        from pathlib import Path

        self.root = Path(self._temporary_directory.name)

    def tearDown(self):
        self._temporary_directory.cleanup()
