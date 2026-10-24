from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.detector_runtime import DetectorCandidate
from app.recognition_pipeline import BBox
from scripts.diagnose_detector_orientation import (
    _candidate_record,
    _write_raw_output_artifact,
)


class CaseAReplayEvidenceTests(unittest.TestCase):
    def test_missing_ground_truth_is_not_reported_as_zero_iou(self) -> None:
        box = BBox(0.1, 0.2, 0.4, 0.8)
        candidate = DetectorCandidate(
            rank=1,
            objectness=0.5,
            fish_probability=0.8,
            confidence=0.4,
            box=box,
            area_ratio=box.area_ratio,
        )

        record = _candidate_record(candidate, box, None)

        self.assertIsNone(record["gt_iou"])

    def test_raw_output_artifact_round_trips_losslessly(self) -> None:
        output = np.arange(36, dtype=np.float32).reshape(1, 6, 6)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw.npy"
            artifact = _write_raw_output_artifact(output, path)
            restored = np.load(path, allow_pickle=False)
            artifact_file_size = path.stat().st_size

        self.assertEqual(restored.dtype, output.dtype)
        self.assertEqual(restored.shape, output.shape)
        self.assertTrue(np.array_equal(restored, output))
        self.assertEqual(
            artifact["raw_output_sha256"],
            hashlib.sha256(output.tobytes(order="C")).hexdigest(),
        )
        self.assertEqual(artifact["file_bytes"], artifact_file_size)


if __name__ == "__main__":
    unittest.main()
