"""Tests for the frozen Dataset release QA gate."""

from app.platform.services.crop_dataset import select_random_50_qa_rows


def _row(image_id, quality_status, split=""):
    return {"image_id": image_id, "batch_id": "batch", "quality_status": quality_status, "split": split}


def _rows():
    return (
        [_row(f"good-train-{i}", "GOOD", "train") for i in range(40)]
        + [_row(f"good-val-{i}", "GOOD", "val") for i in range(10)]
        + [_row(f"good-test-{i}", "GOOD", "test") for i in range(10)]
        + [_row(f"warning-{i}", "WARNING") for i in range(20)]
        + [_row(f"invalid-{i}", "INVALID") for i in range(20)]
    )


def test_random_50_qa_is_stratified_and_deterministic():
    first = select_random_50_qa_rows(_rows(), "DS_CROP_M1_v0.1")
    second = select_random_50_qa_rows(_rows(), "DS_CROP_M1_v0.1")
    assert [row["image_id"] for row in first] == [row["image_id"] for row in second]
    assert len({row["image_id"] for row in first}) == 50
    assert sum(row["quality_status"] == "GOOD" for row in first) == 30
    assert sum(row["quality_status"] == "WARNING" for row in first) == 10
    assert sum(row["quality_status"] == "INVALID" for row in first) == 10
    assert sum(row["quality_status"] == "GOOD" and row["split"] == "train" for row in first) == 20
    assert sum(row["quality_status"] == "GOOD" and row["split"] == "val" for row in first) == 5
    assert sum(row["quality_status"] == "GOOD" and row["split"] == "test" for row in first) == 5


def test_random_50_qa_rejects_insufficient_stratum():
    try:
        select_random_50_qa_rows(_rows()[:10], "DS_CROP_M1_v0.1")
    except ValueError as exc:
        assert "RANDOM_50_QA_INSUFFICIENT" in str(exc)
    else:
        raise AssertionError("expected insufficient QA stratum error")
