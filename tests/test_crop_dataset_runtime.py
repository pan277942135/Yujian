from app.platform.services.crop_dataset import (
    CROP_EXPAND_RATIO,
    CROP_OUTPUT_SIZE,
    evaluate_quality,
    validate_crop_split_summary,
)


def _row(image_id, batch_id, species="鲫鱼", status="GOOD", split=""):
    return {
        "image_id": image_id,
        "batch_id": batch_id,
        "crop_path": "images/x.jpg" if status != "INVALID" else "",
        "species": species,
        "source_image": f"gs://bucket/{batch_id}/{image_id}.jpg",
        "bbox": "[0.2,0.2,0.4,0.4]",
        "pixel_bbox": "[20,20,40,40]",
        "source_size": "[100,100]",
        "expand_ratio": "1.25",
        "fish_bbox_ratio": "0.64",
        "crop_clipped": "false",
        "quality_status": status,
        "quality_reason": "",
        "split": split,
    }


def test_invalid_bbox_is_invalid():
    status, reason = evaluate_quality(
        box=None, species="鲫鱼", presence_status="ok", fish_count=1,
        clipped=False, crop_ok=False, bbox_area_ratio=None,
    )
    assert status == "INVALID"
    assert "accepted_bbox_invalid" in reason


def test_missing_species_is_invalid():
    status, reason = evaluate_quality(
        box=[0.1, 0.1, 0.2, 0.2], species="", presence_status="ok", fish_count=1,
        clipped=False, crop_ok=True, bbox_area_ratio=0.64,
    )
    assert status == "INVALID"
    assert "species_missing" in reason


def test_edge_clip_is_warning():
    status, reason = evaluate_quality(
        box=[0.0, 0.1, 0.2, 0.2], species="鲫鱼", presence_status="ok", fish_count=1,
        clipped=True, crop_ok=True, bbox_area_ratio=0.64,
    )
    assert status == "WARNING"
    assert "edge" in reason


def test_real_split_excludes_warning_and_invalid():
    rows = [_row(f"img-{i}", f"batch-{i}") for i in range(30)]
    rows += [_row("warning", "batch-warning", status="WARNING")]
    rows += [_row("invalid", "batch-invalid", status="INVALID")]
    result = validate_crop_split_summary(rows)
    assert result["quality_sum_check"] is True
    assert result["split_sum_check"] is True
    assert sum(result["counts"].values()) == 30
    assert rows[-2]["split"] == ""
    assert rows[-1]["split"] == ""


def test_split_is_deterministic():
    first = [_row(f"img-{i}", f"batch-{i}") for i in range(60)]
    second = [_row(f"img-{i}", f"batch-{i}") for i in range(60)]
    validate_crop_split_summary(first)
    validate_crop_split_summary(second)
    assert [row["split"] for row in first] == [row["split"] for row in second]


def test_source_group_never_crosses_split():
    rows = [_row("same", "batch-1"), _row("same", "batch-1")]
    validate_crop_split_summary(rows)
    assert len({row["split"] for row in rows}) == 1


def test_contract_constants():
    assert CROP_EXPAND_RATIO == 1.25
    assert CROP_OUTPUT_SIZE == 416
