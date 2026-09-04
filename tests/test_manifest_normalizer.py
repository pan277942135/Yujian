import csv
import io

import pytest

from app.services.manifest_normalizer import (
    ManifestNormalizationError,
    normalize_manifest,
)


def write_manifest(root, name, rows, fieldnames):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue(), encoding="utf-8")
    return path


def test_existing_fish_manifest_is_validated_and_not_replaced(tmp_path):
    existing = write_manifest(
        tmp_path,
        "metadata/fish_manifest.csv",
        [{
            "image_path": "images/manual.jpg",
            "image_id": "MANUAL_001",
            "claimed_species": "鳙鱼",
            "species_key": "bighead_carp",
            "source": "manual",
        }],
        ["image_path", "image_id", "claimed_species", "species_key", "source"],
    )
    before = existing.read_text(encoding="utf-8")

    result = normalize_manifest(tmp_path)

    assert result.generated is False
    assert result.rows == 1
    assert result.output_path == existing
    assert existing.read_text(encoding="utf-8") == before


def test_asset_manifest_is_converted_to_fixed_training_contract(tmp_path):
    write_manifest(
        tmp_path,
        "metadata/manifest.csv",
        [{
            "image_id": "P5_00001",
            "file_name": "bighead_carp_0001.jpg",
            "species_name": "鳙鱼",
            "species_key": "bighead_carp",
            "source_platform": "image_search",
        }],
        ["image_id", "file_name", "species_name", "species_key", "source_platform"],
    )

    result = normalize_manifest(tmp_path)
    rows = list(csv.DictReader(result.output_path.open(encoding="utf-8", newline="")))

    assert result.generated is True
    assert result.rows == 1
    assert result.output_path == tmp_path / "metadata/fish_manifest.csv"
    assert rows == [{
        "image_path": "images/bighead_carp_0001.jpg",
        "image_id": "P5_00001",
        "claimed_species": "鳙鱼",
        "species_key": "bighead_carp",
        "source": "image_search",
    }]


def test_p5_manifest_generates_all_525_rows(tmp_path):
    rows = [
        {
            "image_id": f"P5_{index:05d}",
            "file_name": f"p5_{index:05d}.jpg",
            "species_name": "鳙鱼" if index % 2 else "草鱼",
            "species_key": "bighead_carp" if index % 2 else "grass_carp",
            "source_platform": "P5",
        }
        for index in range(1, 526)
    ]
    write_manifest(
        tmp_path,
        "metadata/manifest.csv",
        rows,
        ["image_id", "file_name", "species_name", "species_key", "source_platform"],
    )

    result = normalize_manifest(tmp_path)

    assert result.rows == 525
    assert len(result.output_path.read_text(encoding="utf-8").splitlines()) == 526


def test_doubao_manifest_aliases_generate_all_900_rows(tmp_path):
    rows = [
        {
            "image_id": f"DB_{index:05d}",
            "filename": f"doubao/{index:05d}.jpg",
            "fish_name": "鲫鱼",
            "category_key": "crucian_carp",
            "dataset_source": "doubao",
        }
        for index in range(1, 901)
    ]
    write_manifest(
        tmp_path,
        "metadata/manifest.csv",
        rows,
        ["image_id", "filename", "fish_name", "category_key", "dataset_source"],
    )

    result = normalize_manifest(tmp_path)

    assert result.rows == 900
    first = next(csv.DictReader(result.output_path.open(encoding="utf-8", newline="")))
    assert first["image_path"] == "doubao/00001.jpg"
    assert first["claimed_species"] == "鲫鱼"
    assert first["species_key"] == "crucian_carp"
    assert first["source"] == "doubao"


def test_missing_species_is_a_manifest_invalid_error(tmp_path):
    write_manifest(
        tmp_path,
        "metadata/manifest.csv",
        [{"image_id": "BAD_001", "file_name": "bad.jpg"}],
        ["image_id", "file_name"],
    )

    with pytest.raises(ManifestNormalizationError) as exc_info:
        normalize_manifest(tmp_path)

    assert exc_info.value.code == "MANIFEST_INVALID"
    assert exc_info.value.reason == "missing species field"
