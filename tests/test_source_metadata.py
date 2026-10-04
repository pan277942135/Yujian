import pytest

from app.services.source_metadata import (
    NormalizedSourceMetadata,
    SourceMetadataError,
    normalize_source_metadata,
    validate_source_metadata,
)


GOOGLE_URL = "https://www.google.com.hk/imgres?q=草鱼&imgurl=example"


def test_normal_metadata_is_unchanged():
    result = normalize_source_metadata("google_images", GOOGLE_URL)

    assert result.source_platform == "google_images"
    assert result.source_url == GOOGLE_URL


def test_google_images_url_in_platform_is_recovered():
    result = normalize_source_metadata(GOOGLE_URL, None)

    assert result.source_platform == "google_images"
    assert result.source_url == GOOGLE_URL


def test_very_long_source_url_is_preserved_without_platform_truncation():
    source_url = "https://www.google.com.hk/imgres?q=草鱼&" + ("query=" + "x" * 4000)
    result = normalize_source_metadata("google_images", source_url)

    assert result.source_platform == "google_images"
    assert result.source_url == source_url
    assert len(result.source_url) > 128


def test_unknown_external_url_is_preserved_with_unknown_platform():
    source_url = "https://images.example.org/search/result?id=123"
    result = normalize_source_metadata(None, source_url)

    assert result.source_platform == "unknown"
    assert result.source_url == source_url


def test_null_url_and_valid_platform_are_accepted():
    result = normalize_source_metadata("manual_upload", None)

    assert result.source_platform == "manual_upload"
    assert result.source_url is None


def test_invalid_platform_fails_with_targeted_metadata_diagnostic():
    metadata = NormalizedSourceMetadata(source_platform="x" * 129, source_url=None)

    with pytest.raises(SourceMetadataError) as exc_info:
        validate_source_metadata(
            metadata,
            batch_id="BATCH_20261004_DB_XP_001",
            image_id="BATCH_EDP_M1_R01_grass_carp_001",
            file_name="草鱼_001.jpg",
        )

    error = exc_info.value
    assert error.code == "INVALID_SOURCE_METADATA"
    assert error.field == "source_platform"
    assert error.as_dict()["batch_id"] == "BATCH_20261004_DB_XP_001"
    assert error.as_dict()["image_id"] == "BATCH_EDP_M1_R01_grass_carp_001"


def test_conflicting_source_urls_stop_before_persistence():
    with pytest.raises(SourceMetadataError) as exc_info:
        normalize_source_metadata(
            GOOGLE_URL,
            "https://example.org/different",
            batch_id="BATCH_20261004_DB_XP_001",
            image_id="BATCH_EDP_M1_R01_grass_carp_001",
            file_name="草鱼_001.jpg",
        )

    assert exc_info.value.field == "source_platform"
    assert "INVALID_SOURCE_METADATA" in str(exc_info.value)
