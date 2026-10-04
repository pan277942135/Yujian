from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


SOURCE_PLATFORM_MAX_LENGTH = 128
UNKNOWN_SOURCE_PLATFORM = "unknown"
_UNRESOLVED_PLATFORM_NAMES = {"", "unknown", "external"}


class SourceMetadataError(ValueError):
    """Invalid source metadata that must be stopped before an ImageAsset write."""

    code = "INVALID_SOURCE_METADATA"

    def __init__(
        self,
        *,
        batch_id: str | None,
        image_id: str | None,
        file_name: str | None,
        field: str,
        reason: str,
    ):
        self.batch_id = batch_id
        self.image_id = image_id
        self.file_name = file_name
        self.field = field
        self.reason = reason
        super().__init__(
            "INVALID_SOURCE_METADATA "
            f"batch_id={batch_id or 'unknown'} "
            f"image_id={image_id or 'unknown'} "
            f"field={field} reason={reason}"
        )

    def as_dict(self) -> dict[str, str]:
        payload = {
            "error": self.code,
            "detail": "来源元数据格式异常",
            "field": self.field,
        }
        if self.batch_id:
            payload["batch_id"] = self.batch_id
        if self.image_id:
            payload["image_id"] = self.image_id
        if self.file_name:
            payload["file_name"] = self.file_name
        return payload


@dataclass(frozen=True)
class NormalizedSourceMetadata:
    source_platform: str
    source_url: str | None


def _clean(value: object | None) -> str:
    return "" if value is None else str(value).strip()


def _is_http_url(value: str) -> bool:
    return value.casefold().startswith(("http://", "https://"))


def infer_source_platform(source_url: str | None) -> str | None:
    """Infer only collection platforms with a stable canonical identifier."""

    value = _clean(source_url)
    if not _is_http_url(value):
        return None
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if host == "google.com" or host.endswith(".google.com") or host == "google.com.hk" or host.endswith(".google.com.hk"):
        return "google_images"
    if host == "bing.com" or host.endswith(".bing.com"):
        return "bing_images"
    if host == "baidu.com" or host.endswith(".baidu.com"):
        return "baidu_images"
    return None


def normalize_source_metadata(
    source_platform: object | None,
    source_url: object | None,
    *,
    batch_id: str | None = None,
    image_id: str | None = None,
    file_name: str | None = None,
) -> NormalizedSourceMetadata:
    """Recover misplaced URLs and return a bounded platform identifier plus full URL."""

    platform = _clean(source_platform)
    url = _clean(source_url) or None

    if _is_http_url(platform):
        if url and url != platform:
            raise SourceMetadataError(
                batch_id=batch_id,
                image_id=image_id,
                file_name=file_name,
                field="source_platform",
                reason="URL is stored in source_platform and conflicts with source_url",
            )
        url = url or platform
        platform = infer_source_platform(url) or UNKNOWN_SOURCE_PLATFORM
    elif platform.casefold() in _UNRESOLVED_PLATFORM_NAMES:
        platform = infer_source_platform(url) or UNKNOWN_SOURCE_PLATFORM

    return NormalizedSourceMetadata(source_platform=platform, source_url=url)


def validate_source_metadata(
    metadata: NormalizedSourceMetadata,
    *,
    batch_id: str | None,
    image_id: str | None,
    file_name: str | None,
) -> None:
    """Enforce the image_assets.source_platform VARCHAR and semantic contract."""

    platform = metadata.source_platform
    if _is_http_url(platform):
        reason = "must be a canonical platform identifier, not a URL"
    elif len(platform) > SOURCE_PLATFORM_MAX_LENGTH:
        reason = f"exceeds the {SOURCE_PLATFORM_MAX_LENGTH}-character platform limit"
    else:
        return
    raise SourceMetadataError(
        batch_id=batch_id,
        image_id=image_id,
        file_name=file_name,
        field="source_platform",
        reason=reason,
    )
