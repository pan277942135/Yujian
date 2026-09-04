"""Conservative review truth suggestions.

The review UI may use trusted signals to reduce typing, but the returned value is
only a suggestion.  This module deliberately does not change review status and
never uses detector confidence as a species signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


AI_AGREEMENT_THRESHOLD = 0.80
SIGNAL_PREFIX = "[yujian_review_signal] "


@dataclass(frozen=True)
class ReviewPrefill:
    truth_species: str | None = None
    source: str | None = None
    ai_prediction: str | None = None
    ai_confidence: float | None = None
    conflict: bool = False
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _confidence(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def encode_review_signals(signals: dict[str, Any]) -> str:
    """Encode non-schema manifest review hints into the existing notes field."""

    clean = {key: value for key, value in signals.items() if value not in (None, "")}
    return SIGNAL_PREFIX + json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def parse_review_signals(notes: Any) -> dict[str, Any]:
    """Read an optional signal marker without treating ordinary notes as data."""

    text = _text(notes)
    for line in text.splitlines():
        if not line.startswith(SIGNAL_PREFIX):
            continue
        try:
            value = json.loads(line[len(SIGNAL_PREFIX) :])
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def trusted_truth_prefill(
    *,
    claimed_species: Any,
    species_check: Any = None,
    classifier_prediction: Any = None,
    classifier_confidence: Any = None,
    detector_confidence: Any = None,
) -> ReviewPrefill:
    """Return a conservative truth suggestion for one review item.

    A high-confidence classifier disagreement always wins over a collected
    ``confirmed`` marker: the reviewer must resolve that conflict explicitly.
    ``detector_confidence`` is accepted for call-site compatibility but is
    intentionally ignored.
    """

    del detector_confidence
    claimed = _text(claimed_species)
    prediction = _text(classifier_prediction)
    confidence = _confidence(classifier_confidence)
    confirmed = _text(species_check).lower() == "confirmed"
    high_confidence = confidence is not None and confidence >= AI_AGREEMENT_THRESHOLD

    if claimed and prediction and high_confidence and prediction != claimed:
        return ReviewPrefill(
            ai_prediction=prediction,
            ai_confidence=confidence,
            conflict=True,
            message="⚠ 标签冲突",
        )
    if claimed and prediction == claimed and high_confidence:
        return ReviewPrefill(
            truth_species=claimed,
            source="ai_agreement",
            ai_prediction=prediction,
            ai_confidence=confidence,
            message=f"AI一致 {round(confidence * 100):d}%",
        )
    if claimed and confirmed:
        return ReviewPrefill(
            truth_species=claimed,
            source="confirmed_label",
            ai_prediction=prediction or None,
            ai_confidence=confidence,
            message="采集标注已确认（仅预填）",
        )
    return ReviewPrefill(
        ai_prediction=prediction or None,
        ai_confidence=confidence,
    )


__all__ = [
    "AI_AGREEMENT_THRESHOLD",
    "ReviewPrefill",
    "SIGNAL_PREFIX",
    "encode_review_signals",
    "parse_review_signals",
    "trusted_truth_prefill",
]
