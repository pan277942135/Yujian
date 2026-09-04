from app.services.review_prefill import (
    encode_review_signals,
    parse_review_signals,
    trusted_truth_prefill,
)


def test_confirmed_collected_label_prefills_only_truth():
    result = trusted_truth_prefill(claimed_species="鳙鱼", species_check="confirmed")

    assert result.truth_species == "鳙鱼"
    assert result.source == "confirmed_label"
    assert result.conflict is False


def test_high_confidence_classifier_agreement_prefills_truth():
    result = trusted_truth_prefill(
        claimed_species="草鱼",
        classifier_prediction="草鱼",
        classifier_confidence=0.87,
    )

    assert result.truth_species == "草鱼"
    assert result.source == "ai_agreement"
    assert result.message == "AI一致 87%"


def test_high_confidence_classifier_conflict_stays_unconfirmed():
    result = trusted_truth_prefill(
        claimed_species="草鱼",
        species_check="confirmed",
        classifier_prediction="鲫鱼",
        classifier_confidence=0.92,
    )

    assert result.truth_species is None
    assert result.conflict is True
    assert result.message == "⚠ 标签冲突"


def test_detector_confidence_never_prefills_truth():
    result = trusted_truth_prefill(
        claimed_species="鲤鱼",
        detector_confidence=0.99,
    )

    assert result.truth_species is None
    assert result.ai_confidence is None


def test_manifest_review_signals_round_trip_through_existing_notes_field():
    marker = encode_review_signals(
        {"species_check": "confirmed", "classifier_prediction": "鳙鱼", "classifier_confidence": "0.87"}
    )

    assert parse_review_signals(f"人工备注\n{marker}") == {
        "species_check": "confirmed",
        "classifier_prediction": "鳙鱼",
        "classifier_confidence": "0.87",
    }
