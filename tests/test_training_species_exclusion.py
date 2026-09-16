from __future__ import annotations

from app.species_policy import is_training_excluded_species, training_eligibility


def test_other_freshwater_labels_are_never_training_classes():
    assert is_training_excluded_species(species_key="other_freshwater_fish")
    assert is_training_excluded_species(common_name_zh="其他淡水鱼")
    assert is_training_excluded_species(common_name_zh="其他淡水")
    assert is_training_excluded_species(common_name_en="Other freshwater fish")
    assert not is_training_excluded_species(species_key="grass_carp", common_name_zh="草鱼")


def test_training_gate_reports_catch_all_as_disabled():
    enabled, reasons = training_eligibility(
        {"total": 100, "group_count": 5, "train": 70, "val": 15, "test": 15},
        is_other=True,
    )
    assert enabled is False
    assert "其他淡水鱼为兜底标签，默认不作为训练类别" in reasons
