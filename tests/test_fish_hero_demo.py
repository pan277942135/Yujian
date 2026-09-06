from app.segmentation_api import FishHeroReviewRequest, _expanded_crop_box, _orientation
from app.segmentation_api import templates


class Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def normalized(self):
        return self


def test_expanded_crop_stays_inside_source():
    assert _expanded_crop_box(Box(0.01, 0.02, 0.30, 0.40), 1000, 800) == (0, 0, 335, 367)


def test_orientation_uses_normalized_source_dimensions():
    from PIL import Image

    assert _orientation(Image.new("RGB", (1000, 700))) == "landscape"
    assert _orientation(Image.new("RGB", (700, 1000))) == "portrait"


def test_review_contract_accepts_multiple_issue_tags_and_notes():
    payload = FishHeroReviewRequest(
        test_id="FH_20260906_001",
        image_id="abc123",
        timestamp="2026-09-06T20:20:31+08:00",
        human={
            "source_hero_capable": "YES",
            "transparent_hero_quality": "HERO_OK",
            "visual_lift_vs_smart_crop": "BETTER",
            "issues": ["EDGE_ARTIFACT", "HAND_RESIDUE"],
            "reviewer_notes": "手机尺寸下边缘问题不明显。",
        },
    )
    assert payload.human["issues"] == ["EDGE_ARTIFACT", "HAND_RESIDUE"]
    assert len(payload.human["reviewer_notes"]) <= 500


def test_template_is_available():
    assert templates is not None
