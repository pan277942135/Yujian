from pathlib import Path


TEMPLATE = Path("app/templates/training.html")


def test_legacy_training_page_exposes_model_publish_action():
    content = TEMPLATE.read_text(encoding="utf-8")

    assert "发布模型" in content
    assert "openLegacyPublish" in content
    assert "/api/models/'+encodeURIComponent(context.model)+'" in content
    assert "/publish/status" in content
    assert "fish_classifier_v0_2.tflite" in content


def test_legacy_training_page_keeps_metrics_action():
    content = TEMPLATE.read_text(encoding="utf-8")

    assert "看指标" in content
    assert "publishAction(x)" in content
    assert "data-publish-model" in content
