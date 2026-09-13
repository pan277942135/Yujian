from app.main import FeedbackEvent, review_queue


def test_review_queue_has_feedback_event_dependency_imported():
    """The review queue enriches rows from the latest feedback event."""
    assert review_queue.__globals__["FeedbackEvent"] is FeedbackEvent
