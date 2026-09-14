from __future__ import annotations

import json
import time

from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from app.db import Base
from app.models import Batch, DatasetVersion, ImageAsset
from app.platform.routes.pages import PLATFORM_PAGES, platform_queue_compatibility_redirect, templates
from app.platform.services import adapters


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'platform-data-factory.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []})


def test_data_factory_pages_use_platform_templates_and_empty_states():
    pages = {page.path: page for page in PLATFORM_PAGES}
    assert pages["/platform/data/import"].template == "platform/data_import.html"
    assert pages["/platform/data/datasets"].template == "platform/data_datasets.html"
    assert pages["/platform/data/review"].template == "platform/data_review.html"
    assert "/platform/data/queue" not in pages
    for path in ("/platform/data/import", "/platform/data/datasets", "/platform/data/review"):
        page = pages[path]
        rendered = templates.env.get_template(page.template).render(
            request=_request(path), page=page, page_title=page.title, platform_pages=PLATFORM_PAGES
        )
        assert page.title in rendered
        if path != "/platform/data/import":
            assert "暂无" in rendered


def test_data_factory_ui_keeps_import_entry_and_removes_queue_menu():
    rendered = templates.env.get_template("platform/sidebar.html").render(
        request=_request("/platform/data/review"), platform_pages=PLATFORM_PAGES
    )
    assert "导入采集数据" in rendered
    assert "数据审核中心" in rendered
    assert "数据处理队列" not in rendered

    review = templates.env.get_template("platform/data_review.html").render(
        request=_request("/platform/data/review"), page_title="数据审核中心", platform_pages=PLATFORM_PAGES
    )
    assert "高可信待确认" in review
    assert "选择鱼种" in review
    assert "prompt(" not in review
    assert "/crop-review?batch_id=" in review


def test_legacy_queue_path_redirects_into_review_center():
    response = platform_queue_compatibility_redirect()
    assert response.status_code == 307
    assert response.headers["location"] == "/platform/data/review"


def test_review_list_is_sql_paginated_and_prefetches_constant_relations(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(Batch(batch_id="BATCH_PERF", source="test", manifest_uri="/tmp/m", raw_uri="/tmp/r", image_count=120, status="INGESTED"))
        db.add_all(
            ImageAsset(
                batch_id="BATCH_PERF",
                image_id=f"image-{index:04d}",
                file_name=f"fish-{index:04d}.jpg",
                object_name=f"fish-{index:04d}.jpg",
                gcs_uri=f"gs://private/fish-{index:04d}.jpg",
                claimed_species="鲤鱼",
                review_status="pending",
                quality="GOOD",
            )
            for index in range(120)
        )
        db.commit()
        queries = []
        event.listen(db.bind, "before_cursor_execute", lambda *args: queries.append(args[2]))
        result = adapters.review_items(db, page=1, page_size=30)
        assert result["total"] == 120
        assert len(result["items"]) == 30
        assert result["items"][0]["image_id"] == "image-0000"
        assert len(queries) <= 5

        queries.clear()
        filtered = adapters.review_items(db, issue="bbox_error", page=2, page_size=30)
        assert filtered["total"] == 120
        assert len(filtered["items"]) == 30
        assert filtered["items"][0]["image_id"] == "image-0030"
        assert len(queries) <= 5
    finally:
        db.close()


def test_review_queue_and_dashboard_do_not_construct_every_review_item(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        db.add(Batch(batch_id="BATCH_QUEUE", source="test", manifest_uri="/tmp/m", raw_uri="/tmp/r", image_count=1, status="INGESTED"))
        db.add(ImageAsset(batch_id="BATCH_QUEUE", image_id="queue-1", file_name="q.jpg", object_name="q.jpg", gcs_uri="gs://q", review_status="pending", quality="GOOD"))
        db.commit()
        monkeypatch.setattr(adapters, "_review_item", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DTO construction must not run")))
        queries = []
        event.listen(db.bind, "before_cursor_execute", lambda *args: queries.append(args[2]))
        queue = adapters.review_queue(db)
        assert queue["pending"] == 1
        assert queue["low_confidence"] == 1
        assert len(queries) == 1
        monkeypatch.setattr(adapters, "review_queue", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dashboard must use lightweight counts")))
        assert adapters.dashboard(db)["review_queue"] == 1
    finally:
        db.close()


def test_platform_datasets_returns_only_dataset_versions(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(Batch(batch_id="BATCH_RAW_001", source="upload", manifest_uri="/tmp/m", raw_uri="/tmp/r", image_count=40, status="READY"))
        db.add(
            DatasetVersion(
                dataset_version="DS_M1_v0.7",
                manifest_uri="gs://private/datasets/DS_M1_v0.7/manifest.json",
                train_count=30,
                val_count=5,
                test_count=5,
                species_count=2,
                git_commit="a" * 40,
                status="FROZEN",
                pipeline_type="WHOLE_IMAGE_V1",
            )
        )
        db.commit()
        rows = adapters.datasets(db)
        assert [row["id"] for row in rows] == ["DS_M1_v0.7"]
        assert all(not row["id"].startswith("BATCH_") for row in rows)
        assert rows[0]["total"] == 40
        assert rows[0]["train"] == 30
        assert rows[0]["val"] == 5
        assert rows[0]["test"] == 5
        assert adapters.dataset_detail(db, "BATCH_RAW_001") is None
    finally:
        db.close()


def test_platform_component_partials_are_present():
    for name in (
        "platform/components/metric_card.html",
        "platform/components/status_tag.html",
    ):
        assert templates.env.get_template(name) is not None


def test_platform_endpoint_perf_evidence_3700_review_rows(tmp_path, monkeypatch):
    """Record endpoint-shaped query counts on a review pool similar to UAT."""
    db = _session(tmp_path)
    try:
        db.add(Batch(batch_id="BATCH_PERF_3700", source="test", manifest_uri="/tmp/m", raw_uri="/tmp/r", image_count=3700, status="INGESTED"))
        db.bulk_save_objects(
            [
                ImageAsset(
                    batch_id="BATCH_PERF_3700",
                    image_id=f"perf-{index:04d}",
                    file_name=f"fish-{index:04d}.jpg",
                    object_name=f"fish-{index:04d}.jpg",
                    gcs_uri=f"gs://private/perf-{index:04d}.jpg",
                    claimed_species="鲤鱼",
                    review_status="pending",
                    quality="GOOD",
                )
                for index in range(3700)
            ]
        )
        db.commit()

        def measure(callback):
            statements = []

            def before_cursor_execute(*args):
                statements.append(args[2])

            event.listen(db.bind, "before_cursor_execute", before_cursor_execute)
            started = time.perf_counter()
            result = callback()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            event.remove(db.bind, "before_cursor_execute", before_cursor_execute)
            return result, len(statements), elapsed_ms

        real_review_queue = adapters.review_queue
        real_review_item = adapters._review_item
        monkeypatch.setattr(adapters, "review_queue", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dashboard must not call full review_queue")))
        dashboard, dashboard_queries, dashboard_ms = measure(lambda: adapters.dashboard(db))

        monkeypatch.setattr(adapters, "review_queue", real_review_queue)
        monkeypatch.setattr(adapters, "_review_item", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("queue must not construct every review DTO")))
        queue, queue_queries, queue_ms = measure(lambda: adapters.review_queue(db))

        monkeypatch.setattr(adapters, "_review_item", real_review_item)
        items, item_queries, items_ms = measure(lambda: adapters.review_items(db, page=1, page_size=30))

        assert dashboard["review_queue"] == 3700
        assert queue["pending"] == 3700
        assert items["total"] == 3700
        assert len(items["items"]) == 30
        assert dashboard_queries < 20
        assert queue_queries == 1
        assert item_queries <= 5
        print(
            "PERF_EVIDENCE "
            + json.dumps(
                {
                    "/api/platform/dashboard": {"queries": dashboard_queries, "latency_ms": dashboard_ms},
                    "/api/platform/review/items?page=1&page_size=30": {"queries": item_queries, "latency_ms": items_ms, "total": items["total"]},
                    "/api/platform/review/queue": {"queries": queue_queries, "latency_ms": queue_ms, "pending": queue["pending"]},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    finally:
        db.close()
