from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from app.db import Base
from app.entry import app
from app.models import Batch, DatasetVersion, ImageAsset
from app.batch_upload_api import UploadFinalizeRequest, UploadStartRequest
from app.platform.routes.api import platform_dataset_manifest
from app.platform.routes.pages import (
    PLATFORM_PAGES,
    PlatformPage,
    platform_dataset_detail_page,
    platform_queue_compatibility_redirect,
    templates,
)
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
        payload = adapters.datasets(db)
        rows = payload["datasets"]
        assert [row["id"] for row in rows] == ["DS_M1_v0.7"]
        assert all(not row["id"].startswith("BATCH_") for row in rows)
        assert rows[0]["total"] == 40
        assert rows[0]["train"] == 30
        assert rows[0]["val"] == 5
        assert rows[0]["test"] == 5
        assert payload["summary"]["dataset_count"] == 1
        assert payload["summary"]["batch_count"] == 1
        assert payload["summary"]["total_images"] == 40
        assert payload["recent_batches"][0]["batch_id"] == "BATCH_RAW_001"
        assert adapters.dataset_detail(db, "BATCH_RAW_001") is None
    finally:
        db.close()


def test_platform_dataset_api_normalizes_version_fields_and_source_batches(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(Batch(batch_id="BATCH_RAW_002", source="upload", manifest_uri="/tmp/m", raw_uri="/tmp/r", image_count=40, status="READY"))
        db.add_all(
            [
                DatasetVersion(
                    dataset_version="DS_M1_v0.6",
                    manifest_uri="/tmp/ds-v06.csv",
                    train_count=3577,
                    val_count=1000,
                    test_count=520,
                    git_commit="b" * 40,
                    status="FROZEN",
                    pipeline_type="WHOLE_IMAGE_V1",
                    created_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
                ),
                DatasetVersion(
                    dataset_version="DS_M1_v0.7",
                    parent_version="DS_M1_v0.6",
                    manifest_uri="/tmp/ds-v07.csv",
                    train_count=1483,
                    val_count=314,
                    test_count=320,
                    git_commit="c" * 40,
                    status="FROZEN",
                    pipeline_type="WHOLE_IMAGE_V1",
                    metadata_json=json.dumps(
                        {
                            "source_batch_id": "BATCH_20260913_DB_002",
                            "source_batch": ["BATCH_20260905_07", "BATCH_20260913_DB_002"],
                        },
                        ensure_ascii=False,
                    ),
                    created_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
                ),
            ]
        )
        db.commit()

        payload = adapters.datasets(db)
        rows = payload["datasets"]

        assert [row["id"] for row in rows] == ["DS_M1_v0.7", "DS_M1_v0.6"]
        assert all(not row["id"].startswith("BATCH_") for row in rows)
        assert set(rows[0]) == {
            "id",
            "type",
            "total",
            "train",
            "val",
            "test",
            "status",
            "source_batches",
            "parent_version",
            "created_at",
        }
        assert rows[0]["total"] == 2117
        assert rows[0]["train"] == 1483
        assert rows[0]["val"] == 314
        assert rows[0]["test"] == 320
        assert rows[0]["source_batches"] == ["BATCH_20260913_DB_002", "BATCH_20260905_07"]
        assert rows[0]["parent_version"] == "DS_M1_v0.6"
        assert "source_batch" not in rows[0]
        assert "name" not in rows[0]
        assert "pipeline_type" not in rows[0]
    finally:
        db.close()


def test_platform_dataset_detail_exposes_counts_lineage_and_clean_report(tmp_path):
    db = _session(tmp_path)
    try:
        db.add_all(
            [
                DatasetVersion(
                    dataset_version="DS_M1_v0.6",
                    manifest_uri="/tmp/ds-v06.csv",
                    train_count=10,
                    val_count=2,
                    test_count=1,
                    git_commit="b" * 40,
                    status="FROZEN",
                ),
                DatasetVersion(
                    dataset_version="DS_M1_v0.7",
                    parent_version="DS_M1_v0.6",
                    manifest_uri="/tmp/ds-v07.csv",
                    train_count=1483,
                    val_count=314,
                    test_count=320,
                    git_commit="c" * 40,
                    status="FROZEN",
                    metadata_json=json.dumps(
                        {
                            "source_batches": ["BATCH_20260913_DB_002"],
                            "clean_report": {"duplicate": 4, "blur": 2, "no_fish": 1},
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        db.commit()

        detail = adapters.dataset_detail(db, "DS_M1_v0.7")

        assert detail["id"] == "DS_M1_v0.7"
        assert detail["type"] == "WHOLE_IMAGE_V1"
        assert detail["status"] == "FROZEN"
        assert detail["counts"] == {"total": 2117, "train": 1483, "val": 314, "test": 320}
        assert detail["source_batches"] == ["BATCH_20260913_DB_002"]
        assert detail["parent_version"] == "DS_M1_v0.6"
        assert detail["version_chain"] == ["DS_M1_v0.6", "DS_M1_v0.7"]
        assert detail["clean_report"] == {"blur": 2, "duplicate": 4, "no_fish": 1, "multi_fish": 0, "scene": 0}
        assert adapters.dataset_detail(db, "BATCH_20260913_DB_002") is None
    finally:
        db.close()


def test_platform_dataset_list_uses_one_dataset_version_query(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_QUERY_1",
                manifest_uri="/tmp/ds.csv",
                train_count=1,
                val_count=1,
                test_count=1,
                git_commit="d" * 40,
                status="FROZEN",
            )
        )
        db.commit()
        monkeypatch.setattr(
            adapters,
            "_dataset_counts",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("list must not load DatasetItem/ImageAsset details")),
        )
        queries = []
        event.listen(db.bind, "before_cursor_execute", lambda *args: queries.append(args[2]))

        payload = adapters.datasets(db)
        rows = payload["datasets"]

        assert len(rows) == 1
        assert rows[0]["total"] == 3
        assert payload["recent_batches"] == []
        assert len(queries) <= 5
    finally:
        db.close()


def test_platform_dataset_detail_page_and_manifest_endpoint(tmp_path):
    db = _session(tmp_path)
    manifest = tmp_path / "dataset-manifest.csv"
    manifest.write_text("image_id\tspecies\nimage-1\t鲤鱼\n", encoding="utf-8")
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_MANIFEST_1",
                manifest_uri=str(manifest),
                train_count=1,
                val_count=0,
                test_count=0,
                git_commit="e" * 40,
                status="FROZEN",
            )
        )
        db.commit()

        response = platform_dataset_detail_page(_request("/platform/data/datasets/DS_MANIFEST_1"), "DS_MANIFEST_1")
        assert response.template.name == "platform/data_dataset_detail.html"
        assert "/platform/data/datasets/{dataset_id}" in app.openapi()["paths"]
        assert "/api/platform/datasets/{dataset_id}/manifest" in app.openapi()["paths"]
        manifest_response = platform_dataset_manifest("DS_MANIFEST_1", db)
        assert manifest_response.status_code == 200
        assert manifest_response.media_type == "text/csv"
        assert manifest_response.body.startswith(b"image_id")
        assert "attachment" in manifest_response.headers["content-disposition"]
    finally:
        db.close()


def test_dataset_templates_separate_detail_and_clean_report_actions():
    dataset_page = next(page for page in PLATFORM_PAGES if page.path == "/platform/data/datasets")
    rendered = templates.env.get_template(dataset_page.template).render(
        request=_request(dataset_page.path), page=dataset_page, page_title=dataset_page.title, platform_pages=PLATFORM_PAGES
    )
    detail_page = PlatformPage(
        "/platform/data/datasets/{dataset_id}",
        "platform/data_dataset_detail.html",
        "数据集详情",
        "AI 数据工厂",
        "详情",
        "/api/platform/datasets/{dataset_id}",
    )
    detail_rendered = templates.env.get_template(detail_page.template).render(
        request=_request("/platform/data/datasets/DS_M1_v0.7"),
        page=detail_page,
        page_title=detail_page.title,
        platform_pages=PLATFORM_PAGES,
    )

    assert "数据集版本" in rendered
    assert "数据批次" in rendered
    assert "累计图片" in rendered
    assert "待审核" in rendered
    assert "查看详情" in rendered
    assert "清洗报告" in rendered
    assert "row.name" in rendered
    assert "row.source_batch||" not in rendered
    assert "暂无完整关联信息" not in rendered
    assert "/api/platform/datasets/${encodeURIComponent(id)}/clean-report" in rendered
    assert "datasetDetailTitle" in detail_rendered
    assert "datasetId" in detail_rendered
    assert "总样本" in detail_rendered
    assert "来源数据" in detail_rendered
    assert "版本链" in detail_rendered
    assert "导出 Manifest" in detail_rendered
    assert "生成训练任务" in detail_rendered


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


def test_dataset_page_payload_separates_recent_batch_review_stats(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(
            Batch(
                batch_id="BATCH_RECENT_001",
                source="real_collection",
                manifest_uri="/tmp/manifest.csv",
                raw_uri="/tmp/raw",
                image_count=7,
                status="REGISTERED",
                notes="2026 秋季真实鱼获采集-01",
            )
        )
        db.add_all(
            [
                ImageAsset(
                    batch_id="BATCH_RECENT_001",
                    image_id="approved-1",
                    file_name="approved.jpg",
                    object_name="approved.jpg",
                    gcs_uri="gs://private/approved.jpg",
                    review_status="approved",
                ),
                ImageAsset(
                    batch_id="BATCH_RECENT_001",
                    image_id="pending-1",
                    file_name="pending.jpg",
                    object_name="pending.jpg",
                    gcs_uri="gs://private/pending.jpg",
                    review_status="pending",
                ),
            ]
        )
        db.commit()

        payload = adapters.datasets(db)

        assert [row["id"] for row in payload["datasets"]] == []
        assert payload["summary"] == {
            "dataset_count": 0,
            "batch_count": 1,
            "total_images": 7,
            "pending_review": 1,
        }
        assert payload["recent_batches"] == [
            {
                "batch_id": "BATCH_RECENT_001",
                "name": "2026 秋季真实鱼获采集-01",
                "source": "real_collection",
                "image_count": 7,
                "ai_valid": 1,
                "pending_review": 1,
                "status": "REGISTERED",
                "created_at": payload["recent_batches"][0]["created_at"],
            }
        ]
    finally:
        db.close()


def test_platform_import_page_reuses_existing_batch_upload_endpoints():
    page = next(page for page in PLATFORM_PAGES if page.path == "/platform/data/import")
    rendered = templates.env.get_template(page.template).render(
        request=_request(page.path), page=page, page_title=page.title, platform_pages=PLATFORM_PAGES
    )

    assert "创建数据批次" in rendered
    assert "导入真实鱼获采集图片" in rendered
    assert "批次名称" in rendered
    assert "/api/batches/upload-start" in rendered
    assert "/api/batches/upload-file" in rendered
    assert "/api/batches/upload-finalize" in rendered
    assert "/api/batches/upload" in rendered
    assert "新建数据集" not in rendered
    assert "iframe" not in rendered.lower()


def test_existing_batch_upload_contract_accepts_optional_platform_batch_name():
    start = UploadStartRequest(batch_name="2026 秋季真实鱼获采集-01", source="real_collection")
    finalize = UploadFinalizeRequest(
        batch_id="BATCH_20260915_TEST_001",
        batch_name="2026 秋季真实鱼获采集-01",
        source="real_collection",
    )
    assert start.batch_name == finalize.batch_name
    assert start.source == finalize.source
