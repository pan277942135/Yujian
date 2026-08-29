#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from app import bulk_review, inference_api, inspect, main, training_api  # noqa: E402
from app.entry import app  # noqa: F401,E402
from app.unified_nav import UnifiedNavLoader  # noqa: E402


EXPECTED_LINKS = [
    ('href="/"', "总览"),
    ('href="/batches"', "数据批次"),
    ('href="/review/bulk"', "快速审核"),
    ('href="/review"', "单张审核"),
    ('href="/species"', "鱼种管理"),
    ('href="/feedback"', "用户反馈"),
    ('href="/datasets"', "数据集"),
    ('href="/training"', "模型训练"),
    ('href="/inference"', "模型实测"),
]

TEMPLATE_ENGINES = [
    main.templates,
    bulk_review.templates,
    inspect.templates,
    training_api.templates,
    inference_api.templates,
]

TEMPLATES = [
    "overview.html",
    "batches.html",
    "bulk_review.html",
    "review.html",
    "species.html",
    "feedback.html",
    "datasets.html",
    "training.html",
    "inference.html",
    "inspect.html",
]


def main_test() -> None:
    for engine in TEMPLATE_ENGINES:
        assert isinstance(engine.env.loader, UnifiedNavLoader), type(engine.env.loader)

    loader = main.templates.env.loader
    for template_name in TEMPLATES:
        source, _filename, _uptodate = loader.get_source(main.templates.env, template_name)
        assert source.count('class="app-nav"') == 1, template_name
        positions = []
        for href, label in EXPECTED_LINKS:
            token = f"{href}"
            assert token in source, (template_name, href)
            assert label in source, (template_name, label)
            positions.append(source.index(token))
        assert positions == sorted(positions), (template_name, positions)
        assert "aria-current=\"page\"" in source, template_name

    print("Unified top navigation smoke OK", len(TEMPLATES), "templates", len(EXPECTED_LINKS), "items")


if __name__ == "__main__":
    main_test()
