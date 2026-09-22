from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile

from app.db import Base
from app.platform.models import (
    BsideBackground,
    BsideBackgroundOutlineProfile,
    BsideOutlineStyle,
    BsideVisualSession,
)
from app.platform.routes.bside_assets import (
    BsideBackgroundCreate,
    BsideBackgroundPatch,
    create_bside_background,
    update_bside_background,
    upload_bside_background_asset,
)
from app.platform.services.bside_assets import (
    B_SIDE_CANVAS_V1,
    BsideAssetError,
    normalize_bside_asset,
    seed_bside_asset_registry,
)
from app.platform.services.bside_visual.asset_registry import (
    get_active_bside_backgrounds,
    get_bside_style_plan,
)


def _db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'bside-assets.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _image_bytes(size=(1080, 1350), mode="RGB", image_format="PNG"):
    image = Image.new(mode, size, (35, 95, 92, 255) if mode == "RGBA" else (35, 95, 92))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_bside_asset_seed_has_three_tables_three_backgrounds_three_outlines_and_nine_profiles(tmp_path):
    db = _db(tmp_path)
    try:
        seed_bside_asset_registry(db)
        assert inspect(db.bind).has_table("bside_background")
        assert inspect(db.bind).has_table("bside_outline_style")
        assert inspect(db.bind).has_table("bside_background_outline_profile")
        assert db.query(BsideBackground).count() == 3
        assert db.query(BsideOutlineStyle).count() == 3
        assert db.query(BsideBackgroundOutlineProfile).count() == 9
        lake = db.scalar(select(BsideBackground).where(BsideBackground.code == "lake_dawn_01"))
        assert lake is not None
        assert lake.status == "DRAFT"
        assert lake.fish_anchor_x == pytest.approx(0.50)
        assert lake.fish_anchor_y == pytest.approx(0.52)
        weights = {
            style.code: profile.weight
            for profile in db.query(BsideBackgroundOutlineProfile).filter_by(background_id=lake.id).all()
            for style in [db.get(BsideOutlineStyle, profile.outline_style_id)]
        }
        assert weights == {"directional_rim": 50, "bottom_water_glow": 40, "none": 10}
    finally:
        db.close()


def test_bside_upload_normalization_keeps_canvas_and_requires_alpha_for_optional_layers():
    background = normalize_bside_asset(_image_bytes(), "background")
    assert background["content_type"] == "image/webp"
    assert background["filename"] == "background.webp"
    assert background["preview_data"]
    with Image.open(io.BytesIO(background["data"])) as image:
        assert image.size == (1080, 1350)
        assert image.format == "WEBP"

    with pytest.raises(BsideAssetError) as error:
        normalize_bside_asset(_image_bytes(mode="RGB"), "foreground")
    assert error.value.code == "ALPHA_REQUIRED"

    with pytest.raises(BsideAssetError) as error:
        normalize_bside_asset(_image_bytes(size=(768, 1024)), "background")
    assert error.value.code == "CANVAS_DIMENSION_MISMATCH"
    assert "1080×1350" in str(error.value)

    light = normalize_bside_asset(_image_bytes(mode="RGBA"), "light")
    assert light["content_type"] == "image/png"
    with Image.open(io.BytesIO(light["data"])) as image:
        assert image.mode == "RGBA"


def test_new_background_gets_configurable_profiles_without_renderer_code_changes(tmp_path):
    db = _db(tmp_path)
    try:
        seed_bside_asset_registry(db)
        row = BsideBackground(
            code="test_background_04",
            name="测试背景 04",
            description="扩展性验证",
            fish_anchor_x=0.5,
            fish_anchor_y=0.5,
            fish_width_min=0.68,
            fish_width_max=0.74,
            status="DRAFT",
        )
        db.add(row)
        db.flush()
        from app.platform.services.bside_assets import ensure_background_profiles

        ensure_background_profiles(db, row)
        db.commit()
        assert db.query(BsideBackgroundOutlineProfile).filter_by(background_id=row.id).count() == 3
        assert {item.code for item in db.query(BsideOutlineStyle).all()} >= {
            "none",
            "directional_rim",
            "bottom_water_glow",
        }
    finally:
        db.close()


def test_style_plan_is_persisted_and_reused_after_refresh(tmp_path):
    db = _db(tmp_path)
    try:
        seed_bside_asset_registry(db)
        background = db.scalar(select(BsideBackground).where(BsideBackground.code == "lake_dawn_01"))
        assert background is not None
        source = tmp_path / "background.webp"
        source.write_bytes(normalize_bside_asset(_image_bytes(), "background")["data"])
        background.background_uri = str(source)
        background.status = "ACTIVE"
        db.commit()

        session = BsideVisualSession(
            session_id="BSIDE_STYLE_TEST",
            source_qwen_run_id="QWEN_STYLE_TEST",
            style_seed=12345,
        )
        db.add(session)
        db.commit()
        first = get_bside_style_plan(session, db)
        db.commit()
        first_ids = (session.background_id, session.outline_style_id, session.outline_profile_id, session.style_seed)
        db.expire_all()
        refreshed = db.get(BsideVisualSession, session.session_id)
        assert refreshed is not None
        second = get_bside_style_plan(refreshed, db)
        assert first_ids == (refreshed.background_id, refreshed.outline_style_id, refreshed.outline_profile_id, refreshed.style_seed)
        assert first["background"].id == second["background"].id
        assert first["profile"].id == second["profile"].id
        assert first["fish_width_ratio"] == second["fish_width_ratio"]
    finally:
        db.close()


def test_background_upload_slots_preview_and_active_gate(tmp_path):
    db = _db(tmp_path)
    try:
        seed_bside_asset_registry(db)
        created = create_bside_background(
            BsideBackgroundCreate(name="可扩展背景 04", code="test_background_04"),
            db,
        )
        background_id = created["id"]

        def upload(name: str, data: bytes):
            return UploadFile(
                file=io.BytesIO(data),
                filename=name,
                headers=Headers({"content-type": "image/png"}),
            )

        uploaded = asyncio.run(
            upload_bside_background_asset(
                background_id,
                "background",
                upload("lake.png", _image_bytes()),
                db,
            )
        )
        assert uploaded["uploaded"]["preview_generated"] is True
        assert uploaded["assets"]["background"]["available"] is True
        assert uploaded["assets"]["preview"]["available"] is True

        asyncio.run(
            upload_bside_background_asset(
                background_id,
                "foreground",
                upload("foreground.png", _image_bytes(mode="RGBA")),
                db,
            )
        )
        asyncio.run(
            upload_bside_background_asset(
                background_id,
                "light",
                upload("light.png", _image_bytes(mode="RGBA")),
                db,
            )
        )
        activated = update_bside_background(
            background_id,
            BsideBackgroundPatch(status="ACTIVE"),
            db,
        )
        assert activated["status"] == "ACTIVE"
        assert activated["activation_ready"] is True
    finally:
        db.close()


def test_renderer_discovers_test_background_04_from_registry_without_code_change(tmp_path):
    db = _db(tmp_path)
    try:
        seed_bside_asset_registry(db)
        created = create_bside_background(
            BsideBackgroundCreate(name="测试背景 04", code="test_background_04"),
            db,
        )

        def upload(name: str, data: bytes):
            return UploadFile(
                file=io.BytesIO(data),
                filename=name,
                headers=Headers({"content-type": "image/png"}),
            )

        asyncio.run(
            upload_bside_background_asset(
                created["id"], "background", upload("background.png", _image_bytes()), db
            )
        )
        activated = update_bside_background(
            created["id"], BsideBackgroundPatch(status="ACTIVE"), db
        )
        assert activated["status"] == "ACTIVE"
        assert [row.code for row in get_active_bside_backgrounds(db)] == ["test_background_04"]

        session = BsideVisualSession(
            session_id="BSIDE_EXTENSIBILITY_TEST",
            source_qwen_run_id="QWEN_EXTENSIBILITY_TEST",
            style_seed=77,
        )
        db.add(session)
        db.commit()
        plan = get_bside_style_plan(session, db)
        assert plan["background"].code == "test_background_04"
        assert plan["profile"].id is not None
    finally:
        db.close()
