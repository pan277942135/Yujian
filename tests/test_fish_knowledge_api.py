from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base, get_db
from app.fish_knowledge import (
    FishFishing,
    FishGalleryImage,
    FishProfile,
    FishSimilarity,
    FishSpecies,
    FishVideo,
)
from app.fish_knowledge.api import router
from app.models import SpeciesCatalog
from app.secure import install_access_guard


def _build_client(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'fish-knowledge.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    app = FastAPI()
    install_access_guard(app)
    app.include_router(router)

    def override_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setenv("CONSOLE_ACCESS_KEY", "console-secret")
    return app, session_factory


def _get(app: FastAPI, path: str) -> tuple[int, object]:
    async def request():
        sent = []
        request_delivered = False

        async def receive():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("utf-8"),
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "client": ("test", 50000),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        start = next(item for item in sent if item["type"] == "http.response.start")
        body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
        return start["status"], json.loads(body.decode("utf-8"))

    return asyncio.run(request())


def _catalog(species_id: str, name_cn: str, order: int) -> SpeciesCatalog:
    return SpeciesCatalog(
        species_key=species_id,
        catalog_order=order,
        common_name_zh=name_cn,
        status="active",
    )


def _seed_api_data(session_factory):
    with session_factory() as db:
        db.add_all(
            [
                _catalog("grass_carp", "草鱼", 0),
                _catalog("black_carp", "青鱼", 1),
                _catalog("draft_fish", "测试鱼", 2),
            ]
        )
        db.add_all(
            [
                FishSpecies(
                    id="grass_carp",
                    name_cn="草鱼",
                    alias=["鲩鱼", "草鲩"],
                    scientific_name="Ctenopharyngodon idella",
                    category="淡水鱼",
                    family="鲤科",
                    genus="草鱼属",
                    summary="体型修长，是常见的大型淡水目标鱼",
                    status="ACTIVE",
                ),
                FishSpecies(
                    id="black_carp",
                    name_cn="青鱼",
                    alias=["螺蛳青"],
                    category="淡水鱼",
                    summary="常见大型底层鱼",
                    status="ACTIVE",
                ),
                FishSpecies(
                    id="draft_fish",
                    name_cn="测试鱼",
                    alias=[],
                    category="淡水鱼",
                    summary="草稿",
                    status="DRAFT",
                ),
            ]
        )
        db.flush()
        db.add_all(
            [
                FishGalleryImage(
                    species_id="grass_carp",
                    type="standard",
                    url="https://cdn.example/grass-standard.jpg",
                    title="标准侧身",
                    sort_order=0,
                ),
                FishGalleryImage(
                    species_id="grass_carp",
                    type="top",
                    url="https://cdn.example/grass-top.jpg",
                    title="俯视",
                    sort_order=1,
                ),
                FishProfile(
                    species_id="grass_carp",
                    body_shape="体型修长侧扁",
                    features=["背部青灰", "鳞片较大", "尾鳍深叉"],
                    habitat=["江河", "湖泊", "水库"],
                    food="草食性为主",
                    season=["春", "夏", "秋"],
                ),
                FishFishing(
                    species_id="grass_carp",
                    water_layer="中下层",
                    season=["夏", "秋"],
                    bait=["玉米", "青草"],
                    method=["浮钓", "底钓"],
                    summary="水温较高时更活跃。",
                ),
                FishVideo(
                    species_id="grass_carp",
                    title="这鱼怎么钓",
                    type="HOW_TO_FISH",
                    cover_url="https://cdn.example/video-cover.jpg",
                    video_url="https://cdn.example/grass-how-to.mp4",
                    duration=90,
                    tags=["草鱼", "玉米"],
                ),
                FishSimilarity(
                    species_id="grass_carp",
                    similar_species_id="black_carp",
                    difference="草鱼体色偏黄绿、吻部较圆；青鱼体色更深、头部更尖长。",
                ),
            ]
        )
        db.commit()


def test_species_list_returns_active_species_and_gallery_cover(tmp_path, monkeypatch):
    app, session_factory = _build_client(tmp_path, monkeypatch)
    _seed_api_data(session_factory)

    status, body = _get(app, "/api/v1/fish/species")

    assert status == 200
    assert [item["id"] for item in body] == ["grass_carp", "black_carp"]
    assert body[0] == {
        "id": "grass_carp",
        "name_cn": "草鱼",
        "cover_image": "https://cdn.example/grass-standard.jpg",
        "summary": "体型修长，是常见的大型淡水目标鱼",
    }
    assert body[1]["cover_image"] is None


def test_species_detail_returns_all_knowledge_sections(tmp_path, monkeypatch):
    app, session_factory = _build_client(tmp_path, monkeypatch)
    _seed_api_data(session_factory)

    status, body = _get(app, "/api/v1/fish/species/grass_carp")

    assert status == 200
    assert set(body) == {"species", "gallery", "profile", "fishing", "videos", "similarity"}
    assert body["species"]["alias"] == ["鲩鱼", "草鲩"]
    assert [image["type"] for image in body["gallery"]["images"]] == ["standard", "top"]
    assert body["profile"]["features"] == ["背部青灰", "鳞片较大", "尾鳍深叉"]
    assert body["fishing"]["bait"] == ["玉米", "青草"]
    assert body["videos"][0]["type"] == "HOW_TO_FISH"
    assert body["similarity"][0]["similar_species_id"] == "black_carp"


def test_species_detail_has_stable_empty_states(tmp_path, monkeypatch):
    app, session_factory = _build_client(tmp_path, monkeypatch)
    _seed_api_data(session_factory)

    status, body = _get(app, "/api/v1/fish/species/black_carp")

    assert status == 200
    assert body["gallery"] == {"species_id": "black_carp", "images": []}
    assert body["profile"]["features"] == []
    assert body["profile"]["habitat"] == []
    assert body["fishing"]["bait"] == []
    assert body["fishing"]["method"] == []
    assert body["videos"] == []
    assert body["similarity"] == []


def test_unknown_or_draft_species_returns_404(tmp_path, monkeypatch):
    app, session_factory = _build_client(tmp_path, monkeypatch)
    _seed_api_data(session_factory)

    assert _get(app, "/api/v1/fish/species/not_a_species")[0] == 404
    assert _get(app, "/api/v1/fish/species/draft_fish")[0] == 404
