from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


# The product asset pack uses one list-page cover plus five square detail cards:
# Hero, identification, ecology, gear, and fishing skill.  The longer names
# from the original v1.1 brief remain accepted as input aliases so existing
# operators do not have to migrate their payloads.
CARD_TYPE_ORDER = ("HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL")
CARD_TYPE_ALIASES = {
    "ECOLOGY": "ECO",
    "FISHING": "SKILL",
    "RECORD": "GEAR",
}
CARD_TYPES = frozenset((*CARD_TYPE_ORDER, *CARD_TYPE_ALIASES))
CARD_TYPE_SORT_ORDER = {card_type: index for index, card_type in enumerate(CARD_TYPE_ORDER)}


def normalize_card_type(value: str) -> str:
    """Return the canonical card type while accepting the v1.1 aliases."""

    normalized = str(value or "").strip().upper()
    return CARD_TYPE_ALIASES.get(normalized, normalized)


def card_type_sort_order(value: str) -> int:
    return CARD_TYPE_SORT_ORDER.get(normalize_card_type(value), len(CARD_TYPE_ORDER))


class FishCard(Base):
    """One square knowledge card belonging to a fish species."""

    __tablename__ = "fish_cards"
    __table_args__ = (
        CheckConstraint(
            "card_type IN ('HERO', 'IDENTIFICATION', 'ECO', 'GEAR', 'SKILL', 'ECOLOGY', 'FISHING', 'RECORD')",
            name="ck_fish_cards_type",
        ),
        CheckConstraint("status IN ('ACTIVE', 'DRAFT')", name="ck_fish_cards_status"),
        CheckConstraint("sort_order >= 0", name="ck_fish_cards_sort_order"),
        # DRAFT rows are allowed to coexist while an editor prepares a card;
        # only one published card of a type may exist for a species.
        Index(
            "uq_fish_cards_active_type",
            "species_id",
            "card_type",
            unique=True,
            sqlite_where=text("status = 'ACTIVE'"),
            postgresql_where=text("status = 'ACTIVE'"),
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    card_type = Column(String(32), nullable=False)
    title = Column(String(256), nullable=False, default="")
    image_url = Column(Text, nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="DRAFT", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", back_populates="cards")
