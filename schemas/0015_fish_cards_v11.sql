-- YuJian Fish Knowledge Admin V1.1: five square detail cards per species.
-- ECO/GEAR/SKILL are the canonical product names; the three longer aliases
-- remain valid for backwards-compatible imports from the original brief.

CREATE TABLE IF NOT EXISTS fish_cards (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    card_type VARCHAR(32) NOT NULL,
    title VARCHAR(256) NOT NULL DEFAULT '',
    image_url TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    sort_order INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_cards_type CHECK (card_type IN ('HERO', 'IDENTIFICATION', 'ECO', 'GEAR', 'SKILL', 'ECOLOGY', 'FISHING', 'RECORD')),
    CONSTRAINT ck_fish_cards_status CHECK (status IN ('ACTIVE', 'DRAFT')),
    CONSTRAINT ck_fish_cards_sort_order CHECK (sort_order >= 0)
);

CREATE INDEX IF NOT EXISTS ix_fish_cards_species_id ON fish_cards(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_cards_status ON fish_cards(status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_fish_cards_active_type
    ON fish_cards(species_id, card_type)
    WHERE status = 'ACTIVE';
