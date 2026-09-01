-- YuJian Fish Knowledge Admin V1.1: one illustrated list-page cover per species.

CREATE TABLE IF NOT EXISTS fish_species_cover (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL UNIQUE REFERENCES fish_species(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL DEFAULT '',
    style VARCHAR(64) NOT NULL DEFAULT 'ANIME_CARD',
    title VARCHAR(256) NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_species_cover_status CHECK (status IN ('ACTIVE', 'DRAFT'))
);

CREATE INDEX IF NOT EXISTS ix_fish_species_cover_species_id ON fish_species_cover(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_species_cover_status ON fish_species_cover(status);
