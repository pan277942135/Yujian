-- YuJian Fish Knowledge Database V1.
-- Fish knowledge IDs reuse species_catalog.species_key so model classes,
-- reviewed labels, and App knowledge pages cannot silently diverge.

CREATE TABLE IF NOT EXISTS fish_species (
    id VARCHAR(128) PRIMARY KEY REFERENCES species_catalog(species_key) ON DELETE RESTRICT,
    name_cn VARCHAR(128) NOT NULL UNIQUE,
    alias JSON NOT NULL DEFAULT '[]',
    scientific_name VARCHAR(256),
    category VARCHAR(64) NOT NULL,
    family VARCHAR(128),
    genus VARCHAR(128),
    summary TEXT NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_species_status CHECK (status IN ('ACTIVE', 'DRAFT'))
);

CREATE TABLE IF NOT EXISTS fish_gallery (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    type VARCHAR(32) NOT NULL,
    url TEXT NOT NULL,
    title VARCHAR(256),
    sort_order INTEGER NOT NULL DEFAULT 0,
    object_name TEXT,
    content_type VARCHAR(128),
    size_bytes INTEGER,
    sha256 VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_gallery_type CHECK (type IN ('standard', 'side', 'top', 'catch', 'environment', 'action')),
    CONSTRAINT uq_fish_gallery_species_order UNIQUE (species_id, sort_order),
    CONSTRAINT uq_fish_gallery_species_url UNIQUE (species_id, url)
);

CREATE TABLE IF NOT EXISTS fish_profile (
    species_id VARCHAR(128) PRIMARY KEY REFERENCES fish_species(id) ON DELETE CASCADE,
    body_shape TEXT,
    features JSON NOT NULL DEFAULT '[]',
    habitat JSON NOT NULL DEFAULT '[]',
    food TEXT,
    season JSON NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fish_fishing (
    species_id VARCHAR(128) PRIMARY KEY REFERENCES fish_species(id) ON DELETE CASCADE,
    water_layer VARCHAR(128),
    season JSON NOT NULL DEFAULT '[]',
    bait JSON NOT NULL DEFAULT '[]',
    method JSON NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fish_video (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    title VARCHAR(256) NOT NULL,
    type VARCHAR(32) NOT NULL,
    cover_url TEXT,
    video_url TEXT NOT NULL,
    duration INTEGER NOT NULL,
    tags JSON NOT NULL DEFAULT '[]',
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_video_type CHECK (type IN ('INTRO', 'HOW_TO_FISH', 'REAL_CATCH', 'EQUIPMENT')),
    CONSTRAINT uq_fish_video_species_url UNIQUE (species_id, video_url)
);

CREATE TABLE IF NOT EXISTS fish_similarity (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    similar_species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    difference TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_similarity_not_self CHECK (species_id <> similar_species_id),
    CONSTRAINT uq_fish_similarity_pair UNIQUE (species_id, similar_species_id)
);

CREATE TABLE IF NOT EXISTS fish_ranking (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    type VARCHAR(32) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    location VARCHAR(256),
    user_id VARCHAR(256) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_fish_ranking_type CHECK (type IN ('MAX_WEIGHT', 'MAX_LENGTH', 'MOST_CATCHES'))
);

CREATE INDEX IF NOT EXISTS ix_fish_species_name_cn ON fish_species(name_cn);
CREATE INDEX IF NOT EXISTS ix_fish_species_status ON fish_species(status);
CREATE INDEX IF NOT EXISTS ix_fish_gallery_species_id ON fish_gallery(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_video_species_id ON fish_video(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_similarity_species_id ON fish_similarity(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_similarity_similar_species_id ON fish_similarity(similar_species_id);
CREATE INDEX IF NOT EXISTS ix_fish_ranking_species_id ON fish_ranking(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_ranking_type ON fish_ranking(type);
CREATE INDEX IF NOT EXISTS ix_fish_ranking_user_id ON fish_ranking(user_id);
