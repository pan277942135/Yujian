-- Fish Knowledge Asset Batch Import V1
CREATE TABLE IF NOT EXISTS fish_asset_import_batches (
    id BIGSERIAL PRIMARY KEY,
    batch_id VARCHAR(128) NOT NULL UNIQUE,
    source_gcs_uri TEXT NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'CREATED',
    created_by VARCHAR(256) NOT NULL DEFAULT 'admin',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_files INTEGER NOT NULL DEFAULT 0,
    recognized_files INTEGER NOT NULL DEFAULT 0,
    valid_files INTEGER NOT NULL DEFAULT 0,
    warning_files INTEGER NOT NULL DEFAULT 0,
    failed_files INTEGER NOT NULL DEFAULT 0,
    species_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    CONSTRAINT ck_fish_asset_import_batch_status CHECK (status IN ('CREATED','SCANNING','READY','IMPORTING','COMPLETED','FAILED','CANCELLED'))
);

CREATE TABLE IF NOT EXISTS fish_asset_import_items (
    id BIGSERIAL PRIMARY KEY,
    batch_id VARCHAR(128) NOT NULL REFERENCES fish_asset_import_batches(batch_id) ON DELETE CASCADE,
    species_id VARCHAR(128) REFERENCES fish_species(id) ON DELETE RESTRICT,
    source_object TEXT NOT NULL,
    asset_type VARCHAR(64),
    direction VARCHAR(16),
    source_filename VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128),
    width INTEGER,
    height INTEGER,
    aspect_ratio TEXT,
    file_size INTEGER,
    sha256 VARCHAR(64),
    validation_status VARCHAR(16) NOT NULL DEFAULT 'INVALID',
    validation_errors TEXT NOT NULL DEFAULT '[]',
    validation_warnings TEXT NOT NULL DEFAULT '[]',
    target_object TEXT,
    version_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fish_knowledge_asset_versions (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE CASCADE,
    asset_type VARCHAR(64) NOT NULL,
    direction VARCHAR(16),
    version INTEGER NOT NULL,
    object_name TEXT NOT NULL UNIQUE,
    image_url TEXT NOT NULL UNIQUE,
    status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
    sha256 VARCHAR(64) NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    batch_id VARCHAR(128) REFERENCES fish_asset_import_batches(batch_id) ON DELETE SET NULL,
    item_id BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fish_knowledge_asset_version_slot UNIQUE (species_id, asset_type, version),
    CONSTRAINT ck_fish_knowledge_asset_version_type CHECK (asset_type IN ('COVER','COVER_CARD','COVER_CARD_TRANSPARENT_LEFT','COVER_CARD_TRANSPARENT_RIGHT','HERO','IDENTIFICATION','ECO','GEAR','SKILL')),
    CONSTRAINT ck_fish_knowledge_asset_version_status CHECK (status IN ('DRAFT','ACTIVE','ARCHIVED'))
);
