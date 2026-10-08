-- Fish Knowledge Asset V1.3 role-aware versioning and review/freeze records.
-- The application startup migration performs the same additive column/backfill
-- work for installations where historical JSON needs tolerant parsing.

ALTER TABLE fish_asset_import_items
    ADD COLUMN IF NOT EXISTS asset_role VARCHAR(32);

ALTER TABLE fish_asset_import_batches
    ADD COLUMN IF NOT EXISTS warnings_acknowledged BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE fish_asset_import_batches
    ADD COLUMN IF NOT EXISTS warnings_acknowledged_at TIMESTAMPTZ;
ALTER TABLE fish_asset_import_batches
    ADD COLUMN IF NOT EXISTS warnings_acknowledged_by VARCHAR(256);

ALTER TABLE fish_knowledge_asset_versions
    ADD COLUMN IF NOT EXISTS asset_role VARCHAR(32);

UPDATE fish_knowledge_asset_versions
SET asset_role = CASE
    WHEN asset_type <> 'COVER' THEN asset_type
    WHEN metadata_json LIKE '%"asset_role":"COVER_LIST"%'
      OR metadata_json LIKE '%"asset_role": "COVER_LIST"%' THEN 'COVER_LIST'
    WHEN metadata_json LIKE '%"asset_role":"COVER_HERO"%'
      OR metadata_json LIKE '%"asset_role": "COVER_HERO"%' THEN 'COVER_HERO'
    WHEN metadata_json LIKE '%"asset_role":"TRANSPARENT_MAIN"%'
      OR metadata_json LIKE '%"asset_role": "TRANSPARENT_MAIN"%' THEN 'TRANSPARENT_MAIN'
    WHEN metadata_json LIKE '%"asset_role":"TRANSPARENT_ALT"%'
      OR metadata_json LIKE '%"asset_role": "TRANSPARENT_ALT"%' THEN 'TRANSPARENT_ALT'
    WHEN metadata_json LIKE '%COVER_CARD_TRANSPARENT_LEFT%' THEN 'TRANSPARENT_MAIN'
    WHEN metadata_json LIKE '%COVER_CARD_TRANSPARENT_RIGHT%' THEN 'TRANSPARENT_ALT'
    WHEN metadata_json LIKE '%COVER_HERO%' THEN 'COVER_HERO'
    ELSE 'COVER_LIST'
END
WHERE asset_role IS NULL;

UPDATE fish_asset_import_items
SET asset_role = CASE
    WHEN lower(source_filename) ~ '(^|/)00_cover_hero(_[^.]*)?[.](png|jpg|jpeg|webp)$' THEN 'COVER_HERO'
    WHEN lower(source_filename) ~ '(^|/)01_transparent_main(_[^.]*)?[.](png|jpg|jpeg|webp)$' THEN 'TRANSPARENT_MAIN'
    WHEN lower(source_filename) ~ '(^|/)02_transparent_alt(_[^.]*)?[.](png|jpg|jpeg|webp)$' THEN 'TRANSPARENT_ALT'
    WHEN asset_type = 'COVER' THEN 'COVER_LIST'
    ELSE asset_type
END
WHERE asset_role IS NULL;

ALTER TABLE fish_knowledge_asset_versions
    DROP CONSTRAINT IF EXISTS uq_fish_knowledge_asset_version_slot;

DROP INDEX IF EXISTS uq_fish_knowledge_asset_version_slot;

CREATE UNIQUE INDEX IF NOT EXISTS uq_fish_knowledge_asset_role_version
    ON fish_knowledge_asset_versions (species_id, asset_role, version);

CREATE INDEX IF NOT EXISTS ix_fish_asset_import_items_asset_role
    ON fish_asset_import_items (asset_role);

CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_versions_asset_role
    ON fish_knowledge_asset_versions (asset_role);

CREATE TABLE IF NOT EXISTS fish_knowledge_asset_reviews (
    id BIGSERIAL PRIMARY KEY,
    batch_id VARCHAR(128) NOT NULL REFERENCES fish_asset_import_batches(batch_id) ON DELETE CASCADE,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE RESTRICT,
    asset_role VARCHAR(32) NOT NULL,
    version_id BIGINT NOT NULL REFERENCES fish_knowledge_asset_versions(id) ON DELETE RESTRICT,
    source_filename VARCHAR(512) NOT NULL,
    source_sha256 VARCHAR(64) NOT NULL,
    derived_media_sha256 VARCHAR(64) NOT NULL,
    object_name TEXT NOT NULL,
    object_generation VARCHAR(64),
    validation_result VARCHAR(16) NOT NULL DEFAULT 'PASS',
    validation_warnings_json TEXT NOT NULL DEFAULT '[]',
    visual_qa_result VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    content_qa_result VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    review_note TEXT NOT NULL DEFAULT '',
    reviewer VARCHAR(256) NOT NULL DEFAULT 'admin',
    binding_type VARCHAR(32),
    binding_id INTEGER,
    binding_status VARCHAR(16),
    binding_image_url TEXT,
    cms_content_sha256 VARCHAR(64),
    asset_status VARCHAR(16),
    warnings_acknowledged_at TIMESTAMPTZ,
    warnings_acknowledged_by VARCHAR(256),
    reviewed_at TIMESTAMPTZ,
    frozen_at TIMESTAMPTZ,
    code_head VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fish_knowledge_review_batch_slot UNIQUE (batch_id, species_id, asset_role),
    CONSTRAINT ck_fish_knowledge_review_visual CHECK (visual_qa_result IN ('PENDING','PASS','BLOCKED_VISUAL_QA')),
    CONSTRAINT ck_fish_knowledge_review_content CHECK (content_qa_result IN ('PENDING','PASS','BLOCKED_CONTENT_MISMATCH'))
);

ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS binding_type VARCHAR(32);
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS binding_id INTEGER;
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS binding_status VARCHAR(16);
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS binding_image_url TEXT;
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS cms_content_sha256 VARCHAR(64);
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS asset_status VARCHAR(16);
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS validation_warnings_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS warnings_acknowledged_at TIMESTAMPTZ;
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS warnings_acknowledged_by VARCHAR(256);

CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_reviews_batch_id
    ON fish_knowledge_asset_reviews (batch_id);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_reviews_species_id
    ON fish_knowledge_asset_reviews (species_id);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_reviews_version_id
    ON fish_knowledge_asset_reviews (version_id);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_reviews_frozen_at
    ON fish_knowledge_asset_reviews (frozen_at);
