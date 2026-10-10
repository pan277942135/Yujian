-- Fish Knowledge v1.4: retain independent visual/content QA evidence.
-- Additive and idempotent; existing combined review fields are preserved.

ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS visual_qa_note TEXT NOT NULL DEFAULT '';
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS visual_qa_reviewer VARCHAR(256) NOT NULL DEFAULT 'admin';
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS visual_qa_reviewed_at TIMESTAMPTZ;
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS content_qa_note TEXT NOT NULL DEFAULT '';
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS content_qa_reviewer VARCHAR(256) NOT NULL DEFAULT 'admin';
ALTER TABLE fish_knowledge_asset_reviews
    ADD COLUMN IF NOT EXISTS content_qa_reviewed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS fish_knowledge_asset_qa_audits (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES fish_knowledge_asset_versions(id) ON DELETE RESTRICT,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE RESTRICT,
    asset_role VARCHAR(32) NOT NULL,
    qa_stage VARCHAR(16) NOT NULL CHECK (qa_stage IN ('VISUAL','CONTENT')),
    result VARCHAR(32) NOT NULL,
    reviewer VARCHAR(256) NOT NULL,
    evidence_note TEXT NOT NULL DEFAULT '',
    content_revision INTEGER,
    card_id BIGINT REFERENCES fish_cards(id) ON DELETE RESTRICT,
    source_sha256 VARCHAR(64) NOT NULL,
    derived_media_sha256 VARCHAR(64) NOT NULL,
    object_name TEXT NOT NULL,
    object_generation VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_qa_version_id
    ON fish_knowledge_asset_qa_audits (version_id);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_qa_species_id
    ON fish_knowledge_asset_qa_audits (species_id);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_asset_qa_version_stage
    ON fish_knowledge_asset_qa_audits (version_id, qa_stage);
