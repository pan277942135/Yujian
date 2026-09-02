-- BBox-only review rows derived from an immutable Dataset Freeze.
CREATE TABLE IF NOT EXISTS dataset_crop_reviews (
    id BIGSERIAL PRIMARY KEY,
    source_dataset_version VARCHAR(128) NOT NULL REFERENCES datasets(dataset_version),
    source_manifest_uri TEXT NOT NULL,
    image_id VARCHAR(256) NOT NULL,
    source_image_id VARCHAR(256) NOT NULL,
    source_image_gcs_uri TEXT NOT NULL,
    species_key VARCHAR(128) NOT NULL,
    species_name VARCHAR(128) NOT NULL,
    class_index INTEGER NOT NULL,
    split VARCHAR(16) NOT NULL,
    group_id VARCHAR(256),
    candidate_bbox_json TEXT,
    accepted_bbox_json TEXT,
    bbox_source VARCHAR(64),
    review_status VARCHAR(32) NOT NULL DEFAULT 'BBOX_REQUIRED',
    reviewer VARCHAR(256),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_dataset_crop_review_source_image UNIQUE (source_dataset_version, image_id)
);

CREATE TABLE IF NOT EXISTS dataset_crop_review_events (
    id BIGSERIAL PRIMARY KEY,
    source_dataset_version VARCHAR(128) NOT NULL,
    image_id VARCHAR(256) NOT NULL,
    action VARCHAR(64) NOT NULL,
    reviewer VARCHAR(256),
    before_json TEXT,
    after_json TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
