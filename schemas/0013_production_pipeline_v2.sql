-- YuJian Production Detector -> Crop -> Classifier pipeline metadata.
-- Candidate detector boxes remain outside the training set until a review
-- endpoint writes an accepted_bbox_json value and status ACCEPTED.

CREATE TABLE IF NOT EXISTS inference_assets (
    image_id VARCHAR(256) PRIMARY KEY,
    source VARCHAR(64) NOT NULL DEFAULT 'android_detector',
    source_batch VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'RECEIVED',
    record_gcs_uri TEXT NOT NULL,
    image_gcs_uri TEXT NOT NULL,
    crop_gcs_uri TEXT,
    record_sha256 VARCHAR(64) NOT NULL,
    image_sha256 VARCHAR(64) NOT NULL,
    crop_sha256 VARCHAR(64),
    detector_version VARCHAR(128),
    classifier_version VARCHAR(128),
    accepted_bbox_json TEXT,
    accepted_species VARCHAR(128),
    reviewed_by VARCHAR(256),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE datasets ADD COLUMN IF NOT EXISTS pipeline_type VARCHAR(64) NOT NULL DEFAULT 'WHOLE_IMAGE_V1';
ALTER TABLE inference_assets ADD COLUMN IF NOT EXISTS source_batch VARCHAR(128);
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS pipeline_type VARCHAR(64) NOT NULL DEFAULT 'WHOLE_IMAGE_V1';
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS detector_version VARCHAR(128);
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS crop_version VARCHAR(128);
ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS classifier_version VARCHAR(128);
ALTER TABLE models ADD COLUMN IF NOT EXISTS pipeline_type VARCHAR(64) NOT NULL DEFAULT 'WHOLE_IMAGE_V1';
ALTER TABLE models ADD COLUMN IF NOT EXISTS detector_version VARCHAR(128);
ALTER TABLE models ADD COLUMN IF NOT EXISTS crop_version VARCHAR(128);
ALTER TABLE models ADD COLUMN IF NOT EXISTS classifier_version VARCHAR(128);
ALTER TABLE models ADD COLUMN IF NOT EXISTS dataset_version VARCHAR(128);

CREATE INDEX IF NOT EXISTS idx_inference_assets_status ON inference_assets(status);
CREATE INDEX IF NOT EXISTS idx_inference_assets_detector ON inference_assets(detector_version);
CREATE INDEX IF NOT EXISTS idx_inference_assets_species ON inference_assets(accepted_species);
