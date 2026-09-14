-- YuJian AI Platform V1 additive indexes.
-- Existing batches, datasets, reviews, training runs and models remain the
-- source of truth.  This migration adds only the three permitted tables.

CREATE TABLE IF NOT EXISTS pipeline_run (
    run_id VARCHAR(128) PRIMARY KEY,
    source_batch_id VARCHAR(128),
    source_image_id VARCHAR(256),
    pipeline_type VARCHAR(64) NOT NULL DEFAULT 'FISH_ASSET',
    status VARCHAR(32) NOT NULL DEFAULT 'QUEUED',
    current_stage VARCHAR(64),
    stage_json TEXT NOT NULL DEFAULT '{}',
    model_version VARCHAR(128),
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    duration_ms INTEGER,
    error_stage VARCHAR(64),
    error_message TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_pipeline_run_status ON pipeline_run (status);
CREATE INDEX IF NOT EXISTS ix_pipeline_run_created_at ON pipeline_run (created_at);
CREATE INDEX IF NOT EXISTS ix_pipeline_run_source_image ON pipeline_run (source_batch_id, source_image_id);

CREATE TABLE IF NOT EXISTS fish_asset (
    asset_id VARCHAR(128) PRIMARY KEY,
    pipeline_run_id VARCHAR(128),
    source_batch_id VARCHAR(128),
    source_image_id VARCHAR(256),
    species VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    original_uri TEXT,
    mask_uri TEXT,
    transparent_uri TEXT,
    sticker_uri TEXT,
    version VARCHAR(64) NOT NULL DEFAULT 'v1',
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_fish_asset_status ON fish_asset (status);
CREATE INDEX IF NOT EXISTS ix_fish_asset_species ON fish_asset (species);
CREATE INDEX IF NOT EXISTS ix_fish_asset_pipeline_run ON fish_asset (pipeline_run_id);

CREATE TABLE IF NOT EXISTS platform_operation_log (
    id INTEGER PRIMARY KEY,
    operation_type VARCHAR(64) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    resource_id VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'SUCCESS',
    message TEXT,
    detail_json TEXT,
    actor VARCHAR(256) DEFAULT 'platform',
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_platform_operation_log_type ON platform_operation_log (operation_type);
CREATE INDEX IF NOT EXISTS ix_platform_operation_log_status ON platform_operation_log (status);
CREATE INDEX IF NOT EXISTS ix_platform_operation_log_created_at ON platform_operation_log (created_at);
