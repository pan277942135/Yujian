-- Independent Image Studio V1 storage.
-- Deliberately separate from pipeline_run and fish/B-side production records.

CREATE TABLE IF NOT EXISTS image_studio_run (
  run_id VARCHAR(128) PRIMARY KEY,
  status VARCHAR(32) NOT NULL DEFAULT 'QUEUED',
  mode VARCHAR(32) NOT NULL DEFAULT 'BASE_EDIT',
  preservation VARCHAR(16) NOT NULL DEFAULT 'STRONG',
  model_version VARCHAR(128) NOT NULL DEFAULT 'qwen-image-edit-2511',
  request_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  base_image_uri TEXT NOT NULL,
  reference_uris_json TEXT NOT NULL DEFAULT '[]',
  mask_uri TEXT,
  output_image_uri TEXT,
  seed BIGINT,
  steps INTEGER NOT NULL DEFAULT 25,
  elapsed_ms INTEGER,
  error_code VARCHAR(64),
  error_message TEXT,
  started_at TIMESTAMP WITH TIME ZONE,
  finished_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_image_studio_run_status
  ON image_studio_run(status);
CREATE INDEX IF NOT EXISTS ix_image_studio_run_mode
  ON image_studio_run(mode);
CREATE INDEX IF NOT EXISTS ix_image_studio_run_created_at
  ON image_studio_run(created_at);
