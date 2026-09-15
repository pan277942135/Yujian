ALTER TABLE models ADD COLUMN is_production BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE models ADD COLUMN published_at TIMESTAMP WITH TIME ZONE;

CREATE TABLE IF NOT EXISTS model_publish_jobs (
  publish_job_id VARCHAR(128) PRIMARY KEY,
  run_id VARCHAR(128) NOT NULL REFERENCES training_runs(run_id),
  model_version VARCHAR(128) NOT NULL REFERENCES models(model_version),
  source_artifact_uri TEXT NOT NULL,
  model_prefix TEXT NOT NULL,
  target_artifact_uri TEXT,
  published_filename VARCHAR(256) NOT NULL DEFAULT 'fish_classifier_v0_2.tflite',
  status VARCHAR(32) NOT NULL DEFAULT 'NOT_PUBLISHED',
  stage VARCHAR(64),
  active_lock VARCHAR(32) UNIQUE,
  callback_token_sha256 VARCHAR(64) NOT NULL,
  workflow_run_id VARCHAR(128),
  workflow_run_url TEXT,
  github_release_url TEXT,
  sha256 VARCHAR(64),
  error_code VARCHAR(64),
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  published_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS ix_model_publish_jobs_model_version ON model_publish_jobs(model_version);
CREATE INDEX IF NOT EXISTS ix_model_publish_jobs_status ON model_publish_jobs(status);
CREATE INDEX IF NOT EXISTS ix_models_is_production ON models(is_production);
