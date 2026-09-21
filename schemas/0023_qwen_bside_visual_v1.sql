-- Qwen B-side visual demo V1.
-- CPU-only, additive tables. One Qwen Image Edit Run can own at most one
-- session; each step points at its latest versioned output while old files
-- remain addressable in storage for debugging.

CREATE TABLE IF NOT EXISTS qwen_bside_visual_session (
  session_id VARCHAR(128) PRIMARY KEY,
  source_qwen_run_id VARCHAR(128) NOT NULL UNIQUE REFERENCES pipeline_run(run_id) ON DELETE CASCADE,
  source_transparent_fish_uri TEXT NOT NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_qwen_bside_visual_session_status
  ON qwen_bside_visual_session(status);

CREATE TABLE IF NOT EXISTS qwen_bside_visual_step (
  step_id SERIAL PRIMARY KEY,
  session_id VARCHAR(128) NOT NULL REFERENCES qwen_bside_visual_session(session_id) ON DELETE CASCADE,
  step_key VARCHAR(32) NOT NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'NOT_STARTED',
  output_uri TEXT,
  preview_uri TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  style_id VARCHAR(64),
  template_id VARCHAR(64),
  version INTEGER NOT NULL DEFAULT 0,
  error_code VARCHAR(64),
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  CONSTRAINT uq_qwen_bside_visual_step UNIQUE(session_id, step_key)
);

CREATE INDEX IF NOT EXISTS ix_qwen_bside_visual_step_session
  ON qwen_bside_visual_step(session_id);
CREATE INDEX IF NOT EXISTS ix_qwen_bside_visual_step_status
  ON qwen_bside_visual_step(status);
