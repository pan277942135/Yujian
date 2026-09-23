-- User fish-catch B-side generation flow V1.  The original catch image and
-- every generated asset remain in GCS; this table stores durable job pointers.

ALTER TABLE fish_catches ADD COLUMN IF NOT EXISTS bside_status VARCHAR(16) NOT NULL DEFAULT 'NONE';
ALTER TABLE fish_catches ADD COLUMN IF NOT EXISTS bside_result_uri TEXT;
ALTER TABLE fish_catches ADD COLUMN IF NOT EXISTS bside_result_object_name TEXT;
ALTER TABLE fish_catches ADD COLUMN IF NOT EXISTS bside_generated_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE fish_catches ADD COLUMN IF NOT EXISTS bside_job_id VARCHAR(36);

CREATE TABLE IF NOT EXISTS fish_bside_job (
  id VARCHAR(36) PRIMARY KEY,
  fish_record_id VARCHAR(36) NOT NULL REFERENCES fish_catches(id) ON DELETE CASCADE,
  user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  status VARCHAR(16) NOT NULL DEFAULT 'PENDING',
  input_image_uri TEXT,
  transparent_fish_uri TEXT,
  standardized_fish_uri TEXT,
  outlined_fish_uri TEXT,
  background_id INTEGER REFERENCES bside_background(id) ON DELETE SET NULL,
  outline_style_id INTEGER REFERENCES bside_outline_style(id) ON DELETE SET NULL,
  outline_profile_id INTEGER REFERENCES bside_background_outline_profile(id) ON DELETE SET NULL,
  style_seed INTEGER,
  result_uri TEXT,
  result_object_name TEXT,
  error_code VARCHAR(128),
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS ix_fish_bside_job_fish_record_id ON fish_bside_job(fish_record_id);
CREATE INDEX IF NOT EXISTS ix_fish_bside_job_user_id ON fish_bside_job(user_id);
CREATE INDEX IF NOT EXISTS ix_fish_bside_job_status ON fish_bside_job(status);
CREATE INDEX IF NOT EXISTS ix_fish_catches_bside_status ON fish_catches(bside_status);
