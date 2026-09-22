-- B-side visual asset registry V1.
-- The runtime also creates these additive tables through SQLAlchemy metadata.
-- This file is the Cloud SQL review/apply artifact.

CREATE TABLE IF NOT EXISTS bside_background (
  id SERIAL PRIMARY KEY,
  code VARCHAR(128) NOT NULL UNIQUE,
  name VARCHAR(256) NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  background_uri TEXT,
  foreground_uri TEXT,
  light_uri TEXT,
  preview_uri TEXT,
  fish_anchor_x DOUBLE PRECISION NOT NULL DEFAULT 0.50,
  fish_anchor_y DOUBLE PRECISION NOT NULL DEFAULT 0.50,
  fish_width_min DOUBLE PRECISION NOT NULL DEFAULT 0.68,
  fish_width_max DOUBLE PRECISION NOT NULL DEFAULT 0.74,
  status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  CONSTRAINT ck_bside_background_status CHECK (status IN ('DRAFT', 'ACTIVE'))
);

CREATE INDEX IF NOT EXISTS ix_bside_background_status ON bside_background(status);

CREATE TABLE IF NOT EXISTS bside_outline_style (
  id SERIAL PRIMARY KEY,
  code VARCHAR(128) NOT NULL UNIQUE,
  name VARCHAR(256) NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  CONSTRAINT ck_bside_outline_style_status CHECK (status IN ('DRAFT', 'ACTIVE'))
);

CREATE INDEX IF NOT EXISTS ix_bside_outline_style_status ON bside_outline_style(status);

CREATE TABLE IF NOT EXISTS bside_background_outline_profile (
  id SERIAL PRIMARY KEY,
  background_id INTEGER NOT NULL REFERENCES bside_background(id) ON DELETE CASCADE,
  outline_style_id INTEGER NOT NULL REFERENCES bside_outline_style(id) ON DELETE CASCADE,
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  weight INTEGER NOT NULL DEFAULT 0,
  render_params_json TEXT NOT NULL DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  CONSTRAINT uq_bside_background_outline_profile_pair UNIQUE (background_id, outline_style_id)
);

CREATE INDEX IF NOT EXISTS ix_bside_background_outline_profile_background
  ON bside_background_outline_profile(background_id);
CREATE INDEX IF NOT EXISTS ix_bside_background_outline_profile_outline
  ON bside_background_outline_profile(outline_style_id);

-- Additive pointers on existing B-side sessions. No old rows are rewritten.
ALTER TABLE qwen_bside_visual_session
  ADD COLUMN IF NOT EXISTS background_id INTEGER,
  ADD COLUMN IF NOT EXISTS outline_style_id INTEGER,
  ADD COLUMN IF NOT EXISTS outline_profile_id INTEGER,
  ADD COLUMN IF NOT EXISTS style_seed BIGINT;
