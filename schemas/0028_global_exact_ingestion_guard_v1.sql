-- Global Exact Duplicate Ingestion Guard V1.
-- ImageAsset remains intentionally non-unique for historical compatibility.
CREATE TABLE IF NOT EXISTS global_image_contents (
  id BIGSERIAL PRIMARY KEY,
  sha256 VARCHAR(64) NOT NULL UNIQUE,
  lifecycle_status VARCHAR(16) NOT NULL DEFAULT 'RESERVED',
  canonical_batch_id VARCHAR(128),
  canonical_image_id VARCHAR(256),
  canonical_image_asset_id BIGINT,
  canonical_object_name TEXT,
  incoming_batch_id VARCHAR(128),
  incoming_path TEXT,
  source VARCHAR(128),
  last_error TEXT,
  first_seen_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
  CONSTRAINT ck_global_image_contents_lifecycle
    CHECK (lifecycle_status IN ('RESERVED', 'ACTIVE', 'FAILED')),
  FOREIGN KEY (canonical_image_asset_id) REFERENCES image_assets(id)
);
CREATE INDEX IF NOT EXISTS idx_global_image_contents_sha256 ON global_image_contents(sha256);
CREATE INDEX IF NOT EXISTS idx_global_image_contents_status ON global_image_contents(lifecycle_status);

CREATE TABLE IF NOT EXISTS global_duplicate_audits (
  id BIGSERIAL PRIMARY KEY,
  sha256 VARCHAR(64) NOT NULL,
  incoming_batch_id VARCHAR(128) NOT NULL,
  incoming_path TEXT NOT NULL,
  source VARCHAR(128) NOT NULL,
  canonical_batch_id VARCHAR(128),
  canonical_image_id VARCHAR(256),
  canonical_image_asset_id BIGINT,
  canonical_object_name TEXT,
  reason VARCHAR(64) NOT NULL DEFAULT 'GLOBAL_EXACT_DUPLICATE',
  blocked_at TIMESTAMP WITH TIME ZONE NOT NULL,
  FOREIGN KEY (canonical_image_asset_id) REFERENCES image_assets(id)
);
CREATE INDEX IF NOT EXISTS idx_global_duplicate_audits_sha256 ON global_duplicate_audits(sha256);
CREATE INDEX IF NOT EXISTS idx_global_duplicate_audits_batch ON global_duplicate_audits(incoming_batch_id);

CREATE TABLE IF NOT EXISTS global_image_duplicate_members (
  id BIGSERIAL PRIMARY KEY,
  sha256 VARCHAR(64) NOT NULL,
  image_asset_id BIGINT NOT NULL,
  batch_id VARCHAR(128) NOT NULL,
  image_id VARCHAR(256) NOT NULL,
  object_name TEXT,
  recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
  CONSTRAINT uq_global_duplicate_member_asset UNIQUE (sha256, image_asset_id),
  FOREIGN KEY (image_asset_id) REFERENCES image_assets(id)
);
CREATE INDEX IF NOT EXISTS idx_global_duplicate_members_sha256 ON global_image_duplicate_members(sha256);
