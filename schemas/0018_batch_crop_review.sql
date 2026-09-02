-- Additive human gate for normal Batch images entering the crop pipeline.
-- candidate_bbox_json is diagnostic only; accepted_bbox_json is written by
-- an explicit reviewer action and is the only box accepted by the builder.
CREATE TABLE IF NOT EXISTS batch_crop_reviews (
  id BIGSERIAL PRIMARY KEY,
  batch_id VARCHAR(128) NOT NULL,
  image_asset_id INTEGER NOT NULL UNIQUE,
  image_id VARCHAR(256) NOT NULL,
  candidate_bbox_json TEXT,
  accepted_bbox_json TEXT,
  detector_version VARCHAR(128),
  species_key VARCHAR(128),
  species_name VARCHAR(128),
  status VARCHAR(32) NOT NULL DEFAULT 'REVIEW_REQUIRED',
  reviewer VARCHAR(256),
  reviewed_at TIMESTAMPTZ,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  FOREIGN KEY(batch_id) REFERENCES batches(batch_id),
  FOREIGN KEY(image_asset_id) REFERENCES image_assets(id)
);

CREATE INDEX IF NOT EXISTS idx_batch_crop_review_batch ON batch_crop_reviews(batch_id);
CREATE INDEX IF NOT EXISTS idx_batch_crop_review_status ON batch_crop_reviews(status);
