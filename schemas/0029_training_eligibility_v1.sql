-- Historical Global Exact Duplicate Cleanup V1: independent training eligibility.
-- Additive and idempotent; existing ImageAsset rows remain eligible by default.
ALTER TABLE image_assets
  ADD COLUMN IF NOT EXISTS training_eligible BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE image_assets
  ADD COLUMN IF NOT EXISTS training_exclusion_reason VARCHAR(64);
ALTER TABLE image_assets
  ADD COLUMN IF NOT EXISTS duplicate_of_image_asset_id INTEGER;
ALTER TABLE image_assets
  ADD COLUMN IF NOT EXISTS training_eligibility_source TEXT;
ALTER TABLE image_assets
  ADD COLUMN IF NOT EXISTS training_eligibility_updated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_image_assets_training_eligible
  ON image_assets (training_eligible);
CREATE INDEX IF NOT EXISTS ix_image_assets_training_exclusion_reason
  ON image_assets (training_exclusion_reason);
CREATE INDEX IF NOT EXISTS ix_image_assets_duplicate_of_image_asset_id
  ON image_assets (duplicate_of_image_asset_id);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'fk_image_assets_duplicate_of_image_asset_id'
  ) THEN
    ALTER TABLE image_assets
      ADD CONSTRAINT fk_image_assets_duplicate_of_image_asset_id
      FOREIGN KEY (duplicate_of_image_asset_id) REFERENCES image_assets(id);
  END IF;
END $$;
