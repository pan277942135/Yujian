-- Fish Knowledge 3+5 additive migration.
-- No table is added and no legacy URI/asset_type value is removed.
ALTER TABLE fish_asset ADD COLUMN IF NOT EXISTS asset_type VARCHAR(64);
ALTER TABLE fish_asset ADD COLUMN IF NOT EXISTS direction VARCHAR(16);
ALTER TABLE fish_asset ADD COLUMN IF NOT EXISTS asset_uri TEXT;
ALTER TABLE fish_asset ADD COLUMN IF NOT EXISTS asset_object_name TEXT;
ALTER TABLE fish_asset_import_items ADD COLUMN IF NOT EXISTS direction VARCHAR(16);
ALTER TABLE fish_knowledge_asset_versions ADD COLUMN IF NOT EXISTS direction VARCHAR(16);
CREATE INDEX IF NOT EXISTS ix_fish_asset_asset_type ON fish_asset(asset_type);
