-- Reversible application rollback. Keep publication audit and immutable
-- content-revision history tables so rollback never discards operator history.
-- These tables intentionally remain orphan-compatible until v1.4 is restored.
DROP INDEX IF EXISTS uq_fish_knowledge_asset_active_role;
DROP INDEX IF EXISTS uq_fish_cards_asset_version_id;
DROP INDEX IF EXISTS ix_fish_cards_asset_version_id;
ALTER TABLE fish_cards DROP COLUMN IF EXISTS asset_version_id;
ALTER TABLE fish_cards DROP COLUMN IF EXISTS content_revision;
