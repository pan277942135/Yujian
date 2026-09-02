-- Additive metadata for reviewed crop-classifier dataset snapshots.
-- Existing Dataset Freeze rows remain immutable and keep their original
-- WHOLE_IMAGE_V1 semantics.
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS metadata_json TEXT;
