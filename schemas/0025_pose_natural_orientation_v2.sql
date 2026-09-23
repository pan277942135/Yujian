-- Persist the formal pose metadata emitted next to standardized_fish_rgba.png.
ALTER TABLE fish_bside_job
  ADD COLUMN IF NOT EXISTS pose_metadata_json TEXT;
