PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  image_count INTEGER NOT NULL DEFAULT 0,
  manifest_uri TEXT NOT NULL,
  raw_uri TEXT NOT NULL,
  status TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS species_catalog (
  species_key TEXT PRIMARY KEY,
  catalog_order INTEGER NOT NULL UNIQUE,
  common_name_zh TEXT NOT NULL UNIQUE,
  common_name_en TEXT,
  scientific_name TEXT,
  status TEXT NOT NULL DEFAULT 'candidate',
  is_other INTEGER NOT NULL DEFAULT 0,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_assets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id TEXT NOT NULL,
  image_id TEXT NOT NULL,
  file_name TEXT NOT NULL,
  object_name TEXT NOT NULL,
  gcs_uri TEXT NOT NULL,
  source_url TEXT,
  source_platform TEXT,
  claimed_species TEXT,
  truth_species TEXT,
  truth_status TEXT NOT NULL DEFAULT 'UNCERTAIN',
  review_status TEXT NOT NULL DEFAULT 'pending',
  scene TEXT,
  lighting TEXT,
  quality TEXT,
  group_id TEXT,
  notes TEXT,
  reviewed_by TEXT,
  reviewed_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(batch_id, image_id),
  FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
);

CREATE TABLE IF NOT EXISTS review_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_asset_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  reviewer TEXT,
  before_json TEXT,
  after_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(image_asset_id) REFERENCES image_assets(id)
);

CREATE TABLE IF NOT EXISTS fish_presence_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_asset_id INTEGER NOT NULL UNIQUE,
  batch_id TEXT NOT NULL,
  status TEXT NOT NULL,
  fish_score REAL NOT NULL DEFAULT 0,
  fish_count INTEGER NOT NULL DEFAULT 0,
  max_box_area_ratio REAL NOT NULL DEFAULT 0,
  provider TEXT NOT NULL DEFAULT 'google_vision',
  model_version TEXT NOT NULL,
  evidence_json TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(image_asset_id) REFERENCES image_assets(id)
);

CREATE TABLE IF NOT EXISTS image_fingerprints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image_asset_id INTEGER NOT NULL UNIQUE,
  batch_id TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  phash_json TEXT NOT NULL,
  dhash TEXT NOT NULL,
  crop_hash TEXT NOT NULL,
  histogram_json TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  fingerprint_version TEXT NOT NULL,
  duplicate_group TEXT,
  is_representative INTEGER NOT NULL DEFAULT 1,
  duplicate_kind TEXT,
  distance_to_representative REAL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(image_asset_id) REFERENCES image_assets(id)
);

CREATE TABLE IF NOT EXISTS datasets (
  dataset_version TEXT PRIMARY KEY,
  parent_version TEXT,
  created_at TEXT NOT NULL,
  manifest_uri TEXT NOT NULL,
  class_map_uri TEXT,
  train_count INTEGER NOT NULL DEFAULT 0,
  val_count INTEGER NOT NULL DEFAULT 0,
  test_count INTEGER NOT NULL DEFAULT 0,
  species_count INTEGER NOT NULL DEFAULT 0,
  gold_version TEXT,
  git_commit TEXT NOT NULL,
  selection_mode TEXT NOT NULL DEFAULT 'ALL_APPROVED',
  source_cutoff_at TEXT,
  status TEXT NOT NULL,
  FOREIGN KEY(parent_version) REFERENCES datasets(dataset_version)
);

CREATE TABLE IF NOT EXISTS feedback_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_event_id TEXT NOT NULL UNIQUE,
  source TEXT NOT NULL DEFAULT 'app',
  image_gcs_uri TEXT,
  model_version TEXT,
  predicted_species TEXT,
  confidence REAL,
  feedback_type TEXT NOT NULL,
  corrected_species TEXT,
  user_note TEXT,
  pipeline_status TEXT NOT NULL DEFAULT 'NEW',
  materialized_batch_id TEXT,
  materialized_image_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_assets (
  image_id TEXT PRIMARY KEY,
  source TEXT NOT NULL DEFAULT 'android_detector',
  status TEXT NOT NULL DEFAULT 'RECEIVED',
  record_gcs_uri TEXT NOT NULL,
  image_gcs_uri TEXT NOT NULL,
  crop_gcs_uri TEXT,
  record_sha256 TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  crop_sha256 TEXT,
  detector_version TEXT,
  classifier_version TEXT,
  accepted_bbox_json TEXT,
  accepted_species TEXT,
  reviewed_by TEXT,
  reviewed_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fish_species (
  id TEXT PRIMARY KEY,
  name_cn TEXT NOT NULL UNIQUE,
  alias TEXT NOT NULL DEFAULT '[]',
  scientific_name TEXT,
  category TEXT NOT NULL,
  family TEXT,
  genus TEXT,
  summary TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'DRAFT' CHECK(status IN ('ACTIVE', 'DRAFT')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(id) REFERENCES species_catalog(species_key) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS fish_cards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  species_id TEXT NOT NULL,
  card_type TEXT NOT NULL CHECK(card_type IN ('HERO', 'IDENTIFICATION', 'ECO', 'GEAR', 'SKILL', 'ECOLOGY', 'FISHING', 'RECORD')),
  title TEXT NOT NULL DEFAULT '',
  image_url TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  sort_order INTEGER NOT NULL DEFAULT 0 CHECK(sort_order >= 0),
  status TEXT NOT NULL DEFAULT 'DRAFT' CHECK(status IN ('ACTIVE', 'DRAFT')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fish_cards_species ON fish_cards(species_id);
CREATE INDEX IF NOT EXISTS idx_fish_cards_status ON fish_cards(status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_fish_cards_active_type
  ON fish_cards(species_id, card_type) WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS fish_gallery (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  species_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK(type IN ('standard', 'side', 'top', 'catch', 'environment', 'action')),
  url TEXT NOT NULL,
  title TEXT,
  sort_order INTEGER NOT NULL DEFAULT 0,
  object_name TEXT,
  content_type TEXT,
  size_bytes INTEGER,
  sha256 TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(species_id, sort_order),
  UNIQUE(species_id, url),
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fish_profile (
  species_id TEXT PRIMARY KEY,
  body_shape TEXT,
  features TEXT NOT NULL DEFAULT '[]',
  habitat TEXT NOT NULL DEFAULT '[]',
  food TEXT,
  season TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fish_fishing (
  species_id TEXT PRIMARY KEY,
  water_layer TEXT,
  season TEXT NOT NULL DEFAULT '[]',
  bait TEXT NOT NULL DEFAULT '[]',
  method TEXT NOT NULL DEFAULT '[]',
  summary TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fish_video (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  species_id TEXT NOT NULL,
  title TEXT NOT NULL,
  type TEXT NOT NULL CHECK(type IN ('INTRO', 'HOW_TO_FISH', 'REAL_CATCH', 'EQUIPMENT')),
  cover_url TEXT,
  video_url TEXT NOT NULL,
  duration INTEGER NOT NULL,
  tags TEXT NOT NULL DEFAULT '[]',
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(species_id, video_url),
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fish_similarity (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  species_id TEXT NOT NULL,
  similar_species_id TEXT NOT NULL,
  difference TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK(species_id <> similar_species_id),
  UNIQUE(species_id, similar_species_id),
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE,
  FOREIGN KEY(similar_species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fish_ranking (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  species_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK(type IN ('MAX_WEIGHT', 'MAX_LENGTH', 'MOST_CATCHES')),
  value REAL NOT NULL,
  location TEXT,
  user_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS training_runs (
  run_id TEXT PRIMARY KEY,
  dataset_version TEXT NOT NULL,
  git_commit TEXT NOT NULL,
  model_family TEXT NOT NULL,
  params_json TEXT NOT NULL,
  seed INTEGER,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  artifact_uri TEXT,
  metrics_uri TEXT,
  pipeline_type TEXT NOT NULL DEFAULT 'WHOLE_IMAGE_V1',
  detector_version TEXT,
  crop_version TEXT,
  classifier_version TEXT,
  FOREIGN KEY(dataset_version) REFERENCES datasets(dataset_version)
);

CREATE TABLE IF NOT EXISTS models (
  model_version TEXT PRIMARY KEY,
  run_id TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  artifact_uri TEXT NOT NULL,
  metrics_uri TEXT,
  status TEXT NOT NULL,
  notes TEXT,
  pipeline_type TEXT NOT NULL DEFAULT 'WHOLE_IMAGE_V1',
  detector_version TEXT,
  crop_version TEXT,
  classifier_version TEXT,
  dataset_version TEXT,
  FOREIGN KEY(run_id) REFERENCES training_runs(run_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
  evaluation_id TEXT PRIMARY KEY,
  model_version TEXT NOT NULL,
  gold_version TEXT,
  created_at TEXT NOT NULL,
  metrics_uri TEXT NOT NULL,
  confusion_matrix_uri TEXT,
  errors_uri TEXT,
  FOREIGN KEY(model_version) REFERENCES models(model_version)
);

CREATE TABLE IF NOT EXISTS error_pool (
  error_id TEXT PRIMARY KEY,
  evaluation_id TEXT NOT NULL,
  image_id TEXT NOT NULL,
  truth_species TEXT,
  predicted_species TEXT,
  confidence REAL,
  hard_pair_type TEXT,
  scene TEXT,
  status TEXT NOT NULL DEFAULT 'OPEN',
  source_uri TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(evaluation_id) REFERENCES evaluations(evaluation_id)
);

CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
CREATE INDEX IF NOT EXISTS idx_species_status ON species_catalog(status);
CREATE INDEX IF NOT EXISTS idx_images_review ON image_assets(review_status);
CREATE INDEX IF NOT EXISTS idx_images_truth ON image_assets(truth_species);
CREATE INDEX IF NOT EXISTS idx_presence_batch ON fish_presence_results(batch_id);
CREATE INDEX IF NOT EXISTS idx_presence_status ON fish_presence_results(status);
CREATE INDEX IF NOT EXISTS idx_fingerprint_batch ON image_fingerprints(batch_id);
CREATE INDEX IF NOT EXISTS idx_fingerprint_sha ON image_fingerprints(sha256);
CREATE INDEX IF NOT EXISTS idx_fingerprint_group ON image_fingerprints(duplicate_group);
CREATE INDEX IF NOT EXISTS idx_feedback_pipeline ON feedback_events(pipeline_status);
CREATE INDEX IF NOT EXISTS idx_feedback_batch ON feedback_events(materialized_batch_id);
CREATE INDEX IF NOT EXISTS idx_inference_assets_status ON inference_assets(status);
CREATE INDEX IF NOT EXISTS idx_fish_species_name ON fish_species(name_cn);
CREATE INDEX IF NOT EXISTS idx_fish_species_status ON fish_species(status);
CREATE INDEX IF NOT EXISTS idx_fish_gallery_species ON fish_gallery(species_id);
CREATE INDEX IF NOT EXISTS idx_fish_video_species ON fish_video(species_id);
CREATE INDEX IF NOT EXISTS idx_fish_similarity_species ON fish_similarity(species_id);
CREATE INDEX IF NOT EXISTS idx_fish_similarity_target ON fish_similarity(similar_species_id);
CREATE INDEX IF NOT EXISTS idx_fish_ranking_species ON fish_ranking(species_id);
CREATE INDEX IF NOT EXISTS idx_fish_ranking_type ON fish_ranking(type);
CREATE INDEX IF NOT EXISTS idx_fish_ranking_user ON fish_ranking(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON training_runs(dataset_version);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_error_pool_status ON error_pool(status);
