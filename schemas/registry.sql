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
  updated_at TEXT NOT NULL,
  FOREIGN KEY(materialized_batch_id) REFERENCES batches(batch_id)
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
CREATE INDEX IF NOT EXISTS idx_feedback_pipeline ON feedback_events(pipeline_status);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON training_runs(dataset_version);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_error_pool_status ON error_pool(status);
