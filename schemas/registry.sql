PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  image_count INTEGER NOT NULL DEFAULT 0,
  manifest_uri TEXT NOT NULL,
  raw_uri TEXT NOT NULL,
  status TEXT NOT NULL,
  approved_count INTEGER NOT NULL DEFAULT 0,
  needs_review_count INTEGER NOT NULL DEFAULT 0,
  rejected_count INTEGER NOT NULL DEFAULT 0,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
  dataset_version TEXT PRIMARY KEY,
  parent_version TEXT,
  created_at TEXT NOT NULL,
  manifest_uri TEXT NOT NULL,
  train_count INTEGER NOT NULL DEFAULT 0,
  val_count INTEGER NOT NULL DEFAULT 0,
  test_count INTEGER NOT NULL DEFAULT 0,
  gold_version TEXT,
  git_commit TEXT NOT NULL,
  status TEXT NOT NULL,
  FOREIGN KEY(parent_version) REFERENCES datasets(dataset_version)
);

CREATE TABLE IF NOT EXISTS dataset_batches (
  dataset_version TEXT NOT NULL,
  batch_id TEXT NOT NULL,
  PRIMARY KEY(dataset_version, batch_id),
  FOREIGN KEY(dataset_version) REFERENCES datasets(dataset_version),
  FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
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
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON training_runs(dataset_version);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_error_pool_status ON error_pool(status);
