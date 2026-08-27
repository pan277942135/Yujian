-- YuJian Dataset Freeze V0.1 lineage migration
-- Adds immutable per-image lineage for frozen Dataset versions.

CREATE TABLE IF NOT EXISTS dataset_items (
    id BIGSERIAL PRIMARY KEY,
    dataset_version VARCHAR(128) NOT NULL REFERENCES datasets(dataset_version) ON DELETE CASCADE,
    image_asset_id INTEGER NOT NULL REFERENCES image_assets(id),
    batch_id VARCHAR(128) NOT NULL,
    image_id VARCHAR(256) NOT NULL,
    gcs_uri TEXT NOT NULL,
    species_key VARCHAR(128) NOT NULL,
    species_name VARCHAR(128) NOT NULL,
    class_index INTEGER NOT NULL,
    split VARCHAR(16) NOT NULL,
    presence_status VARCHAR(32),
    duplicate_group VARCHAR(128),
    group_id VARCHAR(256),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_dataset_item_version_image UNIQUE (dataset_version, image_asset_id)
);

CREATE INDEX IF NOT EXISTS ix_dataset_items_dataset_version ON dataset_items(dataset_version);
CREATE INDEX IF NOT EXISTS ix_dataset_items_image_asset_id ON dataset_items(image_asset_id);
CREATE INDEX IF NOT EXISTS ix_dataset_items_batch_id ON dataset_items(batch_id);
CREATE INDEX IF NOT EXISTS ix_dataset_items_image_id ON dataset_items(image_id);
CREATE INDEX IF NOT EXISTS ix_dataset_items_species_key ON dataset_items(species_key);
CREATE INDEX IF NOT EXISTS ix_dataset_items_species_name ON dataset_items(species_name);
CREATE INDEX IF NOT EXISTS ix_dataset_items_split ON dataset_items(split);
