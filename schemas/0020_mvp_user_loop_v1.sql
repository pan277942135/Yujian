-- Yujian MVP User Loop V1: manual accounts and owner-scoped fish catches.
-- No existing detector, classifier, dataset, or Fish Knowledge tables are changed.

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) PRIMARY KEY,
    username VARCHAR(128) NOT NULL UNIQUE,
    password_hash VARCHAR(256) NOT NULL,
    nickname VARCHAR(128) NOT NULL,
    avatar_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fish_catches (
    id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL DEFAULT '',
    species_id VARCHAR(128) NOT NULL,
    species_name VARCHAR(128) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    model_version VARCHAR(128) NOT NULL DEFAULT '',
    detector_result_json TEXT,
    classifier_result_json TEXT,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_users_username ON users(username);
CREATE INDEX IF NOT EXISTS ix_fish_catches_user_id ON fish_catches(user_id);
CREATE INDEX IF NOT EXISTS ix_fish_catches_species_id ON fish_catches(species_id);
CREATE INDEX IF NOT EXISTS ix_fish_catches_created_at ON fish_catches(created_at);
