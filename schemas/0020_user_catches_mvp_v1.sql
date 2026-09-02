-- YuJian MVP User Loop v1: consumer accounts and private fish-catch archive.

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    nickname VARCHAR(64) NOT NULL,
    avatar_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fish_catches (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id),
    image_url TEXT NOT NULL,
    image_object_name TEXT NOT NULL,
    species_id VARCHAR(128) NOT NULL,
    species_name VARCHAR(128) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    detector_result_json TEXT,
    classifier_result_json TEXT,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_fish_catches_user_captured_at
    ON fish_catches (user_id, captured_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_fish_catches_user_species
    ON fish_catches (user_id, species_id);
