-- YuJian Account & Privacy MVP v1.  This migration is additive and preserves
-- the current consumer-account and JWT authentication model.

ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_object_name TEXT;

CREATE TABLE IF NOT EXISTS user_privacy_settings (
    user_id VARCHAR(36) PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    ai_model_improvement_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ai_model_improvement_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ai_model_improvement_consent_version VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS user_privacy_audit (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ai_model_improvement_enabled BOOLEAN NOT NULL,
    consent_version VARCHAR(128) NOT NULL,
    source VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_user_privacy_audit_user_created
    ON user_privacy_audit (user_id, created_at DESC);
