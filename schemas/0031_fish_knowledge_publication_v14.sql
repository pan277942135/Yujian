-- Fish Knowledge v1.4: bind structured cards to exact immutable image versions.
-- Additive and reversible. No FishCard, asset, or historical revision is deleted.

ALTER TABLE fish_cards
    ADD COLUMN IF NOT EXISTS asset_version_id BIGINT
        REFERENCES fish_knowledge_asset_versions(id) ON DELETE RESTRICT;
ALTER TABLE fish_cards
    ADD COLUMN IF NOT EXISTS content_revision INTEGER NOT NULL DEFAULT 1;

-- Only one-to-one exact species + canonical role + URL matches are migrated.
-- Ambiguous historical rows stay unbound for explicit repair.
WITH version_keys AS (
    SELECT id, species_id,
           CASE WHEN asset_role = 'ECOLOGY' THEN 'ECO' ELSE asset_role END AS role,
           image_url,
           COUNT(*) OVER (PARTITION BY species_id,
               CASE WHEN asset_role = 'ECOLOGY' THEN 'ECO' ELSE asset_role END,
               image_url) AS version_count
    FROM fish_knowledge_asset_versions
), card_keys AS (
    SELECT id, species_id,
           CASE
             WHEN card_type = 'ECOLOGY' THEN 'ECO'
             WHEN card_type = 'FISHING' THEN 'SKILL'
             WHEN card_type = 'RECORD' THEN 'GEAR'
             ELSE card_type
           END AS role,
           image_url,
           COUNT(*) OVER (PARTITION BY species_id,
               CASE
                 WHEN card_type = 'ECOLOGY' THEN 'ECO'
                 WHEN card_type = 'FISHING' THEN 'SKILL'
                 WHEN card_type = 'RECORD' THEN 'GEAR'
                 ELSE card_type
               END,
               image_url) AS card_count
    FROM fish_cards
    WHERE asset_version_id IS NULL
), exact_pairs AS (
    SELECT c.id AS card_id, v.id AS version_id
    FROM card_keys c
    JOIN version_keys v
      ON v.species_id = c.species_id AND v.role = c.role AND v.image_url = c.image_url
    WHERE c.card_count = 1 AND v.version_count = 1
)
UPDATE fish_cards c
SET asset_version_id = p.version_id
FROM exact_pairs p
WHERE c.id = p.card_id AND c.asset_version_id IS NULL;

CREATE INDEX IF NOT EXISTS ix_fish_cards_asset_version_id
    ON fish_cards (asset_version_id);
-- Do not make historical ambiguous data un-migratable. Constraints are
-- installed when the existing rows satisfy them; conflicting rows remain
-- visible in the workspace for explicit operator repair.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM fish_cards
        WHERE asset_version_id IS NOT NULL
        GROUP BY asset_version_id HAVING COUNT(*) > 1
    ) THEN
        CREATE UNIQUE INDEX IF NOT EXISTS uq_fish_cards_asset_version_id
            ON fish_cards (asset_version_id) WHERE asset_version_id IS NOT NULL;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM fish_knowledge_asset_versions
        WHERE status = 'ACTIVE' AND asset_role IS NOT NULL
        GROUP BY species_id, asset_role HAVING COUNT(*) > 1
    ) THEN
        CREATE UNIQUE INDEX IF NOT EXISTS uq_fish_knowledge_asset_active_role
            ON fish_knowledge_asset_versions (species_id, asset_role)
            WHERE status = 'ACTIVE' AND asset_role IS NOT NULL;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS fish_card_content_revisions (
    id BIGSERIAL PRIMARY KEY,
    card_id BIGINT NOT NULL REFERENCES fish_cards(id) ON DELETE RESTRICT,
    asset_version_id BIGINT NOT NULL REFERENCES fish_knowledge_asset_versions(id) ON DELETE RESTRICT,
    content_revision INTEGER NOT NULL,
    title VARCHAR(256) NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    image_url TEXT NOT NULL,
    created_by VARCHAR(256) NOT NULL DEFAULT 'admin',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fish_card_content_revision UNIQUE (card_id, content_revision)
);
CREATE INDEX IF NOT EXISTS ix_fish_card_content_revisions_card_id
    ON fish_card_content_revisions (card_id);
CREATE INDEX IF NOT EXISTS ix_fish_card_content_revisions_asset_version_id
    ON fish_card_content_revisions (asset_version_id);

INSERT INTO fish_card_content_revisions
    (card_id, asset_version_id, content_revision, title, description, image_url)
SELECT id, asset_version_id, content_revision, title, description, image_url
FROM fish_cards
WHERE asset_version_id IS NOT NULL
ON CONFLICT (card_id, content_revision) DO NOTHING;

CREATE TABLE IF NOT EXISTS fish_knowledge_publication_audits (
    id BIGSERIAL PRIMARY KEY,
    species_id VARCHAR(128) NOT NULL REFERENCES fish_species(id) ON DELETE RESTRICT,
    asset_role VARCHAR(32) NOT NULL,
    asset_version_id BIGINT NOT NULL REFERENCES fish_knowledge_asset_versions(id) ON DELETE RESTRICT,
    card_id BIGINT REFERENCES fish_cards(id) ON DELETE RESTRICT,
    previous_version_id BIGINT,
    previous_card_id BIGINT,
    publication_status VARCHAR(16) NOT NULL,
    validation_json TEXT NOT NULL DEFAULT '{}',
    actor VARCHAR(256) NOT NULL DEFAULT 'admin',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_publication_audits_species_id
    ON fish_knowledge_publication_audits (species_id);
CREATE INDEX IF NOT EXISTS ix_fish_knowledge_publication_audits_asset_version_id
    ON fish_knowledge_publication_audits (asset_version_id);
