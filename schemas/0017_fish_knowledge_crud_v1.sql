-- YuJian Fish Knowledge CMS CRUD V1.
-- Species deletion is a soft delete so existing content remains auditable.

ALTER TABLE fish_species
    DROP CONSTRAINT IF EXISTS ck_fish_species_status;

ALTER TABLE fish_species
    ADD CONSTRAINT ck_fish_species_status
    CHECK (status IN ('ACTIVE', 'DRAFT', 'DELETED'));
