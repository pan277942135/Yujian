# Account & Privacy API V1

## Deployment and storage

- Service: `yujian-model-factory-console`
- Project: `gemini-api-project-503706` (`571785698442`)
- Region: `asia-east1`
- UAT URL: `https://yujian-model-factory-console-571785698442.asia-east1.run.app`
- Source: `pan277942135/Yujian` (`app/auth_api.py`, `app/models.py`)
- Database: Cloud SQL PostgreSQL in Cloud Run; SQLite is local-test only.
- Authentication: existing HS256 `Bearer` access token issued by `/api/v1/auth/login`.
- Avatar storage: existing private GCS bucket (`GCS_BUCKET`), served only by an authenticated media gateway.

`schemas/0026_account_privacy_v1.sql` is additive.  Runtime startup also adds
the one new `users.avatar_object_name` column for in-place deployments; the two
new privacy tables are created by SQLAlchemy metadata on startup.

## Endpoint contract

All endpoints below require the existing `Authorization: Bearer <access_token>`.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/me` | Current server-authoritative profile. |
| `PATCH /api/v1/me/profile` | Update trimmed nickname, 1–20 characters. |
| `POST /api/v1/me/avatar` | Validate JPEG/PNG/WEBP (max 5 MB / 12 MP), persist in private GCS, then update the user record. |
| `GET /api/v1/me/avatar/media` | Read the current user's private avatar. |
| `POST /api/v1/auth/change-password` | Verify the existing bcrypt password before replacing its hash. |
| `GET /api/v1/me/privacy` | Read persisted AI model improvement consent (defaults to OFF). |
| `PUT /api/v1/me/privacy/ai-model-improvement` | Persist consent or withdrawal and append an immutable audit row. |

AI consent accepts only `AI_MODEL_IMPROVEMENT_V1` and sources `settings` or
`species_correction_prompt`.  Every change records user ID, enabled value,
version, source, and timestamp.  This setting is the server-side eligibility
source for any future training ingestion; no training path may treat a missing
setting as consent.

## Avatar replacement

The new object is written before the database pointer changes.  After a
successful commit, cleanup best-effort deletes only the previous object beneath
`user_avatars/{user_id}/`; legacy or external avatar URLs are never deleted.
If cleanup fails, the durable current avatar remains valid and the old managed
object can be reclaimed operationally.
