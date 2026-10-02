-- Quick Review Performance V1: durable downstream signal only.
-- The review transaction inserts a PENDING row; the Accepted Pool worker later
-- performs the source scan and GCS materialisation outside the HTTP request.
CREATE TABLE IF NOT EXISTS accepted_pool_sync_requests (
  id BIGSERIAL PRIMARY KEY,
  status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  claimed_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_accepted_pool_sync_requests_status
  ON accepted_pool_sync_requests(status);
CREATE INDEX IF NOT EXISTS idx_accepted_pool_sync_requests_created_at
  ON accepted_pool_sync_requests(created_at);
