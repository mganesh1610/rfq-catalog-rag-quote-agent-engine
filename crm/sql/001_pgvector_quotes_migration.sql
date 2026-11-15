-- PostgreSQL 16 + pgvector + GIN Full-Text BM25 Hybrid Retrieval & Quote Governance Migration
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS catalog_skus (
    sku_id UUID PRIMARY KEY,
    canonical_mpn VARCHAR(64) NOT NULL UNIQUE,
    manufacturer VARCHAR(128) NOT NULL,
    category VARCHAR(128) NOT NULL,
    spec_summary TEXT NOT NULL,
    search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', canonical_mpn || ' ' || manufacturer || ' ' || spec_summary)
    ) STORED,
    embedding vector(384) NOT NULL,
    min_unit_cost NUMERIC(12, 2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_skus_tsv ON catalog_skus USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS idx_catalog_skus_mpn_trgm ON catalog_skus USING GIN (canonical_mpn gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_catalog_skus_hnsw ON catalog_skus USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS quotes (
    quote_id VARCHAR(32) PRIMARY KEY,
    rfq_id VARCHAR(32) NOT NULL UNIQUE,
    source_format VARCHAR(32) NOT NULL,
    ship_to_address TEXT NOT NULL,
    confidence_score NUMERIC(5, 4) NOT NULL,
    review_status VARCHAR(32) NOT NULL CHECK (
        review_status IN ('READY_FOR_SALES_SIGNOFF', 'ESCALATED_HUMAN_REVIEW', 'APPROVED', 'REJECTED')
    ),
    total_quote_usd NUMERIC(14, 2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS quote_audit_trails (
    audit_id BIGSERIAL PRIMARY KEY,
    quote_id VARCHAR(32) NOT NULL REFERENCES quotes(quote_id) ON DELETE CASCADE,
    actor_user_id VARCHAR(64) NOT NULL,
    actor_role VARCHAR(32) NOT NULL,
    action VARCHAR(64) NOT NULL,
    reviewer_feedback JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quotes_review_status ON quotes (review_status, confidence_score);
CREATE INDEX IF NOT EXISTS idx_audit_quote_id ON quote_audit_trails (quote_id, created_at DESC);
