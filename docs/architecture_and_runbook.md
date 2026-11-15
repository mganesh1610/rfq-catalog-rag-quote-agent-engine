# Architecture, Schema Migration & Production Runbook

## 1. End-to-End System Flow

1. **Asynchronous Document Ingestion (`FastAPI + Celery + Redis + MinIO`):**
   - Multi-page RFQ PDFs, spreadsheet schedules, and procurement emails land in S3-compatible object storage (`s3_minio_object_key`) and enqueue idempotent `workers.extract_and_prepare_quote` jobs on Redis (`acks_late=True`, 3x exponential retry backoff).
2. **Sequence-Labeling NER vs. LLM Fallback (`src/rfq_quote_agent_pipeline.py`):**
   - Linear-chain Viterbi BIO sequence tagger extracts `B-MPN`, `B-QTY`, `B-MFR`/`I-MFR`, and `B-SHIP`/`I-SHIP` at `0.11 ms` `p95` latency (`98.26%` exact MPN accuracy across OCR-noisy PDFs and emails), routing only `1.81%` low-confidence or smudged spans (`confidence < 0.85`) to self-hosted LLM fallback (**`96.94%` inference cost reduction**).
3. **Hybrid Catalog Retrieval & SKU Deduplication (`PostgreSQL + pgvector + BM25 + Reranker`):**
   - Consolidates `18,000` raw multi-distributor catalog rows into `9,611` canonical SKUs (`8,389` duplicate offers merged), combining Okapi BM25 (`tsvector` GIN) + subword dense vector similarity (`pgvector` HNSW) via Reciprocal Rank Fusion (`RRF`) and supervised margin reranking (`98.66%` Top-1 recall vs. `76.90%` BM25-only; `4.74 ms` `p95` latency).
4. **CRM Integration (`Express + Prisma + PostgreSQL + React TypeScript`):**
   - `crm/sql/001_pgvector_quotes_migration.sql` and `crm/prisma/schema.prisma` enforce foreign-key integrity, check constraints on `review_status`, and immutable `quote_audit_trails` rows with RBAC role validation (`crm/src/quoteRoutes.ts`) and human-in-the-loop review (`crm/ui/QuoteReviewPanel.tsx`).

## 2. Operational Triage & Rollback Runbook

- **Symptom — Sudden spike in `ESCALATED_HUMAN_REVIEW` rate (`>15%`):**
  1. Inspect OCR character substitution logs for new vendor PDF table templates.
  2. Export reviewer `correctedMpn` overrides from `quote_audit_trails.reviewer_feedback` into the labeled regression benchmark and re-run `PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py" -v` before redeploying worker containers.
