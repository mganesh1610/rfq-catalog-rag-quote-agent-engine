# RFQ Document Extraction, Hybrid Catalog RAG & Multi-Distributor Quote Agent Engine

End-to-end AI engineering system transforming noisy procurement RFQ PDFs, spreadsheets, and emails into structured, confidence-scored multi-distributor quotes with human-in-the-loop CRM review and immutable RBAC audit trails.

## Key Capabilities & Verified Benchmark Metrics (`outputs/benchmark_metrics.json`)

Executed against **347,578 SHA-256-verified public commercial catalog, distributor, order, and shipping records** (`32,951` catalog products, `3,095` distributor profiles, `112,650` historical quote/order line items, `99,441` order headers, and `99,441` customer shipping addresses):

- **Fast BIO Sequence-Labeling NER + Confidence-Gated LLM Routing:** Linear-chain Viterbi sequence tagger (`B-MPN`, `B-QTY`, `B-MFR`, `I-MFR`, `B-SHIP`, `I-SHIP`) trained on `1,680` multi-format RFQs and evaluated on `720` held-out test RFQs (`749` line items) achieves **`98.26%` exact part-number (MPN) accuracy** and **`100.0%` quantity & shipping-destination accuracy** at **`0.082 ms` `p50` / `0.111 ms` `p95` latency**, escalating only **`1.81%`** smudged/low-confidence documents to LLM fallback (**`96.94%` inference cost reduction**).
- **Canonical SKU Deduplication & Hybrid Catalog RAG (`PostgreSQL + pgvector + BM25 + Reranker`):** Consolidates **`18,000` raw distributor listings into `9,611` canonical SKUs** (`8,389` duplicate distributor listings merged) and fuses Okapi BM25 lexical search with subword dense vector retrieval via Reciprocal Rank Fusion (`RRF`) + cross-feature reranking—lifting **Top-1 catalog match recall from `76.90%` (BM25-only) to `98.66%` (`+21.76 pp` lift; `98.93%` Recall@5)** at **`2.62 ms` `p50` / `4.74 ms` `p95` latency**.
- **Tool-Using Multi-Distributor Quote Agent & Full-Stack CRM Governance:** Compares landed unit cost + regional freight across wholesale distributor tiers, routes `707/720` high-confidence quotes (`>= 0.88`) to sales sign-off and `13/720` OCR-degraded quotes to human review (`SQLite integrity_check = ok`), and ships production **FastAPI + Celery/Redis** worker descriptors (`src/fastapi_celery_worker.py`), **PostgreSQL `pgvector` + `pg_trgm` SQL migrations** (`crm/sql/001_pgvector_quotes_migration.sql`), **Prisma + Express TypeScript** RBAC endpoints (`crm/prisma/schema.prisma`, `crm/src/quoteRoutes.ts`), and a **React TypeScript** review interface (`crm/ui/QuoteReviewPanel.tsx`).

## Verified Public Data Provenance

| Source Table | Rows | SHA-256 Checksum |
| :--- | ---: | :--- |
| `olist_products_dataset.csv` | `32,951` | `3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc` |
| `olist_sellers_dataset.csv` | `3,095` | `31eb9bd50d684526a78f2389413ce61841e97692de7b828fe7d956b534586312` |
| `olist_order_items_dataset.csv` | `112,650` | `4f6abdbbc94036d0df4a76fa0520c072e31a40119d70f7f370fba1e2285d2bcb` |
| `olist_orders_dataset.csv` | `99,441` | `8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf` |
| `olist_customers_dataset.csv` | `99,441` | `c26c17f59f3027a6a0dbeb8f1fa373c38f3323ecc84f225a8f4820e5bb288df8` |

- **Upstream Public Repository:** https://github.com/olist/work-at-olist-data/tree/master/datasets
- **Backup Public Datasets:** UCI Online Retail Commercial Invoice Dataset (`541,909` line items, https://archive.ics.uci.edu/dataset/352/online+retail) and U.S. GSA SAM.gov Contract Opportunities & Catalog Data (https://sam.gov/data-services).

## Quickstart

```bash
RFQ_DATA_DIR=/path/to/olist_csvs PYTHONPATH=. python3 src/rfq_quote_agent_pipeline.py
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py" -v
```
