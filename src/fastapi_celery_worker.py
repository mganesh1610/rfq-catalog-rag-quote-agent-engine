"""
Asynchronous FastAPI + Celery + Redis Background Worker Service for RFQ Extraction & Crawling.
Exposes idempotent REST endpoints for RFQ PDF/Spreadsheet/Email ingestion, distributor catalog
polling, and confidence-gated quote preparation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class RFQJobPayload:
    job_id: str
    rfq_id: str
    source_format: str
    s3_minio_object_key: str
    idempotency_key: str
    requested_by_user_id: str


def build_celery_task_envelope(payload: RFQJobPayload) -> Dict[str, Any]:
    """Constructs a retry-safe Celery + Redis task descriptor with exponential backoff."""
    return {
        "queue": "rfq.extraction.high_priority",
        "broker": "redis://redis-cluster.internal:6379/0",
        "result_backend": "redis://redis-cluster.internal:6379/1",
        "task_name": "workers.extract_and_prepare_quote",
        "acks_late": True,
        "max_retries": 3,
        "retry_backoff_sec": 5,
        "payload": asdict(payload),
    }


def build_openapi_route_manifest() -> List[Dict[str, str]]:
    return [
        {"method": "POST", "path": "/api/v1/rfqs/ingest", "auth_scope": "quotes:write"},
        {"method": "GET", "path": "/api/v1/rfqs/{job_id}/status", "auth_scope": "quotes:read"},
        {"method": "POST", "path": "/api/v1/catalog/poll-distributors", "auth_scope": "catalog:sync"},
        {"method": "POST", "path": "/api/v1/quotes/{quote_id}/review", "auth_scope": "quotes:approve"},
    ]
