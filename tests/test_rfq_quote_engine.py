import json
import unittest
from pathlib import Path

from src.fastapi_celery_worker import (
    RFQJobPayload,
    build_celery_task_envelope,
    build_openapi_route_manifest,
)
from src.rfq_quote_agent_pipeline import (
    BIO_LABELS,
    HybridCatalogSearchEngine,
    LinearChainSequenceLabeler,
    canonical_mpn,
    extract_entities_from_bio,
    normalize_ocr_mpn,
    select_best_distributor_offer,
)


class TestRFQQuoteEngine(unittest.TestCase):
    def test_ocr_mpn_normalization(self) -> None:
        self.assertEqual(normalize_ocr_mpn("COMP-1O5B9I2A,"), "COMP-105B912A")
        self.assertEqual(canonical_mpn("1e9e8ef04dbcff4541ed26657ea517e5", "informatica_acessorios"), "INFO-1E9E8EF0")

    def test_sequence_labeler_learns_rfq_entities(self) -> None:
        train_samples = [
            (
                ["RFQ", "101", "Tempe", "AZ", "85281", "ITEM", "INFO-1E9E8EF0", "MFR", "Info", "Systems", "QTY", "12"],
                ["O", "O", "B-SHIP", "I-SHIP", "I-SHIP", "O", "B-MPN", "O", "B-MFR", "I-MFR", "O", "B-QTY"],
            ),
            (
                ["Email", "102", "Phoenix", "AZ", "85004", "need", "5", "units", "Tele", "Systems", "MPN", "TELE-8A2B3C4D"],
                ["O", "O", "B-SHIP", "I-SHIP", "I-SHIP", "O", "B-QTY", "O", "B-MFR", "I-MFR", "O", "B-MPN"],
            ),
        ]
        model = LinearChainSequenceLabeler(BIO_LABELS)
        model.fit(train_samples, epochs=12, lr=0.3, seed=7)
        preds, conf = model.decode_with_confidence(train_samples[0][0])
        self.assertEqual(preds, train_samples[0][1])
        self.assertGreater(conf, 0.85)
        entities = extract_entities_from_bio(train_samples[0][0], preds)
        self.assertEqual(entities["mpns"], ["INFO-1E9E8EF0"])
        self.assertEqual(entities["quantities"], [12])
        self.assertEqual(entities["ship_to"], "Tempe AZ 85281")

    def test_hybrid_catalog_deduplication_and_ocr_recovery(self) -> None:
        listings = [
            {
                "mpn": "INFO-1050A9BC",
                "product_id": "p1",
                "manufacturer": "Info Systems",
                "category": "networking switch",
                "spec_summary": "24 port managed rack switch",
                "offer": {
                    "distributor_id": "DIST01",
                    "distributor_tier": "TierA_GlobalDist",
                    "distributor_state": "CA",
                    "unit_price": 240.0,
                    "freight_value": 18.0,
                },
            },
            {
                "mpn": "INFO-1050A9BC",
                "product_id": "p1",
                "manufacturer": "Info Systems",
                "category": "networking switch",
                "spec_summary": "24 port managed rack switch",
                "offer": {
                    "distributor_id": "DIST02",
                    "distributor_tier": "TierB_EnterpriseDist",
                    "distributor_state": "AZ",
                    "unit_price": 232.0,
                    "freight_value": 12.0,
                },
            },
            {
                "mpn": "STOR-9988FFE0",
                "product_id": "p2",
                "manufacturer": "Storage Systems",
                "category": "nvme ssd drive",
                "spec_summary": "enterprise nvme 3.84tb",
                "offer": {
                    "distributor_id": "DIST03",
                    "distributor_tier": "TierC_SpecialtyDist",
                    "distributor_state": "TX",
                    "unit_price": 410.0,
                    "freight_value": 14.0,
                },
            },
        ]
        engine = HybridCatalogSearchEngine(dim=64)
        stats = engine.build_index(listings)
        self.assertEqual(stats["canonical_deduplicated_skus"], 2)
        self.assertEqual(stats["duplicates_consolidated"], 1)

        # Query with OCR-corrupted O/I instead of 0/1
        hits = engine.search_hybrid_reranked(
            "INFO-IOSOA9BC Info Systems networking",
            mpn_hint="INFO-IOSOA9BC",
            mfr_hint="Info Systems",
            top_k=2,
        )
        self.assertEqual(hits[0][0]["mpn"], "INFO-1050A9BC")
        best_offer = select_best_distributor_offer(hits[0][0], quantity=4, ship_state="AZ")
        self.assertEqual(best_offer["distributor_id"], "DIST02")

    def test_celery_worker_and_benchmark_artifact(self) -> None:
        payload = RFQJobPayload(
            job_id="job-001",
            rfq_id="RFQ-10001",
            source_format="pdf_table",
            s3_minio_object_key="rfq-inbox/2026/11/RFQ-10001.pdf",
            idempotency_key="idem-10001",
            requested_by_user_id="usr_sales_04",
        )
        envelope = build_celery_task_envelope(payload)
        self.assertTrue(envelope["acks_late"])
        self.assertEqual(len(build_openapi_route_manifest()), 4)

        metrics_path = Path(__file__).resolve().parents[1] / "outputs" / "benchmark_metrics.json"
        self.assertTrue(metrics_path.exists())
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["source_rows_total"], 347578)
        self.assertGreater(
            metrics["hybrid_catalog_rag_metrics"]["hybrid_rrf_reranked_top1_recall"],
            metrics["hybrid_catalog_rag_metrics"]["bm25_only_top1_recall"],
        )


if __name__ == "__main__":
    unittest.main()
