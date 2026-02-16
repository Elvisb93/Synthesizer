from pathlib import Path
import time

from core.rag.service import RagService


def test_reingest_after_clear_does_not_skip_cached_file():
    root = Path(__file__).resolve().parent.parent
    pdf_path = root / "examples" / "test coffee.pdf"
    assert pdf_path.exists()

    svc = RagService(
        collection_name=f"test_reingest_{int(time.time())}",
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=3,
        min_score=0.1,
        max_context_chars=1000,
    )

    first = svc.ingest_documents([str(pdf_path)], force_reindex=True)
    assert first.files_processed == 1
    assert first.vectors_upserted > 0

    svc.clear_collection()

    second = svc.ingest_documents([str(pdf_path)], force_reindex=False)
    assert second.files_processed == 1
    assert second.files_skipped == 0
    assert second.vectors_upserted > 0
