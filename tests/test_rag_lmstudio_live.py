import os
import time
from pathlib import Path

import pytest

from core.llm_client import LLMClient
from core.models import AIProvider, GeneratorConfig, RagConfig
from core.rag.service import RagService


@pytest.mark.integration
def test_live_lmstudio_rag_with_benefits_email_pdf():
    if os.getenv("RUN_LIVE_LMSTUDIO_RAG") != "1":
        pytest.skip("Set RUN_LIVE_LMSTUDIO_RAG=1 to run live LM Studio + RAG test")

    requested_pdf = Path(r"C:\Users\longs\Documents\GitHub\Synthesizer\examples\benefits_email_narative.pdf")
    if requested_pdf.exists():
        pdf_path = str(requested_pdf)
    else:
        root = Path(__file__).resolve().parent.parent
        fallback = root / "examples" / "benefits_email_narative.pdf"
        if not fallback.exists():
            pytest.skip("benefits_email_narative.pdf was not found in examples/")
        pdf_path = str(fallback)
    collection = f"lmstudio_live_{int(time.time())}"

    rag_cfg = RagConfig(
        enabled=True,
        collection_name=collection,
        top_k=3,
        min_score=0.15,
        max_context_chars=2000,
        qdrant_url=os.getenv("QDRANT_URL", ":memory:"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    config = GeneratorConfig(
        model_id="gpt-oss-20b",
        provider=AIProvider.LM_STUDIO,
        rag=rag_cfg,
    )

    client = LLMClient(config)
    rag = RagService(
        collection_name=rag_cfg.collection_name,
        qdrant_url=rag_cfg.qdrant_url,
        qdrant_api_key=rag_cfg.qdrant_api_key,
        embedding_model=rag_cfg.embedding_model,
        top_k=rag_cfg.top_k,
        min_score=rag_cfg.min_score,
        max_context_chars=rag_cfg.max_context_chars,
    )
    client.set_rag_service(rag)

    report = rag.ingest_documents([pdf_path], force_reindex=True)
    assert report.vectors_upserted > 0, f"No vectors ingested from {pdf_path}. Report: {report.model_dump()}"

    question = (
        "From the benefits email narrative, what is one specific email request "
        "or employee issue and what benefit/action was suggested?"
    )
    context = client.retrieve_context(question)
    assert context.strip(), "Retrieved context is empty"
    assert "email" in context.lower() or "benefit" in context.lower(), "Retrieved context does not look email-related"

    prompt = (
        "Use only the provided context to answer. If context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer in 3 short bullet points and include one quoted phrase from context."
    )
    response = client.generate_completion(prompt)
    assert response and str(response).strip(), "LM Studio returned empty response"

    rag_stats = client.get_rag_stats()
    assert rag_stats["queries"] >= 1
    assert rag_stats["queries_with_hits"] >= 1
    assert rag_stats["total_context_chars"] > 0
