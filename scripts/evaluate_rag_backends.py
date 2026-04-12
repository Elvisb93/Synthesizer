import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.controller import GeneratorController
from core.models import AIProvider, DocumentEngineConfig, GeneratorConfig, RagBackend, RagConfig


def _load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Evaluation spec must be a JSON object.")
    return data


def _resolve_documents(spec: Dict[str, Any], root: Path) -> List[str]:
    documents = spec.get("documents") or []
    resolved = []
    for item in documents:
        raw = str(item).strip()
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        resolved.append(str(candidate))
    if not resolved:
        raise ValueError("Evaluation spec must include at least one document path.")
    return resolved


def _build_config(model: str, backend: RagBackend, collection_name: str) -> GeneratorConfig:
    return GeneratorConfig(
        model_id=model,
        provider=AIProvider.LM_STUDIO,
        rag=RagConfig(
            backend=backend,
            collection_name=collection_name,
            qdrant_url=":memory:",
        ),
        document_engine=DocumentEngineConfig(mode="hybrid", target_words=700),
    )


def _run_backend(model: str, backend: RagBackend, documents: List[str], spec: Dict[str, Any]) -> Dict[str, Any]:
    controller = GeneratorController()
    config = _build_config(model, backend, f"eval_{backend.value.lower()}_{int(time.time())}")
    controller.set_runtime_config(config)

    started = time.perf_counter()
    ingest = controller.ingest_documents(documents, force_reindex=True)
    ingest_seconds = time.perf_counter() - started

    qa_results = []
    for prompt in spec.get("qa_prompts") or []:
        prompt_text = str(prompt).strip()
        if not prompt_text:
            continue
        qa_started = time.perf_counter()
        result = controller.ask_files(prompt_text)
        qa_results.append(
            {
                "prompt": prompt_text,
                "seconds": round(time.perf_counter() - qa_started, 3),
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "response_mode": result.get("response_mode"),
            }
        )

    document_results = []
    for prompt_spec in spec.get("document_prompts") or []:
        if isinstance(prompt_spec, str):
            prompt_text = prompt_spec.strip()
            target_words = 700
            mode = "hybrid"
        else:
            prompt_text = str(prompt_spec.get("prompt", "")).strip()
            target_words = int(prompt_spec.get("target_words", 700) or 700)
            mode = str(prompt_spec.get("mode", "hybrid") or "hybrid")
        if not prompt_text:
            continue
        doc_started = time.perf_counter()
        result = controller.generate_document(
            prompt_text,
            target_words=target_words,
            audience="General",
            tone="professional",
            mode=mode,
            quality_mode="Fast",
            resume=False,
        )
        document_results.append(
            {
                "prompt": prompt_text,
                "seconds": round(time.perf_counter() - doc_started, 3),
                "mode": mode,
                "target_words": target_words,
                "title": result.get("title", ""),
                "final_word_count": result.get("final_word_count"),
                "text_preview": str(result.get("text", ""))[:1600],
                "citations": result.get("citations", [])[:10],
            }
        )

    controller.clear_rag_collection()
    return {
        "backend": backend.value,
        "ingest_seconds": round(ingest_seconds, 3),
        "ingest": ingest,
        "qa_results": qa_results,
        "document_results": document_results,
        "status": controller.get_rag_status(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Native vs LlamaIndex RAG backends on local documents.")
    parser.add_argument("--spec", required=True, help="Path to the evaluation spec JSON.")
    parser.add_argument("--model", required=True, help="LM Studio model id to use for synthesis tasks.")
    parser.add_argument("--output", default="rag_backend_eval_results.json", help="Where to write the result JSON.")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = _load_spec(spec_path)
    documents = _resolve_documents(spec, spec_path.parent)

    results = {
        "model": args.model,
        "spec": str(spec_path),
        "documents": documents,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backends": [],
    }
    for backend in (RagBackend.NATIVE, RagBackend.LLAMA_INDEX):
        results["backends"].append(_run_backend(args.model, backend, documents, spec))

    output_path = Path(args.output).resolve()
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote evaluation results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
