import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.app_config import (
    build_generator_config,
    default_ui_values,
    normalize_loaded_config,
    pages_label_from_target_words,
    resolve_document_target_words,
    serialize_ui_config,
)
from core.controller import GeneratorController
from core.document_engine.orchestrator import DocumentOrchestrator
from core.document_engine.models import DocumentGenerationOptions, DocumentMode
from core.models import AIProvider
from web_ui.adapters import field_records_to_grid_dataframe
from web_ui.actions import config_actions
from web_ui.state import new_session_state


def test_build_generator_config_maps_web_values_to_runtime_models():
    config = build_generator_config(
        {
            "model_id": "local-model",
            "provider": AIProvider.LM_STUDIO.value,
            "doc_mode": "File-based",
            "doc_pages": "3 pages",
            "rag_backend": "Native",
            "num_rows": 12,
        },
        include_existing_data=False,
    )

    assert config.provider == AIProvider.LM_STUDIO
    assert config.num_rows == 12
    assert config.rag is not None
    assert config.rag.backend.value == "Native"
    assert config.document_engine is not None
    assert config.document_engine.mode == "strict_grounded"
    assert config.document_engine.target_words == 1500


def test_web_config_roundtrip_keeps_browser_facing_labels():
    payload = serialize_ui_config(
        {
            "model_id": "demo-model",
            "provider": AIProvider.OPENAI.value,
            "api_key": "secret",
            "doc_mode": "Creative",
            "doc_pages": "2 pages",
            "quick_qa_mode": "Pinpoint Quick",
            "ocr_dpi": 200,
            "summary_top_k": 4,
            "graph_enabled": False,
        },
        columns=[],
    )

    normalized = normalize_loaded_config(payload)

    assert normalized["model_id"] == "demo-model"
    assert normalized["provider"] == AIProvider.OPENAI.value
    assert normalized["doc_mode"] == "Creative"
    assert normalized["doc_pages"] == "2 pages"
    assert normalized["quick_qa_mode"] == "Pinpoint Quick"
    assert normalized["ocr_dpi"] == 200
    assert normalized["summary_top_k"] == 4
    assert normalized["graph_enabled"] is False


def test_custom_document_page_inputs_are_supported():
    assert resolve_document_target_words("12") == 6000
    assert resolve_document_target_words("12 pages") == 6000
    assert resolve_document_target_words("7-page draft") == 3500
    assert pages_label_from_target_words(6000) == "12 pages"


def test_loaded_config_preserves_custom_document_page_count():
    normalized = normalize_loaded_config({"document_engine": {"target_words": 6000}})
    assert normalized["doc_pages"] == "12 pages"


def test_build_generator_config_maps_advanced_rag_values_to_runtime_models():
    config = build_generator_config(
        {
            "rag_backend": "Native",
            "ocr_dpi": 175,
            "ocr_max_pages": 12,
            "hybrid_search_enabled": False,
            "rerank_enabled": False,
            "summary_first_enabled": False,
            "summary_top_k": 2,
            "dense_top_k": 9,
            "lexical_top_k": 7,
            "parent_context_enabled": False,
            "parent_context_max_chars": 900,
            "graph_enabled": False,
            "graph_hops": 2,
            "graph_source_boost": 0.15,
            "late_interaction_enabled": False,
            "late_interaction_weight": 0.35,
        },
        include_existing_data=False,
    )

    assert config.rag is not None
    assert config.rag.backend.value == "Native"
    assert config.rag.ocr_dpi == 175
    assert config.rag.ocr_max_pages == 12
    assert config.rag.hybrid_search_enabled is False
    assert config.rag.rerank_enabled is False
    assert config.rag.summary_first_enabled is False
    assert config.rag.summary_top_k == 2
    assert config.rag.dense_top_k == 9
    assert config.rag.lexical_top_k == 7
    assert config.rag.parent_context_enabled is False
    assert config.rag.parent_context_max_chars == 900
    assert config.rag.graph_enabled is False
    assert config.rag.graph_hops == 2
    assert config.rag.graph_source_boost == 0.15
    assert config.rag.late_interaction_enabled is False
    assert config.rag.late_interaction_weight == 0.35


def test_creative_polish_fallback_title_does_not_force_report_suffix():
    controller = GeneratorController()
    controller.llm_client = None

    result = controller._polish_document_for_publish(
        title="Write a story based on these positions",
        prompt="Write a story based on these positions of a couple in a club",
        text="Opening scene text.",
        audience="Adults",
        tone="erotic",
        target_words=900,
        mode=DocumentMode.PURE,
    )

    assert "Report" not in result["title"]
    assert result["text"] == "Opening scene text."


def test_creative_outline_fallback_uses_scene_structure():
    fake_llm = SimpleNamespace(generate_completion=lambda prompt, system_prompt=None: "not json")
    orchestrator = DocumentOrchestrator(fake_llm)

    outline = orchestrator._build_outline(
        DocumentGenerationOptions(
            prompt="Write an erotic story based on these positions",
            audience="Adults",
            tone="erotic",
            mode=DocumentMode.PURE,
            target_words=900,
        )
    )

    titles = [section.title for section in outline.sections]
    assert titles[0] == "Opening Scene"
    assert "Recommendations" not in titles


def test_reset_config_restores_defaults_and_clears_web_session_state():
    session = new_session_state(startup_collection_name="synthesizer_session_test")
    session.imported_data = [{"name": "Alice"}]
    session.fields = [{"name": "email", "type": "Short Text", "prompt_instruction": "Work email"}]
    session.rag_files = ["C:\\temp\\sample.pdf"]
    session.file_chat_history = [{"role": "assistant", "content": "Preview"}]
    session.generated_rows = [{"name": "Alice"}]
    session.latest_downloads = {"document_pdf": "x.pdf"}

    result = config_actions.reset_config(session)

    updated_session = result[0]
    assert updated_session.imported_data == []
    assert updated_session.fields == []
    assert updated_session.rag_files == []
    assert updated_session.file_chat_history == []
    assert updated_session.generated_rows == []
    assert updated_session.latest_downloads == {}
    assert result[1]["value"] == "local-model"
    assert result[12] == "synthesizer_session_test"
    assert result[51] == "Mask likely personal values"
    assert "No imported columns yet." in result[53]
    assert result[55]["value"].empty
    assert result[56] == "Start from scratch or import a CSV/JSON file to use existing columns as a base."
    assert "No rows yet." in result[57]
    assert result[61] is None
    assert result[70] == "Search admin messages will appear here."
    assert "Settings reset to the default starting state." in result[71]


def test_get_search_status_formats_rag_snapshot(monkeypatch):
    class FakeController:
        def get_rag_status(self):
            return {
                "enabled": True,
                "collection_name": "demo_collection",
                "collection_size": 42,
                "top_k": 5,
                "min_score": 0.25,
                "ocr_mode": "auto",
                "parser_mode": "docling",
                "hybrid_search_enabled": True,
                "rerank_enabled": True,
                "graph_enabled": True,
                "late_interaction_enabled": False,
            }

    monkeypatch.setattr(config_actions, "_build_runtime_controller_from_values", lambda values: FakeController())

    session = new_session_state()
    session, status_markdown, activity, debug = config_actions.get_search_status(
        session,
        "local-model",
        AIProvider.LM_STUDIO.value,
        "",
        "",
        "",
        0.15,
        0.60,
        10,
        0.85,
        50,
        "LlamaIndex",
        "demo_collection",
        5,
        0.25,
        3000,
        "BAAI/bge-small-en-v1.5",
        "",
        ":memory:",
        "",
        "off",
        150,
        20,
        8,
        18,
        2.5,
        60,
        4000,
        "auto",
        True,
        True,
        True,
        3,
        12,
        12,
        True,
        1200,
        True,
        1,
        0.08,
        True,
        0.2,
        "Broader Analysis",
        "Balanced",
        "Let AI decide",
        "Fast",
        "General",
        "professional",
        False,
        True,
        3,
    )

    assert "demo_collection" in status_markdown
    assert "42" in status_markdown
    assert "Fetched search status" in activity
    assert "Debug Details" in debug


def test_clear_search_index_resets_files_workspace(monkeypatch):
    class FakeController:
        def clear_rag_collection(self):
            return None

    monkeypatch.setattr(config_actions, "_build_runtime_controller_from_values", lambda values: FakeController())

    session = new_session_state()
    session.files_mode = "Quick Q&A"
    session.rag_files = ["C:\\temp\\sample.pdf"]
    session.file_chat_history = [{"role": "assistant", "content": "Grounded answer"}]

    session, files_table, selector, files_chat, files_status, rag_status, activity, debug = config_actions.clear_search_index(
        session,
        "local-model",
        AIProvider.LM_STUDIO.value,
        "",
        "",
        "",
        0.15,
        0.60,
        10,
        0.85,
        50,
        "LlamaIndex",
        "demo_collection",
        5,
        0.25,
        3000,
        "BAAI/bge-small-en-v1.5",
        "",
        ":memory:",
        "",
        "off",
        150,
        20,
        8,
        18,
        2.5,
        60,
        4000,
        "auto",
        True,
        True,
        True,
        3,
        12,
        12,
        True,
        1200,
        True,
        1,
        0.08,
        True,
        0.2,
        "Broader Analysis",
        "Balanced",
        "Let AI decide",
        "Fast",
        "General",
        "professional",
        False,
        True,
        3,
    )

    table_df = files_table["value"] if isinstance(files_table, dict) else files_table
    assert session.rag_files == []
    assert session.file_chat_history == []
    assert table_df.empty
    assert selector["value"] is None
    assert "Results will appear here" in files_chat
    assert "Current files: **0**" in files_status
    assert "cleared" in rag_status.lower()
    assert "Search index cleared." in activity
    assert "Files in session: **0**" in debug


def test_save_and_load_config_roundtrips_visible_schema_rows_and_privacy_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config_actions, "EXPORT_DIR", tmp_path)

    session = new_session_state()
    defaults = default_ui_values()
    grid_value = field_records_to_grid_dataframe(
        [
            {
                "name": "sender_email",
                "type": "Short Text",
                "prompt_instruction": "(Imported)",
                "allow_duplicates": False,
            },
            {
                "name": "reply",
                "type": "Long Text",
                "prompt_instruction": "Write a concise benefits-team reply for the imported email.",
                "allow_duplicates": True,
            },
        ]
    )

    save_result = config_actions.save_config_file(
        session=session,
        grid_value=grid_value,
        row_editor_choice="Row 2",
        row_editor_name="reply",
        row_editor_type="Long Text",
        row_editor_prompt="Write a concise benefits-team reply for the imported email.",
        row_editor_allow_duplicates=True,
        model_id=defaults["model_id"],
        provider=defaults["provider"],
        api_key=defaults["api_key"],
        azure_endpoint=defaults["azure_endpoint"],
        azure_deployment=defaults["azure_deployment"],
        input_price_per_1m=defaults["input_price_per_1m"],
        output_price_per_1m=defaults["output_price_per_1m"],
        num_rows=defaults["num_rows"],
        similarity_threshold=defaults["similarity_threshold"],
        max_retries=defaults["max_retries"],
        rag_backend=defaults["rag_backend"],
        collection_name=defaults["collection_name"],
        top_k=defaults["top_k"],
        min_score=defaults["min_score"],
        max_context_chars=defaults["max_context_chars"],
        embedding_model=defaults["embedding_model"],
        source_filter=defaults["source_filter"],
        qdrant_url=defaults["qdrant_url"],
        qdrant_api_key=defaults["qdrant_api_key"],
        ocr_mode=defaults["ocr_mode"],
        ocr_dpi=defaults["ocr_dpi"],
        ocr_max_pages=defaults["ocr_max_pages"],
        ocr_max_regions_per_page=defaults["ocr_max_regions_per_page"],
        ocr_region_padding_px=defaults["ocr_region_padding_px"],
        ocr_gap_multiplier=defaults["ocr_gap_multiplier"],
        ocr_min_extracted_chars=defaults["ocr_min_extracted_chars"],
        ocr_timeout_ms_per_page=defaults["ocr_timeout_ms_per_page"],
        parser_mode=defaults["parser_mode"],
        hybrid_search_enabled=defaults["hybrid_search_enabled"],
        rerank_enabled=defaults["rerank_enabled"],
        summary_first_enabled=defaults["summary_first_enabled"],
        summary_top_k=defaults["summary_top_k"],
        dense_top_k=defaults["dense_top_k"],
        lexical_top_k=defaults["lexical_top_k"],
        parent_context_enabled=defaults["parent_context_enabled"],
        parent_context_max_chars=defaults["parent_context_max_chars"],
        graph_enabled=defaults["graph_enabled"],
        graph_hops=defaults["graph_hops"],
        graph_source_boost=defaults["graph_source_boost"],
        late_interaction_enabled=defaults["late_interaction_enabled"],
        late_interaction_weight=defaults["late_interaction_weight"],
        quick_qa_mode=defaults["quick_qa_mode"],
        doc_mode=defaults["doc_mode"],
        doc_pages=defaults["doc_pages"],
        doc_quality=defaults["doc_quality"],
        doc_audience=defaults["doc_audience"],
        doc_tone=defaults["doc_tone"],
        doc_chart_enabled=defaults["doc_chart_enabled"],
        doc_flow_enabled=defaults["doc_flow_enabled"],
        doc_max_charts=defaults["doc_max_charts"],
        import_privacy_mode="Mask likely personal values",
    )

    config_path = save_result[1]
    assert config_path
    assert os.path.exists(config_path)
    assert session.fields[1]["name"] == "reply"
    assert session.fields[1]["prompt_instruction"] == "Write a concise benefits-team reply for the imported email."

    saved_payload = json.loads((tmp_path / os.path.basename(config_path)).read_text(encoding="utf-8"))
    assert saved_payload["import_privacy_mode"] == "Mask likely personal values"
    assert saved_payload["columns"][1]["name"] == "reply"
    assert saved_payload["columns"][1]["prompt_instruction"] == "Write a concise benefits-team reply for the imported email."

    reloaded_session = new_session_state()
    load_result = config_actions.load_config_file(reloaded_session, config_path)

    assert reloaded_session.import_privacy_mode == "Mask likely personal values"
    assert reloaded_session.fields[0]["name"] == "sender_email"
    assert reloaded_session.fields[1]["name"] == "reply"
    assert reloaded_session.fields[1]["prompt_instruction"] == "Write a concise benefits-team reply for the imported email."
    assert load_result[51] == "Mask likely personal values"
    assert load_result[52].iloc[1]["name"] == "reply"
    assert load_result[52].iloc[1]["prompt_instruction"] == "Write a concise benefits-team reply for the imported email."
    assert "Loaded config with **2** schema row(s)." == load_result[53]


def test_save_config_captures_pending_row_editor_changes_not_yet_applied_to_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(config_actions, "EXPORT_DIR", tmp_path)

    session = new_session_state()
    defaults = default_ui_values()
    grid_value = field_records_to_grid_dataframe(
        [
            {
                "name": "sender_email",
                "type": "Short Text",
                "prompt_instruction": "(Imported)",
                "allow_duplicates": False,
            },
            {
                "name": "",
                "type": "Short Text",
                "prompt_instruction": "",
                "allow_duplicates": False,
            },
        ]
    )

    save_result = config_actions.save_config_file(
        session=session,
        grid_value=grid_value,
        row_editor_choice="Row 2",
        row_editor_name="summary",
        row_editor_type="Long Text",
        row_editor_prompt="Summarize the imported email in one sentence.",
        row_editor_allow_duplicates=True,
        model_id=defaults["model_id"],
        provider=defaults["provider"],
        api_key=defaults["api_key"],
        azure_endpoint=defaults["azure_endpoint"],
        azure_deployment=defaults["azure_deployment"],
        input_price_per_1m=defaults["input_price_per_1m"],
        output_price_per_1m=defaults["output_price_per_1m"],
        num_rows=defaults["num_rows"],
        similarity_threshold=defaults["similarity_threshold"],
        max_retries=defaults["max_retries"],
        rag_backend=defaults["rag_backend"],
        collection_name=defaults["collection_name"],
        top_k=defaults["top_k"],
        min_score=defaults["min_score"],
        max_context_chars=defaults["max_context_chars"],
        embedding_model=defaults["embedding_model"],
        source_filter=defaults["source_filter"],
        qdrant_url=defaults["qdrant_url"],
        qdrant_api_key=defaults["qdrant_api_key"],
        ocr_mode=defaults["ocr_mode"],
        ocr_dpi=defaults["ocr_dpi"],
        ocr_max_pages=defaults["ocr_max_pages"],
        ocr_max_regions_per_page=defaults["ocr_max_regions_per_page"],
        ocr_region_padding_px=defaults["ocr_region_padding_px"],
        ocr_gap_multiplier=defaults["ocr_gap_multiplier"],
        ocr_min_extracted_chars=defaults["ocr_min_extracted_chars"],
        ocr_timeout_ms_per_page=defaults["ocr_timeout_ms_per_page"],
        parser_mode=defaults["parser_mode"],
        hybrid_search_enabled=defaults["hybrid_search_enabled"],
        rerank_enabled=defaults["rerank_enabled"],
        summary_first_enabled=defaults["summary_first_enabled"],
        summary_top_k=defaults["summary_top_k"],
        dense_top_k=defaults["dense_top_k"],
        lexical_top_k=defaults["lexical_top_k"],
        parent_context_enabled=defaults["parent_context_enabled"],
        parent_context_max_chars=defaults["parent_context_max_chars"],
        graph_enabled=defaults["graph_enabled"],
        graph_hops=defaults["graph_hops"],
        graph_source_boost=defaults["graph_source_boost"],
        late_interaction_enabled=defaults["late_interaction_enabled"],
        late_interaction_weight=defaults["late_interaction_weight"],
        quick_qa_mode=defaults["quick_qa_mode"],
        doc_mode=defaults["doc_mode"],
        doc_pages=defaults["doc_pages"],
        doc_quality=defaults["doc_quality"],
        doc_audience=defaults["doc_audience"],
        doc_tone=defaults["doc_tone"],
        doc_chart_enabled=defaults["doc_chart_enabled"],
        doc_flow_enabled=defaults["doc_flow_enabled"],
        doc_max_charts=defaults["doc_max_charts"],
        import_privacy_mode="Mask likely personal values",
    )

    config_path = save_result[1]
    saved_payload = json.loads((tmp_path / os.path.basename(config_path)).read_text(encoding="utf-8"))

    assert session.fields[1]["name"] == "summary"
    assert session.fields[1]["type"] == "Long Text"
    assert session.fields[1]["prompt_instruction"] == "Summarize the imported email in one sentence."
    assert saved_payload["columns"][1]["name"] == "summary"
    assert saved_payload["columns"][1]["prompt_instruction"] == "Summarize the imported email in one sentence."
