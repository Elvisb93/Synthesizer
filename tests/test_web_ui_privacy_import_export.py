import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm_client import LLMClient
from web_ui.actions import data_actions
from web_ui.adapters import sanitize_imported_records
from web_ui.state import new_session_state


def test_import_data_file_applies_selected_privacy_mode(tmp_path):
    import pandas as pd

    session = new_session_state()
    sample = pd.DataFrame(
        [
            {
                "sender_email": "alice@example.com",
                "sender_name": "Alice Smith",
                "email_text": "Reach me at alice@example.com.",
            }
        ]
    )
    csv_path = tmp_path / "benefits_emails.csv"
    sample.to_csv(csv_path, index=False)

    result = data_actions.import_data_file(session, str(csv_path), "Mask likely personal values")

    updated_session = result[0]
    assert updated_session.import_privacy_mode == "Mask likely personal values"
    assert updated_session.raw_imported_data[0]["sender_email"] == "alice@example.com"
    assert updated_session.imported_data[0]["sender_email"] == "<EMAIL_1>"
    assert "Privacy masking is active" in result[6]
    assert "sender_email" in result[3]["value"]


def test_sanitize_imported_records_masks_likely_personal_values():
    records = [
        {
            "sender_email": "alice@example.com",
            "sender_name": "Alice Smith",
            "email_text": "Contact alice@example.com or +44 20 7946 0958.",
            "department": "Benefits",
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["sender_email"] == "<EMAIL_1>"
    assert masked[0]["sender_name"] == "<NAME_1>"
    assert "<EMAIL_1>" in masked[0]["email_text"]
    assert "<PHONE_" in masked[0]["email_text"]
    assert masked[0]["department"] == "Benefits"


def test_sanitize_imported_records_masks_names_inside_free_text_before_generation():
    records = [
        {
            "sender_name": "Angela Sanchez",
            "email_text": "Please ask Angela Sanchez to confirm whether Sarah Patel is eligible.",
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["sender_name"] == "<NAME_1>"
    assert "<NAME_1>" in masked[0]["email_text"]
    assert "<NAME_1_1>" in masked[0]["email_text"]


def test_sanitize_imported_records_masks_structured_identifiers_and_dates_without_overmasking_categories():
    records = [
        {
            "email_address": "john.doe@healthplan.com",
            "client_name": "John Doe",
            "policy_number": "POL-2023-JD-9876",
            "date_of_birth": "12/04/1985",
            "inquiry_category": "Billing Question",
            "contact_status": "Pending Review",
            "Email_content": "Policy POL-2023-JD-9876 belongs to John Doe. Email john.doe@healthplan.com.",
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["email_address"] == "<EMAIL_1>"
    assert masked[0]["client_name"] == "<NAME_1>"
    assert masked[0]["policy_number"] == "<IDENTIFIER_1_1>"
    assert masked[0]["date_of_birth"] == "<DATE_1_1>"
    assert masked[0]["inquiry_category"] == "Billing Question"
    assert masked[0]["contact_status"] == "Pending Review"
    assert "<IDENTIFIER_1_1>" in masked[0]["Email_content"]
    assert "<NAME_1>" in masked[0]["Email_content"]
    assert "<EMAIL_1>" in masked[0]["Email_content"]


def test_sanitize_imported_records_masks_common_dob_formats_in_free_text():
    records = [
        {
            "client_name": "John Doe",
            "date_of_birth": "12.12.12",
            "email_text": (
                "DOB: 12-12-12. John Doe also listed a previous record as 12 December 2012 "
                "and asked the Benefits Coordinator to review it."
            ),
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["client_name"] == "<NAME_1>"
    assert masked[0]["date_of_birth"] == "<DATE_1_1>"
    assert "12-12-12" not in masked[0]["email_text"]
    assert "12 December 2012" not in masked[0]["email_text"]
    assert "John Doe" not in masked[0]["email_text"]
    assert "Benefits Coordinator" in masked[0]["email_text"]
    assert "<DATE_" in masked[0]["email_text"]
    assert "<NAME_1>" in masked[0]["email_text"]
    assert "<ROLE_" not in masked[0]["email_text"]


def test_sanitize_imported_records_masks_sensitive_spans_in_short_text_columns():
    records = [
        {
            "sender_name": "Kimberly Gonzales",
            "organization": "HealthGuard Solutions",
            "policy_number": "POL-2026-KG-4931",
            "subject": "Re: Policy POL-2026-KG-4931 Inquiry from Kimberly Gonzales",
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["sender_name"] == "<NAME_1>"
    assert masked[0]["organization"] == "<ORG_1_1>"
    assert masked[0]["policy_number"] == "<IDENTIFIER_1_1>"
    assert "Kimberly Gonzales" not in masked[0]["subject"]
    assert "POL-2026-KG-4931" not in masked[0]["subject"]
    assert "<NAME_" in masked[0]["subject"]
    assert "<IDENTIFIER_" in masked[0]["subject"]


def test_sanitize_imported_records_masks_org_and_iso_date_fields_in_billing_style_rows():
    records = [
        {
            "provider_name": "First Horizon Bank",
            "member_id": "MBR.884-20-119",
            "service_date": "2026-09-03",
            "subject": "Billing Inquiry for Member MBR.884-20-119 - Service Date 2026-09-03",
            "email_text": (
                "As the Billing Manager at First Horizon Bank, I am escalating Member ID MBR.884-20-119 "
                "for service date 2026-09-03. Actual Service Rendered differs from the Reported CPT Code."
            ),
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["provider_name"] == "<ORG_1_1>"
    assert masked[0]["member_id"] == "<IDENTIFIER_1_1>"
    assert masked[0]["service_date"] == "<DATE_1_1>"
    assert "First Horizon Bank" not in masked[0]["subject"]
    assert "MBR.884-20-119" not in masked[0]["subject"]
    assert "2026-09-03" not in masked[0]["subject"]
    assert "<ORG_1_1>" in masked[0]["email_text"]
    assert "<IDENTIFIER_1_1>" in masked[0]["email_text"]
    assert "<DATE_1_1>" in masked[0]["email_text"]
    assert "Actual Service" in masked[0]["email_text"]
    assert "Reported CPT Code" in masked[0]["email_text"]
    assert "<PHONE_" not in masked[0]["email_text"]


def test_sanitize_imported_records_reuses_org_placeholder_for_compound_provider_names():
    records = [
        {
            "sender_name": "Melinda Morales",
            "provider_name": "Melinda Morales Healthcare Partners",
            "email_text": "As the Billing Manager at Melinda Morales Healthcare Partners, I need this reviewed.",
        }
    ]

    masked = sanitize_imported_records(records, "Mask likely personal values")

    assert masked[0]["sender_name"] == "<NAME_1>"
    assert masked[0]["provider_name"] == "<ORG_1_1>"
    assert "Melinda Morales Healthcare Partners" not in masked[0]["email_text"]
    assert "<ORG_1_1>" in masked[0]["email_text"]


def test_restore_original_imported_columns_overlays_raw_values_back_on_export_rows():
    session = new_session_state()
    session.fields = [
        {"name": "email_id", "type": "Numeric", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        {"name": "sender_email", "type": "Short Text", "prompt_instruction": "(Imported)", "allow_duplicates": False},
        {"name": "summary", "type": "Long Text", "prompt_instruction": "Summarize the email", "allow_duplicates": True},
    ]
    session.raw_imported_data = [{"email_id": 1, "sender_email": "alice@example.com", "email_text": "Hello"}]

    restored = data_actions.restore_original_imported_columns(
        session,
        [{"email_id": "<EMAIL_1>", "sender_email": "<EMAIL_1>", "summary": "Done"}],
    )

    assert restored[0]["email_id"] == 1
    assert restored[0]["sender_email"] == "alice@example.com"
    assert restored[0]["summary"] == "Done"


def test_restore_original_imported_columns_replaces_unresolved_placeholder_tokens():
    session = new_session_state()
    session.fields = [{"name": "sender_name", "type": "Short Text", "prompt_instruction": "(Imported)", "allow_duplicates": False}]
    session.raw_imported_data = [{"sender_name": "Alice"}]
    session.import_mask_mappings = [{"<NAME_1>": "Alice"}]

    restored = data_actions.restore_original_imported_columns(
        session,
        [{"sender_name": "<NAME_1>", "reply": "Please call back before <TIME_1_1>."}],
    )

    assert restored[0]["sender_name"] == "Alice"
    assert "<TIME_1_1>" not in restored[0]["reply"]
    assert "the listed time" in restored[0]["reply"]


def test_schema_fallback_can_extract_columns_from_raw_json():
    client = object.__new__(LLMClient)
    client.config = SimpleNamespace()
    client.on_log = None
    client.generate_completion = lambda prompt, system_prompt=None: """
    ```json
    {
      "columns": [
        {
          "name": "ticket_id",
          "type": "auto_increment",
          "prompt_instruction": "Unique identifier for each support ticket",
          "constraints": {"allow_duplicates": false}
        },
        {
          "name": "status",
          "type": "categorical",
          "prompt_instruction": "Current workflow state",
          "constraints": {"options": ["Open", "Pending", "Resolved"]}
        }
      ]
    }
    ```
    """

    columns = client._generate_schema_fallback("Support ticket dataset")

    assert len(columns) == 2
    assert columns[0]["type"] == "Auto Increment (ID)"
    assert columns[1]["type"] == "Categorical"
    assert columns[1]["constraints"]["options"] == ["Open", "Pending", "Resolved"]


def test_heuristic_schema_fallback_returns_email_rows():
    client = object.__new__(LLMClient)

    columns = client._generate_heuristic_schema("Insurance inbox emails from clients with at least 7 columns")

    assert len(columns) >= 7
    assert columns[0]["name"] == "message_id"
    assert any(column["name"] == "message_subject" for column in columns)
    assert any(column["name"] == "message_body" for column in columns)
