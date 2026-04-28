from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.controller import GeneratorController
from core.models import AIProvider, ColumnConstraints, ColumnDefinition, ColumnType, GeneratorConfig
from web_ui.actions import data_actions
from web_ui.adapters import (
    EMAIL_RE,
    GENERIC_DATE_RE,
    GENERIC_IDENTIFIER_RE,
    NAME_PAIR_RE,
    PHONE_RE,
    _classify_name_pair,
    _is_free_text_column,
    _select_sensitive_spans,
    _sensitive_column_label,
    field_records_to_columns,
    infer_field_records_from_dataframe,
    mask_imported_records,
)
from web_ui.state import new_session_state


OUTPUT_DIR = Path(".web_ui_exports/live_privacy_eval")
MODEL_ID = "qwen/qwen3.5-9b"
PLACEHOLDER_RE = re.compile(r"<[A-Z]+(?:_[0-9A-Z]+)+>")


def _claims_columns() -> list[ColumnDefinition]:
    return [
        ColumnDefinition(name="email_id", type=ColumnType.AUTO_INCREMENT, prompt_instruction="Unique email row id"),
        ColumnDefinition(
            name="sender_name",
            type=ColumnType.DETERMINISTIC,
            prompt_instruction="Deterministic realistic sender full name",
            constraints=ColumnConstraints(faker_provider="name"),
        ),
        ColumnDefinition(
            name="sender_email",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A valid and unique email address for @[sender_name] that reflects reference @[email_id].",
            constraints=ColumnConstraints(regex_pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"),
        ),
        ColumnDefinition(
            name="sender_role",
            type=ColumnType.CATEGORICAL,
            prompt_instruction="Professional role of @[sender_name].",
            constraints=ColumnConstraints(
                options=[
                    "Policyholder",
                    "Spouse",
                    "Benefits Coordinator",
                    "HR Manager",
                    "Claims Analyst",
                    "Payroll Specialist",
                ],
                allow_duplicates=True,
            ),
        ),
        ColumnDefinition(
            name="organization",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A realistic insurer, employer, broker, clinic, or vendor name involved in the message.",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="policy_number",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A realistic unique policy, member, or claim identifier that distinctly incorporates reference @[email_id] in a varied business format such as POL-2026-AB-4931, MBR.204-77-991, or CLM/7781/AZ.",
        ),
        ColumnDefinition(
            name="callback_phone",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A callback phone number in a varied format such as +1 212 555 0188, (415) 555-0199, 020 7946 0958, or 555.210.7788.",
        ),
        ColumnDefinition(
            name="date_of_birth",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A date of birth in a varied format such as 12/04/1985, 12.04.85, 4 April 1985, Apr 4, 1985, or 1985-04-12.",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="subject",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A realistic and distinct email subject line for @[policy_number] and @[organization].",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="email_text",
            type=ColumnType.LONG_TEXT,
            prompt_instruction=(
                "Write a realistic business email from @[sender_name], who is a @[sender_role]. "
                "Mention @[policy_number], @[callback_phone], @[date_of_birth], and @[organization] naturally in the body. "
                "Use one clearly distinct scenario such as enrollment delay, coordination-of-benefits question, dependent update, payroll deduction mismatch, or provider-network issue. "
                "Use varied styles such as signatures, forwarded notes, bullets, or partial quote chains. Do not use placeholders."
            ),
            constraints=ColumnConstraints(min_length=180, max_length=1600, allow_duplicates=True),
        ),
    ]


def _billing_columns() -> list[ColumnDefinition]:
    return [
        ColumnDefinition(name="email_id", type=ColumnType.AUTO_INCREMENT, prompt_instruction="Unique email row id"),
        ColumnDefinition(
            name="sender_name",
            type=ColumnType.DETERMINISTIC,
            prompt_instruction="Deterministic realistic sender full name",
            constraints=ColumnConstraints(faker_provider="name"),
        ),
        ColumnDefinition(
            name="sender_email",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A valid and unique email address for @[sender_name] that reflects reference @[email_id].",
            constraints=ColumnConstraints(regex_pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$"),
        ),
        ColumnDefinition(
            name="sender_role",
            type=ColumnType.CATEGORICAL,
            prompt_instruction="Professional role of @[sender_name].",
            constraints=ColumnConstraints(
                options=[
                    "Billing Manager",
                    "Customer Service Lead",
                    "Appeals Specialist",
                    "Provider Relations Analyst",
                    "Clinic Administrator",
                    "Operations Coordinator",
                ],
                allow_duplicates=True,
            ),
        ),
        ColumnDefinition(
            name="provider_name",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A realistic insurer, hospital group, clinic network, or third-party administrator name.",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="member_id",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A realistic unique member, invoice, or case identifier that distinctly incorporates reference @[email_id] in varied business formats such as INV-2026-4431, CASE/7712-QA, MBR.884-20-119, or REF-88-AZ-204.",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="service_date",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A service date in varied formats such as 03/09/2026, 03.09.26, 9 September 2026, Sep 9, 2026, or 2026-09-03.",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="secondary_phone",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A secondary contact phone number in a varied format such as +1 646 555 0132, (312) 555-0147, 555.884.2100, or 0161 496 0991.",
        ),
        ColumnDefinition(
            name="subject",
            type=ColumnType.SHORT_TEXT,
            prompt_instruction="A realistic and distinct escalation or billing subject line for @[member_id] and @[provider_name].",
            constraints=ColumnConstraints(allow_duplicates=True),
        ),
        ColumnDefinition(
            name="email_text",
            type=ColumnType.LONG_TEXT,
            prompt_instruction=(
                "Write a realistic escalation, billing, or provider follow-up email from @[sender_name], who is a @[sender_role]. "
                "Mention @[member_id], @[service_date], @[secondary_phone], and @[provider_name] naturally in the body. "
                "Choose one clearly distinct scenario such as denial follow-up, overpayment refund request, coding mismatch, prior-authorization dispute, EOB clarification, or duplicate-charge escalation. "
                "Use mixed formatting such as bullet lists, quote chains, internal note snippets, or sign-off blocks. Do not use placeholders."
            ),
            constraints=ColumnConstraints(min_length=180, max_length=1600, allow_duplicates=True),
        ),
    ]


def _new_generated_fields() -> list[dict[str, Any]]:
    return [
        {
            "name": "Summary",
            "type": ColumnType.LONG_TEXT.value,
            "prompt_instruction": "Provide a concise summary of @[email_text].",
            "allow_duplicates": True,
        },
        {
            "name": "reply",
            "type": ColumnType.LONG_TEXT.value,
            "prompt_instruction": "Write a professional reply to the sender of @[email_text].",
            "allow_duplicates": True,
        },
    ]


def _candidate_sensitive_tokens(raw_rows: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for raw_row in raw_rows:
        for key, value in raw_row.items():
            if not isinstance(value, str):
                continue
            label = _sensitive_column_label(str(key))
            if label in {"NAME", "EMAIL", "PHONE", "ADDRESS", "IDENTIFIER", "DATE"}:
                tokens.add(value.strip())
                if label == "NAME":
                    tokens.update(part.strip() for part in re.split(r"\s+", value) if len(part.strip()) >= 3)
            tokens.update(EMAIL_RE.findall(value))
            tokens.update(PHONE_RE.findall(value))
            tokens.update(GENERIC_IDENTIFIER_RE.findall(value))
            tokens.update(GENERIC_DATE_RE.findall(value))
            if _is_free_text_column(str(key)):
                tokens.update(
                    token
                    for token in NAME_PAIR_RE.findall(value)
                    if _classify_name_pair(token) not in {None, "ROLE"}
                )
                for _, _, _, token, _ in _select_sensitive_spans(value, str(key)):
                    tokens.add(token)
    return {token.strip() for token in tokens if isinstance(token, str) and len(token.strip()) >= 3}


def _generate_raw_dataset(name: str, columns: list[ColumnDefinition], rows: int) -> Path:
    controller = GeneratorController()
    config = GeneratorConfig(
        model_id=MODEL_ID,
        provider=AIProvider.LM_STUDIO,
        num_rows=rows,
        similarity_threshold=0.55,
        max_retries=30,
    )
    controller.initialize(config, columns)
    controller._run_generation_loop()
    if len(controller.generated_rows) != rows:
        raise RuntimeError(f"{name}: expected {rows} rows, got {len(controller.generated_rows)}")
    path = OUTPUT_DIR / f"{name}_raw.csv"
    controller.export_csv(str(path))
    return path


def _run_masked_enrichment(raw_csv: Path) -> dict[str, Any]:
    raw_df = pd.read_csv(raw_csv)
    raw_rows = raw_df.to_dict(orient="records")
    masked_rows, mappings = mask_imported_records(raw_rows, "Mask likely personal values")

    fields = infer_field_records_from_dataframe(raw_df)
    fields.extend(_new_generated_fields())
    columns = field_records_to_columns(fields)

    controller = GeneratorController()
    config = GeneratorConfig(
        model_id=MODEL_ID,
        provider=AIProvider.LM_STUDIO,
        num_rows=len(raw_rows),
        similarity_threshold=0.82,
        max_retries=20,
        existing_data=masked_rows,
    )
    controller.initialize(config, columns)

    original_generate = controller.llm_client.generate_completion
    prompts: list[str] = []

    def capture(prompt: str, system_prompt: str = "") -> str:
        prompts.append(prompt)
        return original_generate(prompt, system_prompt)

    controller.llm_client.generate_completion = capture
    controller._run_generation_loop()

    if len(controller.generated_rows) != len(raw_rows):
        raise RuntimeError(f"{raw_csv.name}: enrichment expected {len(raw_rows)} rows, got {len(controller.generated_rows)}")

    joined_generation_prompts = "\n".join(prompt for prompt in prompts if prompt.startswith("Generate a single"))
    leaked_tokens = sorted(
        token for token in _candidate_sensitive_tokens(raw_rows) if token.lower() in joined_generation_prompts.lower()
    )

    session = new_session_state()
    session.import_privacy_mode = "Mask likely personal values"
    session.raw_imported_data = raw_rows
    session.imported_data = masked_rows
    session.import_mask_mappings = mappings
    session.fields = fields
    session.generated_rows = [row.data for row in controller.generated_rows]

    session.generated_rows = data_actions.restore_original_imported_columns(session, session.generated_rows)

    original_export_dir = data_actions.EXPORT_DIR
    data_actions.EXPORT_DIR = OUTPUT_DIR
    try:
        session, export_path, export_status, _ = data_actions.export_generated_data(session, "csv")
    finally:
        data_actions.EXPORT_DIR = original_export_dir

    if not export_path:
        raise RuntimeError(f"{raw_csv.name}: export failed")

    exported_df = pd.read_csv(export_path)
    placeholder_rows: list[int] = []
    for index, row in exported_df.iterrows():
        if any(PLACEHOLDER_RE.search(str(value)) for value in row.tolist() if isinstance(value, str)):
            placeholder_rows.append(index + 1)

    return {
        "raw_csv": str(raw_csv),
        "masked_preview_csv": str(_write_masked_preview(raw_csv.stem, masked_rows)),
        "generated_count": len(controller.generated_rows),
        "prompt_count": len([prompt for prompt in prompts if prompt.startswith("Generate a single")]),
        "leaked_prompt_tokens": leaked_tokens,
        "export_csv": str(export_path),
        "export_status": export_status,
        "export_placeholder_rows": placeholder_rows,
        "sample_generated_row": session.generated_rows[0],
    }


def _write_masked_preview(stem: str, masked_rows: list[dict[str, Any]]) -> Path:
    path = OUTPUT_DIR / f"{stem}_masked_preview.csv"
    pd.DataFrame(masked_rows).to_csv(path, index=False)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("claims_intake_emails", _claims_columns(), 3),
        ("billing_escalation_emails", _billing_columns(), 3),
    ]

    results: dict[str, Any] = {}
    for name, columns, rows in datasets:
        raw_csv = _generate_raw_dataset(name, columns, rows)
        results[name] = _run_masked_enrichment(raw_csv)

    report_path = OUTPUT_DIR / "live_privacy_dataset_regression.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
