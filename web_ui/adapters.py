from __future__ import annotations

import copy
import importlib.util
import logging
import re
from functools import lru_cache
from html import escape
from typing import Any

import pandas as pd
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
except Exception:  # pragma: no cover - fallback when Presidio is unavailable.
    AnalyzerEngine = None
    Pattern = None
    PatternRecognizer = None
    LemmaContextAwareEnhancer = None
    NlpEngineProvider = None

from core.models import ColumnConstraints, ColumnDefinition, ColumnType


FIELD_TYPE_CHOICES = [column_type.value for column_type in ColumnType]
IMPORT_PRIVACY_CHOICES = [
    "Keep original values",
    "Mask likely personal values",
]
SUMMARY_HEADERS = ["name", "type", "prompt_instruction", "rules"]
GRID_HEADERS = ["row_id", "name", "type", "prompt_instruction", "allow_duplicates"]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d().\-\u2010-\u2015\s]{7,}\d)")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
GENERIC_IDENTIFIER_RE = re.compile(r"\b[A-Z]{2,6}[.-][A-Z0-9]{1,12}(?:[.-][A-Z0-9]{1,16})+\b")
URL_RE = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)
NAME_PAIR_RE = re.compile(
    r"\b((?!(?:Hi|Hello|Dear|Best|Kind|Thanks|Thank|Regards)\b)[A-Z][A-Za-z]+(?:[-'][A-Za-z]+)? [A-Z][A-Za-z]+(?:[-'][A-Za-z]+)?)\b"
)
NON_PERSON_NAME_PAIRS = {
    "Hi Team",
    "Best Regards",
    "Kind Regards",
    "Open Enrollment",
    "Enrollment Delay",
    "Important Update",
    "Health Savings",
    "Health Plan",
    "Dental Plans",
    "Dental Benefits",
    "Vision Benefits",
    "Dear Benefits",
    "Dear Team",
    "Your Health",
    "Claim Management",
    "Contribution Matching",
    "Enrollment Period",
    "Preferred Provider",
    "Upcoming Benefits",
    "Please Review",
    "Client ID",
    "Forwarded Note",
    "HR Notification",
    "Internal Note",
    "Service Date",
    "Actual Service",
    "Date of Service",
    "Claim Reference",
    "Disputed Item",
    "Our Position",
    "Prior Authorization",
    "Potential Denial",
    "CPT Code",
    "Reported CPT",
    "Invoice INV",
    "Member MBR",
}
ORG_PAIR_HINTS = {
    "partners",
    "team",
    "department",
    "group",
    "services",
    "provider",
    "network",
    "healthfirst",
    "benefits",
    "solutions",
    "insurance",
    "medical",
    "wellness",
    "health",
    "bank",
    "clinic",
    "hospital",
    "networks",
    "group",
    "life",
    "cross",
    "shield",
}
ROLE_PAIR_HINTS = {
    "coordinator",
    "manager",
    "director",
    "specialist",
    "administrator",
    "officer",
    "lead",
}
GENERIC_PAIR_WORDS = {
    "annual",
    "urgent",
    "unexpected",
    "inquiry",
    "regarding",
    "unresolved",
    "billing",
    "charges",
    "discrepancies",
    "outstanding",
    "balance",
    "account",
    "question",
    "claims",
    "processing",
    "policy",
    "number",
    "active",
    "pending",
    "review",
    "high",
    "stakes",
    "escalation",
    "assistance",
    "required",
    "administrative",
    "adjustments",
    "recalibration",
    "fees",
    "general",
    "support",
    "claims",
    "processing",
    "contact",
    "status",
    "your",
    "phone",
    "email",
    "premium",
    "discrepancy",
    "unauthorized",
    "charge",
    "client",
    "id",
    "statement",
    "services",
    "business",
    "days",
    "monthly",
    "actual",
    "service",
    "rendered",
    "reported",
    "code",
    "codes",
    "visit",
    "visits",
    "complexity",
    "internal",
    "note",
    "notes",
    "system",
    "auto",
    "reply",
    "log",
    "denial",
    "reason",
    "claims",
    "provider",
    "payer",
    "member",
    "invoice",
    "office",
    "level",
    "processing",
    "explanation",
    "outpatient",
    "surgery",
    "procedure",
    "authorization",
    "authorized",
    "reference",
    "position",
    "item",
    "modifier",
    "mismatch",
}
NAME_CUE_RE = re.compile(
    r"\b(?:employee|new employee|spouse|dependent|child|children|manager|coordinator|contact|regarding|from|to)\b[:,]?\s+([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)? [A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)"
)
GENERIC_DATE_RE = re.compile(
    r"\b(?:\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]{2,9}\s+\d{2,4}|[A-Z][a-z]{2,9}\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{2,4})\b"
)

PRESIDIO_LANGUAGE = "en"
PRESIDIO_ENTITY_MAP = {
    "PERSON": "NAME",
    "EMAIL_ADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "DATE_TIME": "DATE",
    "ORGANIZATION": "ORG",
    "LOCATION": "ADDRESS",
    "URL": "URL",
    "IDENTIFIER": "IDENTIFIER",
    "ROLE": "ROLE",
}
PRESIDIO_PRIORITY = {
    "EMAIL": 0,
    "IDENTIFIER": 1,
    "DATE": 2,
    "PHONE": 3,
    "ADDRESS": 4,
    "ORG": 5,
    "ROLE": 6,
    "NAME": 7,
    "URL": 8,
}
PRESIDIO_CONTEXT_HINTS = {
    "NAME": ["name", "person", "employee", "client", "customer", "member", "sender", "contact", "spouse", "dependent"],
    "EMAIL": ["email", "mail", "sender", "contact", "reply"],
    "PHONE": ["phone", "mobile", "tel", "call", "contact"],
    "DATE": ["date", "dob", "birth", "effective", "deadline", "service", "enrollment"],
    "IDENTIFIER": ["policy", "claim", "member", "client", "account", "invoice", "reference", "identifier", "id", "number"],
    "ORG": ["organization", "company", "provider", "carrier", "payer", "vendor", "partner", "team", "department", "bank", "clinic", "hospital", "employer"],
    "ADDRESS": ["address", "street", "postcode", "postal", "zip", "location"],
    "URL": ["url", "website", "portal", "link"],
}
PRESIDIO_ALLOW_LIST = sorted(NON_PERSON_NAME_PAIRS)


def _preferred_spacy_model_name() -> str:
    for model_name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        if importlib.util.find_spec(model_name) is not None:
            return model_name
    return "en_core_web_sm"


@lru_cache(maxsize=1)
def _privacy_analyzer() -> Any:
    if (
        AnalyzerEngine is None
        or NlpEngineProvider is None
        or PatternRecognizer is None
        or Pattern is None
        or LemmaContextAwareEnhancer is None
    ):
        return None

    try:
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": PRESIDIO_LANGUAGE, "model_name": _preferred_spacy_model_name()}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[PRESIDIO_LANGUAGE],
            context_aware_enhancer=LemmaContextAwareEnhancer(
                context_similarity_factor=0.45,
                min_score_with_context_similarity=0.45,
            ),
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="IDENTIFIER",
                name="DomainIdentifierRecognizer",
                patterns=[
                    Pattern("policy_or_member_id", GENERIC_IDENTIFIER_RE.pattern, 0.95),
                    Pattern("compact_policy_id", r"\b[A-Z]{1,6}\d{4,}[A-Z0-9]{0,12}\b", 0.35),
                ],
                context=PRESIDIO_CONTEXT_HINTS["IDENTIFIER"],
                global_regex_flags=0,
            )
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="DATE_TIME",
                name="FlexibleDateRecognizer",
                patterns=[Pattern("flexible_date", GENERIC_DATE_RE.pattern, 0.7)],
                context=PRESIDIO_CONTEXT_HINTS["DATE"],
                global_regex_flags=0,
            )
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="ROLE",
                name="RoleTitleRecognizer",
                patterns=[
                    Pattern(
                        "role_title",
                        r"\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,2} (?:Coordinator|Manager|Director|Specialist|Administrator|Officer|Lead)\b",
                        0.8,
                    )
                ],
                context=["role", "title", "coordinator", "manager", "director", "specialist", "administrator", "officer", "lead"],
                global_regex_flags=0,
            )
        )
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity="ORGANIZATION",
                name="OrgPhraseRecognizer",
                patterns=[
                    Pattern(
                        "org_phrase",
                        r"\b[A-Z][A-Za-z]+(?:[A-Z][a-z]+| [A-Z][A-Za-z]+){0,4} (?:Team|Department|Group|Services|Partners|Network|Portal|Provider|Insurance|Solutions|Bank|Clinic|Hospital|Inc|LLC|Ltd|Corporation|Company|Employer)\b",
                        0.8,
                    )
                ],
                context=PRESIDIO_CONTEXT_HINTS["ORG"],
                global_regex_flags=0,
            )
        )
        return analyzer
    except Exception:
        return None


def blank_field_record() -> dict[str, Any]:
    return {
        "name": "",
        "type": ColumnType.SHORT_TEXT.value,
        "prompt_instruction": "",
        "options": "",
        "regex_pattern": "",
        "min_value": "",
        "max_value": "",
        "min_length": "",
        "max_length": "",
        "allow_duplicates": False,
        "faker_provider": "",
    }


def normalize_field_type_value(raw_value: Any) -> str:
    if isinstance(raw_value, ColumnType):
        return raw_value.value

    text = str(raw_value or "").strip()
    if not text:
        return ColumnType.SHORT_TEXT.value

    if text.startswith("ColumnType."):
        text = text.split(".", 1)[1].strip()

    for column_type in ColumnType:
        if text in {column_type.value, column_type.name}:
            return column_type.value

    return ColumnType.SHORT_TEXT.value


def normalize_field_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if record and "constraints" in record:
        try:
            return definition_to_field_record(ColumnDefinition(**record))
        except Exception:
            pass
    merged = blank_field_record()
    if record:
        merged.update(record)
    merged["name"] = str(merged.get("name", "") or "").strip()
    merged["type"] = normalize_field_type_value(merged.get("type", ColumnType.SHORT_TEXT.value))
    merged["prompt_instruction"] = str(merged.get("prompt_instruction", "") or "").strip()
    merged["options"] = str(merged.get("options", "") or "").strip()
    merged["regex_pattern"] = str(merged.get("regex_pattern", "") or "").strip()
    merged["faker_provider"] = str(merged.get("faker_provider", "") or "").strip()
    merged["allow_duplicates"] = bool(merged.get("allow_duplicates", False))
    for key in ("min_value", "max_value", "min_length", "max_length"):
        merged[key] = "" if merged.get(key, "") in (None, "") else merged.get(key)
    return merged


def definition_to_field_record(column: ColumnDefinition) -> dict[str, Any]:
    constraints = column.constraints or ColumnConstraints()
    return normalize_field_record(
        {
            "name": column.name,
            "type": column.type.value,
            "prompt_instruction": column.prompt_instruction,
            "options": ",".join(constraints.options or []),
            "regex_pattern": constraints.regex_pattern or "",
            "min_value": "" if constraints.min_value is None else constraints.min_value,
            "max_value": "" if constraints.max_value is None else constraints.max_value,
            "min_length": "" if constraints.min_length is None else constraints.min_length,
            "max_length": "" if constraints.max_length is None else constraints.max_length,
            "allow_duplicates": bool(constraints.allow_duplicates),
            "faker_provider": constraints.faker_provider or "",
        }
    )


def field_record_to_definition(record: dict[str, Any]) -> ColumnDefinition:
    record = normalize_field_record(record)
    try:
        column_type = ColumnType(record["type"])
    except ValueError:
        column_type = ColumnType.SHORT_TEXT

    constraints_kwargs: dict[str, Any] = {
        "options": [item.strip() for item in record["options"].split(",") if item.strip()],
        "regex_pattern": record["regex_pattern"] or None,
        "faker_provider": record["faker_provider"] or None,
        "allow_duplicates": bool(record["allow_duplicates"]),
    }

    for key in ("min_value", "max_value"):
        raw = record.get(key, "")
        if str(raw).strip():
            constraints_kwargs[key] = float(raw)

    for key in ("min_length", "max_length"):
        raw = record.get(key, "")
        if str(raw).strip():
            constraints_kwargs[key] = int(float(raw))

    return ColumnDefinition(
        name=record["name"],
        type=column_type,
        prompt_instruction=record["prompt_instruction"],
        constraints=ColumnConstraints(**constraints_kwargs),
    )


def columns_to_field_records(columns: list[ColumnDefinition]) -> list[dict[str, Any]]:
    return [definition_to_field_record(column) for column in columns]


def field_records_to_columns(records: list[dict[str, Any]]) -> list[ColumnDefinition]:
    columns: list[ColumnDefinition] = []
    for record in records:
        normalized = normalize_field_record(record)
        if not normalized["name"]:
            continue
        columns.append(field_record_to_definition(normalized))
    return columns


def field_choice_labels(records: list[dict[str, Any]]) -> list[str]:
    labels = []
    for index, record in enumerate(records, start=1):
        normalized = normalize_field_record(record)
        name = normalized["name"] or f"Field {index}"
        labels.append(f"{index}. {name}")
    return labels


def field_record_from_choice(records: list[dict[str, Any]], choice: str | None) -> dict[str, Any]:
    if not choice:
        return blank_field_record()
    try:
        index = int(str(choice).split(".", 1)[0]) - 1
    except Exception:
        return blank_field_record()
    if index < 0 or index >= len(records):
        return blank_field_record()
    return normalize_field_record(records[index])


def summary_rules_text(record: dict[str, Any]) -> str:
    normalized = normalize_field_record(record)
    bits: list[str] = []
    if normalized["options"]:
        bits.append("options")
    if normalized["regex_pattern"]:
        bits.append("regex")
    if str(normalized["min_value"]).strip() or str(normalized["max_value"]).strip():
        bits.append("numeric range")
    if str(normalized["min_length"]).strip() or str(normalized["max_length"]).strip():
        bits.append("length")
    if normalized["faker_provider"]:
        bits.append(f"faker={normalized['faker_provider']}")
    if normalized["allow_duplicates"]:
        bits.append("duplicates allowed")
    return ", ".join(bits) if bits else "basic"


def field_summary_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        normalized = normalize_field_record(record)
        rows.append(
            {
                "name": normalized["name"],
                "type": normalized["type"],
                "prompt_instruction": normalized["prompt_instruction"],
                "rules": summary_rules_text(normalized),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_HEADERS) if rows else pd.DataFrame(columns=SUMMARY_HEADERS)


def field_rows_markup(records: list[dict[str, Any]], selected_choice: str | None = None) -> str:
    if not records:
        return (
            "<div style=\"padding:16px;border:1px dashed #cbd5e1;border-radius:14px;background:#f8fafc;color:#475569;\">"
            "<strong>No rows yet.</strong><br>"
            "Use <em>Generate Fields</em> or <em>Add Row</em> to build your schema."
            "</div>"
        )

    selected_index = None
    if selected_choice:
        try:
            selected_index = int(str(selected_choice).split(".", 1)[0]) - 1
        except Exception:
            selected_index = None

    imported_rows: list[str] = []
    generated_rows: list[str] = []

    for index, record in enumerate(records, start=1):
        normalized = normalize_field_record(record)
        prompt = escape(normalized["prompt_instruction"] or "No instruction yet.")
        rules = escape(summary_rules_text(normalized))
        is_imported = normalized["prompt_instruction"] == "(Imported)"
        kind_label = "Imported column" if is_imported else "New/generated column"
        kind_bg = "#dbeafe" if is_imported else "#dcfce7"
        kind_text = "#1d4ed8" if is_imported else "#166534"
        card_bg = "#eff6ff" if is_imported else "#f0fdf4"
        border = "#2563eb" if is_imported else "#16a34a"
        if selected_index == index - 1:
            card_bg = "#fff7ed"
            border = "#ea580c"

        row_markup = "".join(
            [
                (
                    f"<div style=\"padding:14px;border:1px solid {border};border-radius:18px;background:{card_bg};\">"
                ),
                (
                    "<div style=\"display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;\">"
                    f"<span style=\"display:inline-flex;align-items:center;justify-content:center;min-width:74px;padding:6px 10px;"
                    "border-radius:999px;background:#111827;color:#ffffff;font-size:0.88rem;font-weight:700;\">"
                    f"Row {index}</span>"
                    f"<span style=\"display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;background:{kind_bg};"
                    f"color:{kind_text};font-size:0.88rem;font-weight:700;\">{kind_label}</span>"
                    f"<span style=\"color:#475569;font-size:0.92rem;\">{escape(normalized['type'])}</span>"
                    "</div>"
                ),
                (
                    "<div style=\"font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:8px;\">"
                    f"{escape(normalized['name'] or f'field_{index}')}"
                    "</div>"
                ),
                (
                    "<div style=\"color:#0f172a;line-height:1.45;\">"
                    f"{prompt}"
                    "</div>"
                ),
                (
                    "<div style=\"margin-top:8px;color:#64748b;font-size:0.92rem;\">"
                    f"Rules: {rules}"
                    "</div>"
                ),
                "</div>",
            ]
        )
        if is_imported:
            imported_rows.append(row_markup)
        else:
            generated_rows.append(row_markup)

    def _section(title: str, subtitle: str, rows_markup: list[str], tone: str) -> str:
        if not rows_markup:
            return ""
        tone_bg = "#eff6ff" if tone == "imported" else "#f0fdf4"
        tone_border = "#bfdbfe" if tone == "imported" else "#bbf7d0"
        return (
            f"<div style=\"padding:14px;border:1px solid {tone_border};border-radius:18px;background:{tone_bg};margin-bottom:14px;\">"
            f"<div style=\"display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;\">"
            f"<div><div style=\"font-weight:800;color:#0f172a;\">{escape(title)}</div>"
            f"<div style=\"color:#475569;font-size:0.94rem;\">{escape(subtitle)}</div></div>"
            f"<div style=\"font-weight:800;color:#0f172a;\">{len(rows_markup)}</div>"
            f"</div>"
            f"<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:14px;\">"
            + "".join(rows_markup)
            + "</div></div>"
        )

    return (
        "<div style=\"border:1px solid #d8dee8;border-radius:20px;background:#ffffff;padding:14px;\">"
        "<div style=\"display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px;\">"
        "<div><div style=\"font-weight:800;color:#0f172a;\">Schema overview</div>"
        "<div style=\"color:#64748b;font-size:0.94rem;\">Imported columns stay separate from the new fields you add for generation.</div></div>"
        f"<div style=\"font-weight:800;color:#0f172a;\">{len(records)} total</div>"
        "</div>"
        + _section("Imported columns", "Columns detected from the file you uploaded.", imported_rows, "imported")
        + _section("New fields to generate", "Columns the model will synthesize or enrich.", generated_rows, "generated")
        + "</div>"
    )


def field_records_to_grid_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        normalized = normalize_field_record(record)
        rows.append(
            {
                "row_id": f"Row {index}",
                "name": normalized["name"],
                "type": normalized["type"],
                "prompt_instruction": normalized["prompt_instruction"],
                "allow_duplicates": bool(normalized["allow_duplicates"]),
            }
        )
    if not rows:
        rows.append(
            {
                "row_id": "Row 1",
                "name": "",
                "type": ColumnType.SHORT_TEXT.value,
                "prompt_instruction": "",
                "allow_duplicates": False,
            }
        )
    return pd.DataFrame(rows, columns=GRID_HEADERS)


def import_preview_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).head(12)


def _sensitive_column_label(column_name: str) -> str | None:
    lowered = (column_name or "").strip().lower()
    if not lowered:
        return None

    if _is_free_text_column(column_name):
        return None

    tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
    org_tokens = {"organization", "org", "provider", "company", "carrier", "payer", "vendor", "partner", "partners", "bank", "clinic", "hospital", "insurer", "employer"}

    if any(token in tokens for token in ("member", "policy", "claim", "account", "employee", "customer", "client")) and any(
        token in tokens for token in ("id", "number", "no")
    ):
        return "IDENTIFIER"

    if "date" in tokens or {"dob", "birth"} & tokens:
        return "DATE"

    if ("name" in tokens and tokens & org_tokens) or ("organization" in tokens) or ("provider" in tokens and "name" in tokens):
        return "ORG"

    if "email" in tokens and "id" not in tokens:
        return "EMAIL"
    if {"phone", "mobile", "tel"} & tokens:
        return "PHONE"
    if "name" in tokens:
        return "NAME"
    if {"address", "street", "postcode", "postal", "zip"} & tokens:
        return "ADDRESS"
    if {"ssn", "passport", "license"} & tokens:
        return "IDENTIFIER"

    if any(token in lowered for token in ("social_security", "national_insurance", "tax_id")):
        return "IDENTIFIER"
    return None


def _mask_inline_text(text: str) -> str:
    masked = EMAIL_RE.sub("<EMAIL>", text)
    masked = PHONE_RE.sub("<PHONE>", masked)
    masked = SSN_RE.sub("<IDENTIFIER>", masked)
    masked = URL_RE.sub("<URL>", masked)
    return masked


def _is_free_text_column(column_name: str) -> bool:
    lowered = (column_name or "").strip().lower()
    return any(token in lowered for token in ("text", "body", "message", "content", "note", "description", "comment"))


def _mask_name_pairs(text: str) -> str:
    non_person_terms = {
        "team",
        "plan",
        "period",
        "benefits",
        "health",
        "enrollment",
        "coverage",
        "provider",
        "network",
        "policy",
        "summary",
        "deadline",
        "portal",
        "services",
    }

    def replace(match: re.Match[str]) -> str:
        phrase = match.group(1)
        if phrase in NON_PERSON_NAME_PAIRS:
            return phrase
        first, second = phrase.split(" ", 1)
        if first.lower() in non_person_terms or second.lower() in non_person_terms:
            return phrase
        return "<NAME>"

    return NAME_PAIR_RE.sub(replace, text)


def _replace_literal_token(text: str, token: str, replacement: str) -> str:
    if not token:
        return text
    pattern = re.escape(token)
    if re.search(r"\s", token):
        return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(rf"\b{pattern}\b", replacement, text, flags=re.IGNORECASE)


def _classify_name_pair(phrase: str) -> str | None:
    if not phrase or phrase in NON_PERSON_NAME_PAIRS:
        return None

    lowered_words = {part.lower() for part in phrase.split()}
    if lowered_words & ROLE_PAIR_HINTS:
        return "ROLE"
    if lowered_words & ORG_PAIR_HINTS:
        return "ORG"
    if lowered_words & GENERIC_PAIR_WORDS:
        return None
    return "NAME"


def _presidio_context(column_name: str) -> list[str]:
    lowered = (column_name or "").strip().lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    label = _sensitive_column_label(column_name)
    if label and label in PRESIDIO_CONTEXT_HINTS:
        tokens.extend(PRESIDIO_CONTEXT_HINTS[label])
    if {"first", "last", "given", "family"} & set(tokens):
        tokens.extend(PRESIDIO_CONTEXT_HINTS["NAME"])
    if {"dob", "birth"} & set(tokens):
        tokens.extend(PRESIDIO_CONTEXT_HINTS["DATE"])
    if {"member", "policy", "claim", "invoice", "account", "reference"} & set(tokens):
        tokens.extend(PRESIDIO_CONTEXT_HINTS["IDENTIFIER"])
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            deduped.append(token)
            seen.add(token)
    return deduped


def _entity_category(entity_type: str, raw_value: str) -> str | None:
    normalized = str(raw_value or "").strip()
    if not normalized or normalized in NON_PERSON_NAME_PAIRS:
        return None

    mapped = PRESIDIO_ENTITY_MAP.get(entity_type)
    if mapped is None:
        return None

    if mapped == "NAME":
        if normalized.lower() == normalized:
            return None
        if " " not in normalized:
            return None
        if GENERIC_IDENTIFIER_RE.fullmatch(normalized) or re.search(r"\d", normalized):
            return "IDENTIFIER"
        lowered_words = {part.lower() for part in normalized.split()}
        classified = _classify_name_pair(normalized)
        if classified == "ROLE":
            return None
        if classified is not None:
            return classified
        if lowered_words & GENERIC_PAIR_WORDS:
            return None
        if any(part.isupper() and len(part) <= 4 for part in normalized.split()):
            return None

    if mapped == "ORG":
        if normalized in NON_PERSON_NAME_PAIRS:
            return None
        if normalized.isupper() and len(normalized) <= 5:
            return None
        if len(normalized.split()) < 2:
            return None
        if normalized.lower() == normalized:
            return None
        classified = _classify_name_pair(normalized)
        if classified == "ORG":
            return classified
        if re.search(r"\b(?:Team|Department|Group|Services|Partners|Network|Portal|Provider)\b$", normalized):
            return "ORG"
        return None

    if mapped == "ROLE":
        return None

    if mapped == "DATE":
        if not re.search(r"\d", normalized):
            return None

    if mapped == "ADDRESS" and len(normalized) < 6:
        return None
    return mapped


def _select_sensitive_spans(text: str, column_name: str) -> list[tuple[int, int, str, str, float]]:
    candidates: list[tuple[int, int, str, str, float]] = []
    analyzer = _privacy_analyzer()

    if analyzer is not None:
        try:
            results = analyzer.analyze(
                text=text,
                language=PRESIDIO_LANGUAGE,
                entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME", "ORGANIZATION", "URL", "IDENTIFIER", "ROLE"],
                context=_presidio_context(column_name),
                allow_list=PRESIDIO_ALLOW_LIST,
                score_threshold=0.35,
            )
        except Exception:
            results = []

        for result in results:
            start = result.start
            end = result.end
            raw_value = text[start:end].strip()
            if raw_value.startswith("Dear "):
                start += 5
                raw_value = raw_value[5:].strip()
            category = _entity_category(result.entity_type, raw_value)
            if category is None:
                continue
            if "\n" in raw_value and category in {"ROLE", "ORG"}:
                parts = [part.strip(" ,") for part in re.split(r"[\r\n]+", raw_value) if part.strip()]
                matched_part = False
                for part in reversed(parts):
                    part_category = _entity_category(result.entity_type, part)
                    if part_category == category:
                        start += raw_value.rfind(part)
                        raw_value = part
                        matched_part = True
                        break
                if not matched_part:
                    continue
            if category == "IDENTIFIER":
                identifier_match = GENERIC_IDENTIFIER_RE.search(raw_value)
                if identifier_match:
                    start = start + identifier_match.start()
                    end = start + len(identifier_match.group(0))
                    raw_value = identifier_match.group(0)
            elif category == "DATE":
                date_match = GENERIC_DATE_RE.search(raw_value)
                if not date_match:
                    continue
                start = start + date_match.start()
                end = start + len(date_match.group(0))
                raw_value = date_match.group(0)
            candidates.append((start, end, category, raw_value, float(result.score)))

    for regex, category in (
        (EMAIL_RE, "EMAIL"),
        (PHONE_RE, "PHONE"),
        (GENERIC_IDENTIFIER_RE, "IDENTIFIER"),
        (GENERIC_DATE_RE, "DATE"),
        (SSN_RE, "IDENTIFIER"),
        (URL_RE, "URL"),
    ):
        for match in regex.finditer(text):
            raw_value = match.group(0).strip()
            if raw_value:
                candidates.append((match.start(), match.end(), category, raw_value, 0.99))

    for match in NAME_CUE_RE.finditer(text):
        raw_value = match.group(1).strip()
        if raw_value:
            candidates.append((match.start(1), match.end(1), "NAME", raw_value, 0.7))

    for match in NAME_PAIR_RE.finditer(text):
        raw_value = match.group(1).strip()
        category = _classify_name_pair(raw_value)
        if category is not None and category != "ROLE":
            candidates.append((match.start(1), match.end(1), category, raw_value, 0.65))

    selected: list[tuple[int, int, str, str, float]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (PRESIDIO_PRIORITY.get(item[2], 99), -(item[1] - item[0]), -item[4], item[0]),
    ):
        start, end, _, raw_value, _ = candidate
        if not raw_value:
            continue
        if any(not (end <= existing_start or start >= existing_end) for existing_start, existing_end, *_ in selected):
            continue
        selected.append(candidate)

    return sorted(selected, key=lambda item: item[0], reverse=True)


def _mask_free_text_with_presidio(
    text: str,
    column_name: str,
    inline_replacements: list[tuple[str, str]],
    assign_placeholder,
    raw_to_placeholder: dict[str, str],
) -> str:
    masked_text = text
    for raw_value, replacement in sorted(inline_replacements, key=lambda item: len(item[0]), reverse=True):
        masked_text = _replace_literal_token(masked_text, raw_value, replacement)
    for start, end, category, raw_value, _ in _select_sensitive_spans(masked_text, column_name):
        if "<" in raw_value or ">" in raw_value:
            continue
        placeholder = assign_placeholder(category, raw_value, preferred=raw_to_placeholder.get(raw_value))
        masked_text = f"{masked_text[:start]}{placeholder}{masked_text[end:]}"
    return masked_text


def _mask_free_text_value(text: str, replacements: list[tuple[str, str]]) -> str:
    masked_text = _mask_inline_text(text)
    for raw_value, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        masked_text = _replace_literal_token(masked_text, raw_value, replacement)
    return _mask_name_pairs(masked_text)


def mask_imported_records(records: list[dict[str, Any]], privacy_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if privacy_mode != "Mask likely personal values":
        copied = copy.deepcopy(records)
        return copied, [{} for _ in copied]

    sanitized: list[dict[str, Any]] = []
    mappings: list[dict[str, str]] = []
    for row_index, row in enumerate(records, start=1):
        cleaned_row: dict[str, Any] = {}
        placeholder_to_raw: dict[str, str] = {}
        raw_to_placeholder: dict[str, str] = {}
        inline_replacements: list[tuple[str, str]] = []
        category_counts: dict[str, int] = {
            "NAME": 0,
            "ORG": 0,
            "ROLE": 0,
            "PHONE": 0,
            "EMAIL": 0,
            "IDENTIFIER": 0,
            "DATE": 0,
            "ADDRESS": 0,
            "URL": 0,
        }

        def assign_placeholder(category: str, raw_value: str, *, preferred: str | None = None) -> str:
            normalized = str(raw_value or "").strip()
            if not normalized:
                return normalized
            if normalized in raw_to_placeholder:
                return raw_to_placeholder[normalized]
            if preferred:
                placeholder = preferred
            else:
                category_counts[category] = category_counts.get(category, 0) + 1
                placeholder = f"<{category}_{row_index}_{category_counts[category]}>"
            raw_to_placeholder[normalized] = placeholder
            placeholder_to_raw[placeholder] = normalized
            return placeholder

        # Seed structured field replacements first so the same placeholders are reused in free text.
        for key, value in row.items():
            if not isinstance(value, str):
                continue
            if _is_free_text_column(str(key)):
                continue
            label = _sensitive_column_label(str(key))
            if label == "NAME":
                placeholder = assign_placeholder("NAME", value, preferred=f"<NAME_{row_index}>")
                inline_replacements.append((value, placeholder))
                for part in [piece.strip() for piece in re.split(r"\s+", value) if len(piece.strip()) >= 3]:
                    inline_replacements.append((part, placeholder))
            elif label == "EMAIL":
                placeholder = assign_placeholder("EMAIL", value, preferred=f"<EMAIL_{row_index}>")
                inline_replacements.append((value, placeholder))
            elif label in {"PHONE", "ADDRESS", "DATE", "IDENTIFIER", "ORG"}:
                placeholder = assign_placeholder(label, value)
                inline_replacements.append((value, placeholder))

        for key, value in row.items():
            label = _sensitive_column_label(str(key))
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cleaned_row[key] = value
                continue

            if label is not None and not _is_free_text_column(str(key)):
                if isinstance(value, str):
                    if label == "NAME":
                        cleaned_row[key] = assign_placeholder("NAME", value, preferred=f"<NAME_{row_index}>")
                    elif label == "EMAIL":
                        cleaned_row[key] = assign_placeholder("EMAIL", value, preferred=f"<EMAIL_{row_index}>")
                    else:
                        cleaned_row[key] = assign_placeholder(label, value)
                else:
                    cleaned_row[key] = value
                continue

            if isinstance(value, str):
                masked_text = _mask_free_text_with_presidio(
                    value,
                    str(key),
                    inline_replacements,
                    assign_placeholder,
                    raw_to_placeholder,
                )
                cleaned_row[key] = masked_text
            else:
                cleaned_row[key] = value

        sanitized.append(cleaned_row)
        mappings.append(placeholder_to_raw)

    return sanitized, mappings


def sanitize_imported_records(records: list[dict[str, Any]], privacy_mode: str) -> list[dict[str, Any]]:
    sanitized, _ = mask_imported_records(records, privacy_mode)
    return sanitized


def detect_privacy_leaks(raw_records: list[dict[str, Any]], masked_records: list[dict[str, Any]]) -> list[str]:
    leaks: list[str] = []
    for row_index, (raw_row, masked_row) in enumerate(zip(raw_records, masked_records), start=1):
        masked_text = " ".join(str(value) for value in masked_row.values() if value is not None)
        masked_text_lower = masked_text.lower()
        candidate_tokens: set[str] = set()

        for key, value in raw_row.items():
            if not isinstance(value, str):
                continue
            label = _sensitive_column_label(str(key))
            if label in {"NAME", "EMAIL", "PHONE", "ADDRESS", "IDENTIFIER"}:
                candidate_tokens.add(value.strip())
                if label == "NAME":
                    candidate_tokens.update(part.strip() for part in re.split(r"\s+", value) if len(part.strip()) >= 3)
            if label == "DATE":
                candidate_tokens.add(value.strip())
            candidate_tokens.update(EMAIL_RE.findall(value))
            candidate_tokens.update(PHONE_RE.findall(value))
            candidate_tokens.update(GENERIC_IDENTIFIER_RE.findall(value))
            candidate_tokens.update(GENERIC_DATE_RE.findall(value))
            if _is_free_text_column(str(key)):
                candidate_tokens.update(
                    token
                    for token in NAME_PAIR_RE.findall(value)
                    if _classify_name_pair(token) not in {None, "ROLE"}
                )
                for _, _, _, token, _ in _select_sensitive_spans(value, str(key)):
                    candidate_tokens.add(token)

        for token in sorted(candidate_tokens):
            normalized = token.strip()
            if len(normalized) < 3:
                continue
            if normalized.lower() in masked_text_lower:
                leaks.append(f"Row {row_index}: {normalized}")
    return leaks


def imported_columns_markup(records: list[dict[str, Any]], privacy_mode: str) -> str:
    if not records:
        return "No imported columns yet."

    columns = [str(name) for name in records[0].keys()]
    chips = "".join(
        f"<span style=\"display:inline-block;margin:0 8px 8px 0;padding:7px 12px;border:1px solid #93c5fd;border-radius:999px;background:#eff6ff;color:#1e3a8a;font-size:0.94rem;font-weight:700;\">{escape(name)}</span>"
        for name in columns
    )
    mode_note = "Masked preview + AI context" if privacy_mode == "Mask likely personal values" else "Original import values"
    return (
        "<div style=\"padding:14px;border:1px solid #bfdbfe;border-radius:18px;background:#f8fbff;\">"
        f"<div style=\"display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;\">"
        f"<div><div style=\"font-weight:800;color:#0f172a;\">Imported columns ({len(columns)})</div>"
        f"<div style=\"color:#475569;font-size:0.94rem;\">These are the source columns already present in your file.</div></div>"
        f"<div style=\"display:inline-flex;align-items:center;padding:6px 10px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:0.88rem;font-weight:700;\">{escape(mode_note)}</div>"
        "</div>"
        f"<div style=\"margin-top:12px;\">{chips}</div>"
        "</div>"
    )


def infer_field_records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    inferred: list[dict[str, Any]] = []
    for column_name in df.columns:
        series = df[column_name]
        if pd.api.types.is_integer_dtype(series):
            column_type = ColumnType.NUMERIC.value
        elif pd.api.types.is_float_dtype(series):
            column_type = ColumnType.NUMERIC.value
        elif pd.api.types.is_bool_dtype(series):
            column_type = ColumnType.BOOLEAN.value
        else:
            column_type = ColumnType.SHORT_TEXT.value
        inferred.append(
            normalize_field_record(
                {
                    "name": str(column_name),
                    "type": column_type,
                    "prompt_instruction": "(Imported)",
                }
            )
        )
    return inferred


def build_schema_context(records: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    sample_row = records[0]
    context_lines = []
    for header, value in sample_row.items():
        type_hint = "String"
        if isinstance(value, bool):
            type_hint = "Boolean"
        elif isinstance(value, int):
            type_hint = "Integer"
        elif isinstance(value, float):
            type_hint = "Float"
        context_lines.append(f"Column: {header} ({type_hint}) | Sample: {value}")
    return "\n".join(context_lines)


def visibility_for_field_type(field_type: str) -> dict[str, bool]:
    selected = (field_type or ColumnType.SHORT_TEXT.value).strip()
    return {
        "show_options": selected == ColumnType.CATEGORICAL.value,
        "show_numeric": selected == ColumnType.NUMERIC.value,
        "show_text": selected in {ColumnType.SHORT_TEXT.value, ColumnType.LONG_TEXT.value},
        "show_faker": selected == ColumnType.DETERMINISTIC.value,
    }
