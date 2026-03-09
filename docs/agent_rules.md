# Agent Rules
_Auto-generated and maintained by SELF_LEARNING_AGENT skill v3.0_
_Project: Synthesizer_
_Last updated: 2026-03-07_
_Sessions: 14 (coding: 9, research: 1, writing: 0, file ops: 4, mixed: 0)_

---

## Rules
<!-- Rules added automatically after each session. Format: see SELF_LEARNING_AGENT SKILL.md -->
- [USER PREF] R001 2026-03-02 RULE: At the start of each task/session, read the project documentation in `docs/` to familiarize with existing project context before making changes.
  REASON: User explicitly requested consistent project familiarization at agent startup.
  CONFIDENCE: high
  TRIGGERS: 4
  EXPIRES: never
- [WORKFLOW] R002 2026-03-02 RULE:
  IF the change adds/updates app features or code behavior:
    THEN launch the app, confirm no runtime/UI errors or crashes, and close the app immediately before continuing.
  ELSE:
    skip app launch for documentation-only updates.
  REASON: User requires runtime verification for feature/code changes and also requires closing run-check windows after successful validation to avoid system slowdown.
  CONFIDENCE: high
  TRIGGERS: 4
  EXPIRES: never
- [CODE] R003 2026-03-02 RULE: In Flet forms/cards, do not place `TextField(expand=True)` in layouts that can grow vertically without explicit constraints; keep fields in controlled rows and avoid multiline unless required.
  REASON: A layout refactor created a large gray stretched input area in the column editor due to uncontrolled field expansion behavior.
  CONFIDENCE: medium
  TRIGGERS: 1
  EXPIRES: 2026-05-31
- [WORKFLOW] R004 2026-03-02 RULE:
  IF multiple tests cover the same behavior:
    THEN keep the most comprehensive/high-signal test and remove redundant duplicates.
  ELSE IF a newly added duplicate/ad-hoc test fails and does not add unique coverage:
    THEN delete that test and any generated result artifacts to avoid folder bloat.
  ELSE:
    keep canonical project tests and fix failures rather than duplicating tests.
  REASON: User prefers a clean, optimal test suite with minimal duplication and no unnecessary failed test artifacts.
  CONFIDENCE: medium
  TRIGGERS: 1
  EXPIRES: 2026-05-31
- [USER PREF] R005 2026-03-07 RULE: Apply the SELF_LEARNING_AGENT workflow on each user task (load/update rules, run risk scan, and maintain session hygiene) unless the user explicitly asks to skip it.
  REASON: User explicitly requested consistent application of the self-learning rule process while working tasks.
  CONFIDENCE: high
  TRIGGERS: 4
  EXPIRES: never
- [CODE] R006 2026-03-07 RULE:
  IF ingesting image files with RAG OCR mode set to `auto` or `on`:
    THEN preflight-check OCR runtime availability (`rapidocr-onnxruntime`) before running ingest.
  ELSE:
    proceed without OCR dependency checks.
  REASON: Image ingest failed when OCR runtime was missing, while spreadsheet ingest succeeded; preflight avoids avoidable runtime failures.
  CONFIDENCE: low
  TRIGGERS: 1
  EXPIRES: 2026-06-05
- [USER PREF] R007 2026-03-07 RULE:
  IF running a live LM Studio generation test:
    THEN run a single session only, wait for completion, and avoid launching duplicate concurrent runs.
  ELSE:
    use standard execution flow.
  REASON: User explicitly requested patient single-session execution to prevent duplicate model sessions and partial outputs.
  CONFIDENCE: medium
  TRIGGERS: 2
  EXPIRES: 2026-06-05
- [USER PREF] R008 2026-03-07 RULE:
  IF tests or generation runs create output artifacts (PDFs, charts, files, reports):
    THEN inspect the produced artifacts/content directly and report verification evidence before claiming success.
  ELSE:
    report standard test pass/fail status.
  REASON: User correction: pass/fail status alone is insufficient; deliverables must be content-verified for professional use.
  CONFIDENCE: medium
  TRIGGERS: 1
  EXPIRES: 2026-06-05
- [USER PREF] R009 2026-03-07 RULE:
  IF document generation uses page or word targets:
    THEN treat them as minimum planning targets rather than hard chunk/document ceilings, and prefer natural section completion over strict upper-bound retries.
  ELSE:
    use standard validation flow.
  REASON: User identified that hard maximum constraints were degrading analysis quality and producing unnatural chunk retry failures.
  CONFIDENCE: high
  TRIGGERS: 1
  EXPIRES: 2026-06-05
- [CODE] R010 2026-03-07 RULE:
  IF LM Studio model responses include reasoning text before a final JSON payload:
    THEN extraction logic must parse and prefer the last valid structured payload instead of assuming the whole response is pure JSON.
  ELSE:
    use normal JSON parsing.
  REASON: Qwen produced long reasoning traces before the final `{\"chunk\": ...}` block; naive parsing let raw reasoning leak into document generation.
  CONFIDENCE: high
  TRIGGERS: 1
  EXPIRES: 2026-06-05

## Pending (Awaiting User Review)
<!-- Conflicts and uncertain rules staged here until user approves -->
