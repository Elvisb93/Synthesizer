# Agent Rules
_Auto-generated and maintained by SELF_LEARNING_AGENT skill v3.0_
_Project: Synthesizer_
_Last updated: 2026-03-02_
_Sessions: 4 (coding: 2, research: 0, writing: 0, file ops: 2, mixed: 0)_

---

## Rules
<!-- Rules added automatically after each session. Format: see SELF_LEARNING_AGENT SKILL.md -->
- [USER PREF] R001 2026-03-02 RULE: At the start of each task/session, read the project documentation in `docs/` to familiarize with existing project context before making changes.
  REASON: User explicitly requested consistent project familiarization at agent startup.
  CONFIDENCE: medium
  TRIGGERS: 1
  EXPIRES: 2026-05-31
- [WORKFLOW] R002 2026-03-02 RULE:
  IF the change adds/updates app features or code behavior:
    THEN launch the app, confirm no runtime/UI errors or crashes, and close the app immediately before continuing.
  ELSE:
    skip app launch for documentation-only updates.
  REASON: User requires runtime verification for feature/code changes and also requires closing run-check windows after successful validation to avoid system slowdown.
  CONFIDENCE: medium
  TRIGGERS: 2
  EXPIRES: 2026-05-31
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

## Pending (Awaiting User Review)
<!-- Conflicts and uncertain rules staged here until user approves -->
