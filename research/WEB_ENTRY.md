# WEB_ENTRY - entry point for Web (ChatGPT) sessions

This repository uses Research OS. Web scientific-reasoning sessions start here.

## Rules

1. Repository evidence overrides chat memory. If chat memory conflicts with any
   file in this repository, the repository wins.
2. `diagnosis/FINAL_METHOD_CANON.md` remains the ultimate canonical method
   definition. Nothing in `research/` overrides it.
3. `research/METHOD_SPEC.md` is a navigation mirror of the canon, not an
   independent authority. On any conflict, the canon wins.
4. Read order:
   1. `research/WEB_ENTRY.md` (this file);
   2. `research/STATE.md` (current status and handoff).
5. Read the remaining state only as needed for the task:
   - `research/METHOD_SPEC.md` for method orientation;
   - `research/DECISIONS.md` for accepted decisions;
   - `research/CLAIMS.md` for claim status;
   - the specific task packet under `research/tasks/`;
   - the specific result packet under `research/results/`.
6. Web ChatGPT is responsible for scientific review and task design.
7. Local agents execute bounded tasks and must not promote scientific claims;
   claim promotion requires an explicit decision (see `research/WORKFLOW.md`
   and `research/CLAIMS.md`).
8. Do not silently change frozen scientific assumptions, accepted claims,
   dataset/split definitions, evaluation rules, or experiment provenance.

For the scientific canon itself, follow the reading order in
`diagnosis/README.md`: `FINAL_METHOD_CANON.md`, `FINAL_RESULT_LEDGER.md`,
`REJECTED_METHOD_HYPOTHESES.md`.
