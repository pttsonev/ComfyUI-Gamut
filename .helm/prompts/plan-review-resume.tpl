The plan at `{{TARGET}}` has been updated since your previous review. Re-read it and produce
an incremental review:

  1. For each prior finding: quote it briefly, then state addressed / not addressed /
     partially addressed with the line numbers that resolved (or didn't).
  2. Flag any NEW issues introduced by the edits.

## Implementer notes

Findings explicitly marked as intentional decisions with a doc-update to-do should NOT be
re-flagged — the plan IS the change request for those docs.

{{IMPLEMENTER_NOTES}}

End with the same tag on its own line:
  APPROVED
  REQUEST_CHANGES
  NEEDS_REWORK

{{EXTRA_PROMPT}}
