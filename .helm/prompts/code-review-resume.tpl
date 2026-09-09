The code change for `{{TARGET}}` has been updated since your previous review. Re-run
`git status -s` and `git diff HEAD` (the same view as turn 1), then produce an incremental
review:

  1. For each prior finding: quote it briefly, state addressed / not addressed / partially
     addressed with the file:line references that resolved (or didn't).
  2. Flag any NEW issues introduced by the edits, using the same priorities, severities,
     and gate as your initial review.

## Implementer notes

Findings explicitly marked as intentional decisions, environment limitations, or with a
doc-update to-do should NOT be re-flagged.

{{IMPLEMENTER_NOTES}}

End with the same tag on its own line:
  APPROVED
  REQUEST_CHANGES
  NEEDS_REWORK

{{EXTRA_PROMPT}}
