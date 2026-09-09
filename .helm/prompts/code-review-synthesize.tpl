The review loop for `{{TARGET}}` has converged (or been capped). Produce a consolidated
final review for archival — the canonical record of how this change was reviewed.

Cover every finding from the whole thread — addressed, overridden, or open — with final
status and file:line references.

## Format (markdown)

# Code Review: <feature or change name>
**Review Date**: <YYYY-MM-DD from the context block>
**Version**: <x.y.z>   (leave as <x.y.z> — the requester fills it at release)
**Files Reviewed**: <from git diff --name-only HEAD>
**Plan**: <{{TARGET}} if a plan path, else "no plan — unplanned change">

## Executive Summary
<1-3 sentences; end with the verdict line>

## Findings
### Critical / ### Major / ### Minor / ### Suggestions
<each: short title, file:line, description, disposition (addressed / overridden / open);
"None." where empty>

## Verdict
**<APPROVED / APPROVED with observations / NEEDS REVISION>**

Output only the rendered markdown — no preamble. After the review, on its own line:
PROMOTION_READY

{{EXTRA_PROMPT}}
