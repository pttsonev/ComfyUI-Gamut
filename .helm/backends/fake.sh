#!/usr/bin/env bash
# Fake backend — test double implementing backends/CONTRACT.md.
# Echoes the rendered prompt into the review file (so tests assert rendering)
# and dumps the received env contract into the events file (so tests assert
# role→model/sandbox resolution). Never calls a model.
set -euo pipefail
cmd="$1"
{
  printf 'cmd=%s\n'     "$cmd"
  printf 'role=%s\n'    "${HELM_ROLE-}"
  printf 'target=%s\n'  "${HELM_TARGET-}"
  printf 'model=%s\n'   "${PEER_MODEL-}"
  printf 'effort=%s\n'  "${PEER_EFFORT-}"
  printf 'sandbox=%s\n' "${HELM_SANDBOX-}"
  printf 'thread=%s\n'  "${HELM_THREAD_ID-}"
} > "$HELM_EVENTS_FILE"
if [ "$cmd" = start ]; then
  printf 'fake-thread-0001\n' > "$HELM_THREAD_FILE"
fi
cat "$HELM_PROMPT_FILE" > "$HELM_REVIEW_FILE"
