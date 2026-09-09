#!/usr/bin/env bash
# Codex backend — drives `codex exec` per backends/CONTRACT.md.
# Derived from TRIP-workflow's codex scripts (MIT, PiLastDigit), refactored
# under the Helm env contract. Notes:
#   - stdin closed (</dev/null) so codex skips its read-from-stdin detour
#   - `codex exec resume` inherits sandbox/color from the original session;
#     passing --sandbox there is an error — do not add it.
set -euo pipefail
cmd="$1"

if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex CLI not found on PATH — install it or set PEER_BACKEND to another backend" >&2
  exit 1
fi

PROMPT="$(cat "$HELM_PROMPT_FILE")"
[ -n "$PROMPT" ] || { echo "error: empty prompt file: $HELM_PROMPT_FILE" >&2; exit 1; }

extract_thread_id() {
  local line
  line="$(grep -F 'thread.started' "$HELM_EVENTS_FILE" 2>/dev/null | head -1 || true)"
  [ -n "$line" ] || return 0
  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$line" | jq -r '.thread_id // empty' 2>/dev/null || true
  else
    printf '%s\n' "$line" | sed -n 's/.*"thread_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
  fi
}

fail_with_stderr() {
  local rc="$1" what="$2"
  echo "error: $what failed (rc=$rc); stderr tail:" >&2
  tail -20 "$HELM_EVENTS_FILE.stderr" >&2 || true
  exit 1
}

if [ "$cmd" = start ]; then
  codex exec \
    --json --skip-git-repo-check --color never \
    --sandbox "$HELM_SANDBOX" \
    -c model="$PEER_MODEL" \
    -c model_reasoning_effort="$PEER_EFFORT" \
    -o "$HELM_REVIEW_FILE" \
    "$PROMPT" \
    </dev/null >"$HELM_EVENTS_FILE" 2>"$HELM_EVENTS_FILE.stderr" \
    || fail_with_stderr $? "codex exec"
  tid="$(extract_thread_id || true)"
  if [ -z "$tid" ] || [ "$tid" = "null" ]; then
    echo "error: no thread.started event captured; first events:" >&2
    head -20 "$HELM_EVENTS_FILE" >&2
    exit 1
  fi
  printf '%s\n' "$tid" > "$HELM_THREAD_FILE"
else
  codex exec resume "$HELM_THREAD_ID" \
    --json --skip-git-repo-check \
    -c model="$PEER_MODEL" \
    -c model_reasoning_effort="$PEER_EFFORT" \
    -o "$HELM_REVIEW_FILE" \
    "$PROMPT" \
    </dev/null >"$HELM_EVENTS_FILE" 2>"$HELM_EVENTS_FILE.stderr" \
    || fail_with_stderr $? "codex exec resume"
fi
