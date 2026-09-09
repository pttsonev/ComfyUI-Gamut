#!/usr/bin/env bash
# Helm peer driver — the model-agnostic seam to the second-model peer.
#
# Usage:
#   peer.sh start  --role implement|review|ask --prompt <tpl> [--notes "..."] <target> [extra...]
#   peer.sh resume --role <role>               --prompt <tpl> [--notes "..."] <target> [extra...]
#   peer.sh reset  --role <role> <target>
#   peer.sh show   --role <role> <target>
#
# Exit codes: 0 ok · 1 backend/show failure · 2 thread exists (start) or missing (resume) · 64 usage.
# State: peer/state/<role>/<key>.{thread,review.txt,events.ndjson,prompt.txt}
# Verdict tags (APPROVED / REQUEST_CHANGES / NEEDS_REWORK / IMPLEMENTATION_COMPLETE|PARTIAL)
# are a prompt/skill concern — the driver and backends never parse them.
set -euo pipefail

PEER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "usage: peer.sh start|resume|reset|show --role <implement|review|ask> [--prompt <name>] [--notes '...'] <target> [extra...]" >&2
  exit 64
}

# --- config ---------------------------------------------------------------
if [ -f "$PEER_DIR/config" ]; then
  # CRs stripped so a CRLF-saved config (Windows editors) can't poison values
  # with trailing \r — that breaks backend paths and garbles error output.
  # shellcheck source=/dev/null
  source <(tr -d '\r' < "$PEER_DIR/config")
fi
PEER_BACKEND="${PEER_BACKEND:-codex}"
PEER_MODEL_IMPLEMENT="${PEER_MODEL_IMPLEMENT:-gpt-6-astra}"
PEER_MODEL_REVIEW="${PEER_MODEL_REVIEW:-gpt-6-astra}"
PEER_MODEL_ASK="${PEER_MODEL_ASK:-$PEER_MODEL_REVIEW}"
PEER_EFFORT="${PEER_EFFORT:-xhigh}"

# --- args -----------------------------------------------------------------
CMD="${1-}"; [ -n "$CMD" ] && shift || usage
case "$CMD" in start|resume|reset|show) ;; *) usage ;; esac

ROLE="" PROMPT_NAME="" NOTES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --role)     [ $# -ge 2 ] || { echo "error: --role requires a value" >&2; exit 64; }; ROLE="$2"; shift 2 ;;
    --role=*)   ROLE="${1#*=}"; shift ;;
    --prompt)   [ $# -ge 2 ] || { echo "error: --prompt requires a value" >&2; exit 64; }; PROMPT_NAME="$2"; shift 2 ;;
    --prompt=*) PROMPT_NAME="${1#*=}"; shift ;;
    --notes)    [ $# -ge 2 ] || { echo "error: --notes requires a value" >&2; exit 64; }; NOTES="$2"; shift 2 ;;
    --notes=*)  NOTES="${1#*=}"; shift ;;
    --) shift; break ;;
    -*) echo "error: unknown flag: $1" >&2; exit 64 ;;
    *) break ;;
  esac
done
case "$ROLE" in implement|review|ask) ;; *) echo "error: --role must be implement|review|ask" >&2; exit 64 ;; esac
[ $# -ge 1 ] || { echo "error: missing <target>" >&2; exit 64; }
TARGET="$1"; shift
if [ -z "$TARGET" ]; then echo "error: target must be non-empty" >&2; exit 64; fi
EXTRA_PROMPT="${*:-}"

# --- state ------------------------------------------------------------
STATE_DIR="$PEER_DIR/state/$ROLE"

# Per-target key: literal, deterministic sanitization of the argument string.
# Same string → same key, regardless of cwd or whether the path exists yet.
# Skills must pass targets consistently (repo-relative paths recommended).
target_key() {
  printf '%s' "$1" | sed 's|^/||; s|:||g; s|/|__|g; s|[^A-Za-z0-9._-]|_|g'
}
KEY="$(target_key "$TARGET")"
THREAD_FILE="$STATE_DIR/$KEY.thread"
REVIEW_FILE="$STATE_DIR/$KEY.review.txt"
EVENTS_FILE="$STATE_DIR/$KEY.events.ndjson"
RENDERED_PROMPT="$STATE_DIR/$KEY.prompt.txt"

# --- reset / show ---------------------------------------------------------
if [ "$CMD" = reset ]; then
  removed=0
  for f in "$THREAD_FILE" "$REVIEW_FILE" "$EVENTS_FILE" "$EVENTS_FILE.stderr" "$RENDERED_PROMPT"; do
    if [ -f "$f" ]; then rm -- "$f"; echo "removed $f"; removed=$((removed+1)); fi
  done
  if [ "$removed" = 0 ]; then echo "no peer state for $TARGET (role=$ROLE)"; fi
  exit 0
fi
if [ "$CMD" = show ]; then
  if [ ! -f "$REVIEW_FILE" ]; then
    echo "error: no peer output on file for $TARGET (role=$ROLE)" >&2
    exit 1
  fi
  if [ -f "$THREAD_FILE" ]; then echo "thread id: $(cat "$THREAD_FILE")"; fi
  echo "output file: $REVIEW_FILE"
  echo "---"
  cat "$REVIEW_FILE"
  exit 0
fi

mkdir -p "$STATE_DIR"

BACKEND="$PEER_DIR/backends/$PEER_BACKEND.sh"
if [ ! -f "$BACKEND" ]; then
  echo "error: unknown backend '$PEER_BACKEND' ($BACKEND not found)" >&2
  exit 64
fi

# --- start/resume guards --------------------------------------------------
if [ "$CMD" = start ] && [ -f "$THREAD_FILE" ]; then
  echo "error: peer session already exists for $TARGET (role=$ROLE)" >&2
  echo "       thread id: $(cat "$THREAD_FILE")" >&2
  echo "       use resume to continue, or reset to start fresh." >&2
  exit 2
fi
if [ "$CMD" = resume ] && [ ! -f "$THREAD_FILE" ]; then
  echo "error: no peer session for $TARGET (role=$ROLE) — run start first." >&2
  exit 2
fi

# --- render prompt --------------------------------------------------------
if [ -z "$PROMPT_NAME" ]; then echo "error: --prompt <name> required for $CMD" >&2; exit 64; fi
TPL="$PEER_DIR/prompts/$PROMPT_NAME.tpl"
if [ ! -f "$TPL" ]; then echo "error: prompt template not found: $TPL" >&2; exit 64; fi
# Render via ENVIRON + index/substr: immune to awk -v escape processing and
# to gsub's '&' replacement semantics. POSIX-awk portable.
HELM_TPL_TARGET="$TARGET" HELM_TPL_EXTRA="$EXTRA_PROMPT" HELM_TPL_NOTES="$NOTES" \
awk '
  function subst(s, k, v,    i, out) {
    out = ""
    while ((i = index(s, k)) > 0) {
      out = out substr(s, 1, i - 1) v
      s = substr(s, i + length(k))
    }
    return out s
  }
  {
    s = $0
    s = subst(s, "{{TARGET}}",            ENVIRON["HELM_TPL_TARGET"])
    s = subst(s, "{{EXTRA_PROMPT}}",      ENVIRON["HELM_TPL_EXTRA"])
    s = subst(s, "{{IMPLEMENTER_NOTES}}", ENVIRON["HELM_TPL_NOTES"])
    print s
  }
' "$TPL" > "$RENDERED_PROMPT"

# --- role → model/sandbox -------------------------------------------------
case "$ROLE" in
  implement) PEER_MODEL="$PEER_MODEL_IMPLEMENT"; HELM_SANDBOX="workspace-write" ;;
  review)    PEER_MODEL="$PEER_MODEL_REVIEW";    HELM_SANDBOX="read-only" ;;
  ask)       PEER_MODEL="$PEER_MODEL_ASK";       HELM_SANDBOX="read-only" ;;
esac

# --- backend env contract (documented in backends/CONTRACT.md) ------------
export PEER_MODEL PEER_EFFORT
export HELM_ROLE="$ROLE" HELM_TARGET="$TARGET" HELM_SANDBOX
export HELM_THREAD_FILE="$THREAD_FILE" HELM_REVIEW_FILE="$REVIEW_FILE"
export HELM_EVENTS_FILE="$EVENTS_FILE" HELM_PROMPT_FILE="$RENDERED_PROMPT"
if [ -f "$THREAD_FILE" ]; then HELM_THREAD_ID="$(cat "$THREAD_FILE")"; else HELM_THREAD_ID=""; fi
export HELM_THREAD_ID

# --- dispatch -------------------------------------------------------------
if ! bash "$BACKEND" "$CMD"; then
  echo "error: backend '$PEER_BACKEND' $CMD failed for $TARGET (role=$ROLE)" >&2
  exit 1
fi

if [ ! -f "$REVIEW_FILE" ]; then
  echo "error: backend '$PEER_BACKEND' exited 0 but wrote no output to $REVIEW_FILE (contract violation)" >&2
  exit 1
fi

echo "peer $CMD ok  role=$ROLE  backend=$PEER_BACKEND  model=$PEER_MODEL  effort=$PEER_EFFORT"
if [ -f "$THREAD_FILE" ]; then echo "  thread: $(cat "$THREAD_FILE")"; fi
echo "  output: $REVIEW_FILE"
echo "---"
cat "$REVIEW_FILE"
