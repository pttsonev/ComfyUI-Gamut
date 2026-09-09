#!/usr/bin/env bash
# helm-meta.sh <bump|reset|status> [meta-path] — dream-counter ops on meta.json.
# Vendored to .helm/meta.sh by /helm-init; default path resolves from the cwd.
# Python does the JSON (atomic tmp+rename, never clobbers on bad input); bash
# owns the exact output strings so no unicode dies in a Windows codec.
set -u

CMD="${1:-}"; META="${2:-docs/agent/memory/meta.json}"
case "$CMD" in bump|reset|status) ;; *)
  echo "usage: helm-meta.sh <bump|reset|status> [meta-path]" >&2; exit 64 ;;
esac
[ -f "$META" ] || { echo "helm-meta: meta file not found: $META" >&2; exit 1; }

# Native Windows python cannot open MSYS-form paths (/c/..., /tmp/...) passed
# inside argv it receives from a script; hand it the Windows form when cygpath
# exists (same rationale as tests/harness.sh). No-op on Linux/macOS.
PYMETA="$META"
if command -v cygpath >/dev/null 2>&1; then
  W="$(cygpath -w "$META" 2>/dev/null | tr '\\' '/')"
  [ -n "$W" ] && PYMETA="$W"
fi

TODAY="$(date +%F)"
OUT="$(python - "$CMD" "$PYMETA" "$TODAY" <<'PY'
import json, os, sys, tempfile
cmd, path, today = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
except (OSError, ValueError, TypeError, AttributeError):
    sys.stderr.write("helm-meta: %s missing or not valid JSON; nothing written\n" % path)
    sys.exit(1)
n = int(d.get("sessions_since_dream", 0))
t = int(d.get("dream_threshold", 5))
if cmd == "bump":
    n += 1
    d["sessions_since_dream"] = n
elif cmd == "reset":
    n = 0
    d["sessions_since_dream"] = n
    d["last_dream"] = today
if cmd in ("bump", "reset"):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
print("%s %d %d" % (cmd.upper(), n, t))
PY
)" || exit 1

set -- $OUT
case "$1" in
  STATUS) if [ "$2" -lt "$3" ]; then printf 'dream: %s/%s — not due\n' "$2" "$3"
          else printf 'dream due → run /helm-dream next\n'; fi ;;
  BUMP)   echo "$2" ;;
  RESET)  printf 'dream counter reset — sessions_since_dream=0, last_dream=%s\n' "$TODAY" ;;
  *)      echo "helm-meta: unexpected helper output: $OUT" >&2; exit 1 ;;
esac
