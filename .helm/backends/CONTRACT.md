# Helm Backend Contract

A backend is one executable: `backends/<name>.sh`, selected by `PEER_BACKEND` in `config`.
It is invoked as `bash <name>.sh start|resume` with this environment:

| Var | Meaning |
|---|---|
| `HELM_ROLE` | `implement` \| `review` \| `ask` |
| `HELM_TARGET` | The target (path or label) — informational |
| `HELM_SANDBOX` | `workspace-write` (implement) \| `read-only` (review/ask). MUST be enforced |
| `PEER_MODEL` | Model id for this role (from config) |
| `PEER_EFFORT` | Reasoning effort hint (backend maps or ignores) |
| `HELM_PROMPT_FILE` | Rendered prompt text (read this; may be large — never an env var) |
| `HELM_THREAD_FILE` | Where `start` MUST write the persistent thread/session id |
| `HELM_THREAD_ID` | On `resume`: the previously captured id. Empty on `start` |
| `HELM_REVIEW_FILE` | Where the model's FINAL message MUST be written (plain text) |
| `HELM_EVENTS_FILE` | Where raw event/progress output SHOULD be written (`.stderr` twin for stderr) |

Obligations:

1. **start** creates a persistent conversation and writes its resume token to
   `HELM_THREAD_FILE`. If the harness has no resume capability, write a synthetic id and
   emulate continuity by prepending prior context — but say so in your backend's header.
2. **resume** continues the conversation identified by `HELM_THREAD_ID` with the new prompt.
3. The final assistant message goes to `HELM_REVIEW_FILE` verbatim. Helm skills parse its
   LAST LINE for verdict tags (`APPROVED`/`REQUEST_CHANGES`/`NEEDS_REWORK`,
   `IMPLEMENTATION_COMPLETE`/`IMPLEMENTATION_PARTIAL`, `PROMOTION_READY`, `<ACK_OK>`) —
   backends never parse or alter them.
4. Sandbox: `read-only` roles must not be able to modify the working tree; `implement` may
   write inside the repo only. Network access off by default. If the harness cannot enforce
   this, the backend must refuse `implement` (exit 1 with a clear message).
5. Exit 0 on success, non-zero on failure with a useful stderr. The driver handles
   thread-exists (exit 2) checks — backends never see that case.

The driver never inspects model output; a backend works iff any model behind it can follow
the prompt templates. That is the whole agnosticism guarantee — an `ollama` or `opencode`
backend is this one file.

Backends run in the caller's cwd — that cwd is the workspace the sandbox scopes to.

Reference implementations: `codex.sh` (real), `fake.sh` (test double).

Concurrency: the driver has no locking — two simultaneous `start` calls for the same
(role, target) can race past the thread-exists guard. Helm is a single-operator tool;
backends need not defend against concurrent invocation, but must never corrupt state
files on crash (write-then-rename if your harness makes that a real risk).
