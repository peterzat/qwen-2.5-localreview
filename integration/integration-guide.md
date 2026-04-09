# Integration Guide: Adding qwen-2.5-localreview to review-external.sh

Apply these changes to `~/src/zat.env/bin/review-external.sh` from a zat.env session.

## 1. Update header comment (line ~11)

Add after the `REVIEW_TIMEOUT` line:

```
#   LOCAL_REVIEW_SCRIPT (path to review.py), LOCAL_REVIEW_VENV (path to .venv)
```

## 2. Add HAS_LOCAL gate (after line ~41)

After `[[ -n "${GEMINI_API_KEY:-}" ]] && HAS_GOOGLE=true`:

```bash
HAS_LOCAL=false
[[ -n "${LOCAL_REVIEW_SCRIPT:-}" && -n "${LOCAL_REVIEW_VENV:-}" ]] && HAS_LOCAL=true
```

## 3. Update no-providers early exit (line ~43)

Change:
```bash
if ! ${HAS_OPENAI} && ! ${HAS_GOOGLE}; then
```

To:
```bash
if ! ${HAS_OPENAI} && ! ${HAS_GOOGLE} && ! ${HAS_LOCAL}; then
```

## 4. Add call_local function

After the `call_google` function (before the `# --- Main` section), paste the
contents of `integration/call_local.sh`.

## 5. Add LOCAL_OUT temp file and update trap (line ~289)

After `GOOGLE_OUT=$(mktemp)`:
```bash
LOCAL_OUT=$(mktemp)
```

After `GOOGLE_PID=""`:
```bash
LOCAL_PID=""
```

Update the trap to include `${LOCAL_OUT:-}`:
```bash
trap 'wait 2>/dev/null; rm -f "${SYSTEM_FILE}" "${USER_FILE}" "${OPENAI_OUT:-}" "${GOOGLE_OUT:-}" "${LOCAL_OUT:-}"' EXIT
```

## 6. Add parallel launch

After the Google launch block:
```bash
if ${HAS_LOCAL}; then
  call_local > "${LOCAL_OUT}" 2>&1 &
  LOCAL_PID=$!
fi
```

## 7. Add wait

After the Google wait:
```bash
[[ -n "${LOCAL_PID}" ]] && wait "${LOCAL_PID}" || true
```

## 8. Add LOCAL_OUT to output demux loop

Change:
```bash
for outfile in "${OPENAI_OUT}" "${GOOGLE_OUT}"; do
```

To:
```bash
for outfile in "${OPENAI_OUT}" "${GOOGLE_OUT}" "${LOCAL_OUT}"; do
```

## Config

Add to `~/.config/claude-reviewers/.env` (done automatically by `setup.sh`):

```bash
# --- Local (vLLM offline, qwen-2.5-localreview) ---
LOCAL_REVIEW_SCRIPT=/home/peter/src/qwen-2.5-localreview/review.py
LOCAL_REVIEW_VENV=/home/peter/src/qwen-2.5-localreview/.venv
#LOCAL_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
```

## Testing

After applying changes, run the existing test suite from zat.env:

```bash
cd ~/src/zat.env
tests/test-review-external.sh
```

The local provider will be tested via `tests/test-call-local.sh` in this project.
