#!/usr/bin/env bash
set -euo pipefail

# Tests for review.py guard logic and output contract.
# Tests 1-2 and 4 run without a GPU or model (~0.5s).
# Test 3 requires the model to be downloaded and a GPU available (~90s).
#
# Usage:
#   tests/test-review.sh          # fast tests only (default)
#   tests/test-review.sh --full   # include GPU inference test

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="${REPO_DIR}/review.py"
VENV="${REPO_DIR}/.venv"
PYTHON="${VENV}/bin/python"

FAILS=0
TOTAL=0
pass() { TOTAL=$((TOTAL + 1)); printf '  ok   %s\n' "$1"; }
fail() { TOTAL=$((TOTAL + 1)); FAILS=$((FAILS + 1)); printf '  FAIL %s\n' "$1"; }

RUN_FULL=false
if [[ "${1:-}" == "--full" ]]; then
  RUN_FULL=true
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv not found at ${VENV}. Run setup.sh first."
  exit 1
fi

TEST_DIR=$(mktemp -d)
cleanup() { rm -rf "${TEST_DIR}"; }
trap cleanup EXIT

# ============================================================
echo "==> Test 1: missing args"
# ============================================================

STDOUT=$("${PYTHON}" "${SCRIPT}" 2>/dev/null || true)
EXIT_CODE=$?
# argparse exits 2 on missing required args; the outer try/except catches it
# and exits 0 (fail-open), or argparse exits before our handler.
# Either way, no findings on stdout.
if [[ -z "${STDOUT}" ]]; then
  pass "missing args: no stdout"
else
  fail "missing args: unexpected stdout: ${STDOUT}"
fi

# ============================================================
echo ""
echo "==> Test 2: empty input file"
# ============================================================

echo "You are a reviewer." > "${TEST_DIR}/system.txt"
echo "" > "${TEST_DIR}/empty.txt"

STDERR_FILE="${TEST_DIR}/stderr2.txt"
STDOUT=$("${PYTHON}" "${SCRIPT}" --system "${TEST_DIR}/system.txt" --input "${TEST_DIR}/empty.txt" 2>"${STDERR_FILE}")
EXIT_CODE=$?

if [[ "${EXIT_CODE}" -eq 0 ]]; then
  pass "empty input: exit code 0"
else
  fail "empty input: exit code ${EXIT_CODE}"
fi

if [[ -z "${STDOUT}" ]]; then
  pass "empty input: no stdout"
else
  fail "empty input: unexpected stdout: ${STDOUT}"
fi

if grep -q "Empty input" "${STDERR_FILE}"; then
  pass "empty input: warning on stderr"
else
  fail "empty input: missing warning on stderr"
fi

# ============================================================
echo ""
echo "==> Test 3: live inference (requires GPU + model)"
# ============================================================

# Check if GPU is available and model is downloaded.
HAS_GPU=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=true
fi

MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct-AWQ}"
HF_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--${MODEL//\//--}"
HAS_MODEL=false
if [[ -d "${HF_CACHE_DIR}" ]]; then
  HAS_MODEL=true
fi

if ! ${RUN_FULL}; then
  echo "  SKIP (use --full to run GPU inference test)"
elif ${HAS_GPU} && ${HAS_MODEL}; then
  # System prompt from review-external.sh.
  cat > "${TEST_DIR}/system.txt" <<'SYSPROMPT'
You are a Principal Software Engineer performing an adversarial code review.

Review the provided diff. Report only findings you have high confidence in.

Evaluate against these dimensions:
1. Correctness: bugs, off-by-one errors, null handling, edge cases, race conditions
2. Security: hardcoded secrets, injection vectors, unsafe deserialization, path traversal, unvalidated input at trust boundaries
3. Code quality: dead code, duplication, inappropriate abstraction level
4. Solution approach: is there a simpler or more robust alternative?
5. Regression risk: could this break existing functionality?

Classify every finding:
- BLOCK: must fix before pushing (bugs, data loss, security vulnerabilities)
- WARN: should fix (missing error handling, untested critical paths)
- NOTE: informational only (optional improvements)

Format each finding EXACTLY as:
[SEVERITY] file:line -- description

If you find no issues, output exactly: No issues found.

Do not comment on formatting, naming, or style unless they indicate a functional problem. Output only the finding lines. No preamble, no summary, no explanation paragraphs.
SYSPROMPT

  cat > "${TEST_DIR}/diff.txt" <<'DIFF'
=== DIFF ===
diff --git a/app.py b/app.py
new file mode 100644
--- /dev/null
+++ b/app.py
@@ -0,0 +1,5 @@
+import os
+
+def run_command():
+    cmd = input("Enter command: ")
+    os.system(cmd)
DIFF

  STDERR_FILE="${TEST_DIR}/stderr3.txt"
  STDOUT=$("${PYTHON}" "${SCRIPT}" \
    --system "${TEST_DIR}/system.txt" \
    --input "${TEST_DIR}/diff.txt" \
    2>"${STDERR_FILE}") || true

  if [[ -n "${STDOUT}" ]]; then
    pass "live inference: produced output"
  else
    fail "live inference: no output"
  fi

  # Check that output contains properly formatted findings.
  if echo "${STDOUT}" | grep -qE '^\[(BLOCK|WARN|NOTE)\]'; then
    pass "live inference: findings in correct format"
  elif echo "${STDOUT}" | grep -q "No issues found"; then
    pass "live inference: model found no issues (valid response)"
  else
    fail "live inference: output not in expected format"
    echo "    stdout: ${STDOUT}"
  fi

  # Check stderr has token counts.
  if grep -q "qwen-2.5-localreview" "${STDERR_FILE}"; then
    pass "live inference: status on stderr"
  else
    fail "live inference: missing status on stderr"
  fi
else
  echo "  SKIP (GPU or model not available)"
  if ! ${HAS_GPU}; then echo "    no GPU detected"; fi
  if ! ${HAS_MODEL}; then echo "    model not in HF cache: ${HF_CACHE_DIR}"; fi
fi

# ============================================================
echo ""
echo "==> Test 4: oversized input truncation"
# ============================================================

echo "You are a reviewer." > "${TEST_DIR}/system.txt"

# Generate input larger than the default max (~98K chars).
"${PYTHON}" -c "print('x' * 120000)" > "${TEST_DIR}/big.txt"

STDERR_FILE="${TEST_DIR}/stderr4.txt"
# Run with a very small max_model_len to trigger truncation without needing
# the actual model. This will fail at model load (no GPU needed for the
# truncation check), but the truncation warning should appear before that.
LOCAL_MAX_MODEL_LEN=1024 "${PYTHON}" "${SCRIPT}" \
  --system "${TEST_DIR}/system.txt" \
  --input "${TEST_DIR}/big.txt" \
  2>"${STDERR_FILE}" || true

if grep -q "truncating" "${STDERR_FILE}" || grep -q "TRUNCATED" "${STDERR_FILE}"; then
  pass "oversized input: truncation warning"
else
  fail "oversized input: no truncation warning on stderr"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

# ============================================================
echo ""
echo "==> Results: ${TOTAL} checks, ${FAILS} failures"
# ============================================================

if [[ "${FAILS}" -gt 0 ]]; then
  exit 1
fi
