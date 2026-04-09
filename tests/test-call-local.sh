#!/usr/bin/env bash
set -euo pipefail

# Tests for the call_local bash function (integration/call_local.sh).
# Tests guard logic and output contract without requiring a GPU.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

FAILS=0
TOTAL=0
pass() { TOTAL=$((TOTAL + 1)); printf '  ok   %s\n' "$1"; }
fail() { TOTAL=$((TOTAL + 1)); FAILS=$((FAILS + 1)); printf '  FAIL %s\n' "$1"; }

TEST_DIR=$(mktemp -d)
cleanup() { rm -rf "${TEST_DIR}"; }
trap cleanup EXIT

# Set up the scaffolding that review-external.sh provides to provider functions.
SYSTEM_FILE="${TEST_DIR}/system.txt"
USER_FILE="${TEST_DIR}/user.txt"
TIMEOUT=300

echo "You are a reviewer." > "${SYSTEM_FILE}"
echo "Review this diff." > "${USER_FILE}"

export SYSTEM_FILE USER_FILE TIMEOUT

# Source the function.
# shellcheck source=../integration/call_local.sh
source "${REPO_DIR}/integration/call_local.sh"

# ============================================================
echo "==> Test 1: fail-open with nonexistent script"
# ============================================================

LOCAL_REVIEW_SCRIPT="/nonexistent/review.py"
LOCAL_REVIEW_VENV="/nonexistent/.venv"
export LOCAL_REVIEW_SCRIPT LOCAL_REVIEW_VENV

STDOUT_FILE="${TEST_DIR}/stdout1.txt"
STDERR_FILE="${TEST_DIR}/stderr1.txt"
call_local > "${STDOUT_FILE}" 2>"${STDERR_FILE}" || true

if [[ ! -s "${STDOUT_FILE}" ]]; then
  pass "nonexistent script: no findings on stdout"
else
  fail "nonexistent script: unexpected stdout: $(cat "${STDOUT_FILE}")"
fi

if grep -q "qwen" "${STDERR_FILE}"; then
  pass "nonexistent script: warning on stderr"
else
  fail "nonexistent script: no warning on stderr"
fi

# ============================================================
echo ""
echo "==> Test 2: fail-open with script that produces no output"
# ============================================================

EMPTY_SCRIPT="${TEST_DIR}/empty_review.py"
cat > "${EMPTY_SCRIPT}" <<'PY'
#!/usr/bin/env python3
# Produces no output (simulates empty model response).
PY

LOCAL_REVIEW_SCRIPT="${EMPTY_SCRIPT}"
LOCAL_REVIEW_VENV="${REPO_DIR}/.venv"
export LOCAL_REVIEW_SCRIPT LOCAL_REVIEW_VENV

STDOUT_FILE="${TEST_DIR}/stdout2.txt"
STDERR_FILE="${TEST_DIR}/stderr2.txt"

# Use system python if venv doesn't exist (for CI).
if [[ ! -x "${LOCAL_REVIEW_VENV}/bin/python" ]]; then
  LOCAL_REVIEW_VENV="$(python3 -c 'import sys; print(sys.prefix)')"
  # Create a fake bin/python symlink.
  mkdir -p "${TEST_DIR}/fake_venv/bin"
  ln -sf "$(which python3)" "${TEST_DIR}/fake_venv/bin/python"
  LOCAL_REVIEW_VENV="${TEST_DIR}/fake_venv"
  export LOCAL_REVIEW_VENV
fi

call_local > "${STDOUT_FILE}" 2>"${STDERR_FILE}" || true

if [[ ! -s "${STDOUT_FILE}" ]]; then
  pass "empty output: no findings on stdout"
else
  fail "empty output: unexpected stdout: $(cat "${STDOUT_FILE}")"
fi

if grep -q "Empty response" "${STDERR_FILE}"; then
  pass "empty output: 'Empty response' on stderr"
else
  fail "empty output: missing 'Empty response' on stderr"
fi

# ============================================================
echo ""
echo "==> Test 3: output tagging"
# ============================================================

TAG_SCRIPT="${TEST_DIR}/tag_review.py"
cat > "${TAG_SCRIPT}" <<'PY'
#!/usr/bin/env python3
# Simulates model output with findings.
print("[BLOCK] app.py:5 -- command injection via os.system")
print("[WARN] app.py:3 -- missing input validation")
print("[NOTE] app.py:1 -- unused import")
print("No issues found.")
PY

LOCAL_REVIEW_SCRIPT="${TAG_SCRIPT}"
export LOCAL_REVIEW_SCRIPT

STDOUT_FILE="${TEST_DIR}/stdout3.txt"
STDERR_FILE="${TEST_DIR}/stderr3.txt"
call_local > "${STDOUT_FILE}" 2>"${STDERR_FILE}" || true

# Findings should be tagged with (qwen).
if grep -q '(qwen)' "${STDOUT_FILE}"; then
  pass "tagging: findings tagged with (qwen)"
else
  fail "tagging: findings not tagged"
  echo "    stdout: $(cat "${STDOUT_FILE}")"
fi

# Severity should be preserved.
if grep -q '^\[BLOCK\] (qwen)' "${STDOUT_FILE}" && \
   grep -q '^\[WARN\] (qwen)' "${STDOUT_FILE}" && \
   grep -q '^\[NOTE\] (qwen)' "${STDOUT_FILE}"; then
  pass "tagging: all severity levels preserved"
else
  fail "tagging: missing severity levels"
fi

# "No issues found." should go to stderr, not stdout.
if grep -q "No issues found" "${STDERR_FILE}"; then
  pass "tagging: 'No issues found' routed to stderr"
else
  fail "tagging: 'No issues found' not on stderr"
fi

if ! grep -q "No issues found" "${STDOUT_FILE}"; then
  pass "tagging: 'No issues found' not on stdout"
else
  fail "tagging: 'No issues found' leaked to stdout"
fi

# No duplicate cost/timing line from call_local (review.py emits its own).
COST_LINES=$(grep -cE '\$0\.00' "${STDERR_FILE}" || true)
if [[ "${COST_LINES}" -eq 0 ]]; then
  pass "tagging: no spurious cost line"
else
  fail "tagging: unexpected cost line on stderr (duplicate)"
fi

# ============================================================
echo ""
echo "==> Test 4: timeout on hung script"
# ============================================================

HANG_SCRIPT="${TEST_DIR}/hang_review.py"
cat > "${HANG_SCRIPT}" <<'PY'
#!/usr/bin/env python3
import time
time.sleep(300)
PY

LOCAL_REVIEW_SCRIPT="${HANG_SCRIPT}"
export LOCAL_REVIEW_SCRIPT
TIMEOUT=2
export TIMEOUT

STDOUT_FILE="${TEST_DIR}/stdout4.txt"
STDERR_FILE="${TEST_DIR}/stderr4.txt"
START=$(date +%s)
call_local > "${STDOUT_FILE}" 2>"${STDERR_FILE}" || true
ELAPSED=$(( $(date +%s) - START ))

if [[ "${ELAPSED}" -lt 10 ]]; then
  pass "timeout: killed within timeout"
else
  fail "timeout: took ${ELAPSED}s (expected < 10)"
fi

if [[ ! -s "${STDOUT_FILE}" ]]; then
  pass "timeout: no findings on stdout"
else
  fail "timeout: unexpected stdout: $(cat "${STDOUT_FILE}")"
fi

if grep -q "qwen" "${STDERR_FILE}"; then
  pass "timeout: warning on stderr"
else
  fail "timeout: no warning on stderr"
fi

# ============================================================
echo ""
echo "==> Results: ${TOTAL} checks, ${FAILS} failures"
# ============================================================

if [[ "${FAILS}" -gt 0 ]]; then
  exit 1
fi
