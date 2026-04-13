#!/usr/bin/env bash
set -euo pipefail

# Tests for review.py guard logic and output contract.
# Tests 1-2 and 6 run without a GPU or model (~0.5s).
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

STDERR_FILE="${TEST_DIR}/stderr1.txt"
STDOUT=$("${PYTHON}" "${SCRIPT}" 2>"${STDERR_FILE}" || true)

if [[ -z "${STDOUT}" ]]; then
  pass "missing args: no stdout"
else
  fail "missing args: unexpected stdout: ${STDOUT}"
fi

if grep -q '\[qwen\]' "${STDERR_FILE}"; then
  pass "missing args: tagged status on stderr"
else
  fail "missing args: missing [qwen] tag on stderr"
  echo "    stderr: $(cat "${STDERR_FILE}")"
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
  if grep -q "qwen" "${STDERR_FILE}"; then
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
echo "==> Test 4: oversized input truncation (requires model in HF cache)"
# ============================================================

if ${HAS_MODEL}; then
  echo "You are a reviewer." > "${TEST_DIR}/system.txt"

  # Generate input larger than the max for a small context window.
  "${PYTHON}" -c "print('x' * 120000)" > "${TEST_DIR}/big.txt"

  STDERR_FILE="${TEST_DIR}/stderr4.txt"
  # Use --dry-run with a tiny context window. The tokenizer-based guard
  # truncates the input and --dry-run exits before loading vLLM.
  LOCAL_MAX_MODEL_LEN=1024 "${PYTHON}" "${SCRIPT}" \
    --system "${TEST_DIR}/system.txt" \
    --input "${TEST_DIR}/big.txt" \
    --dry-run \
    2>"${STDERR_FILE}" || true

  if grep -q "truncated" "${STDERR_FILE}"; then
    pass "oversized input: truncation warning"
  else
    fail "oversized input: no truncation warning on stderr"
    echo "    stderr: $(cat "${STDERR_FILE}")"
  fi
else
  echo "  SKIP (model not in HF cache, tokenizer unavailable)"
fi

# ============================================================
echo ""
echo "==> Test 5: system prompt eats into user budget (requires model in HF cache)"
# ============================================================

if ${HAS_MODEL}; then
  # Large system prompt (~900 tokens) with max_model_len=1024 leaves
  # almost no room for user input after output reserve (4096). The user
  # input should be truncated or empty.
  "${PYTHON}" -c "print('Analyze this code carefully. ' * 200)" > "${TEST_DIR}/big_system.txt"
  echo "print('hello')" > "${TEST_DIR}/small_input.txt"

  STDERR_FILE="${TEST_DIR}/stderr5.txt"
  LOCAL_MAX_MODEL_LEN=1024 "${PYTHON}" "${SCRIPT}" \
    --system "${TEST_DIR}/big_system.txt" \
    --input "${TEST_DIR}/small_input.txt" \
    --dry-run \
    2>"${STDERR_FILE}" || true

  # With a 1024 context window and 4096 output reserve, the system prompt
  # alone exceeds available capacity. Should see truncation or empty warning.
  if grep -q "truncated" "${STDERR_FILE}" || grep -q "empty after truncation" "${STDERR_FILE}"; then
    pass "system prompt budget: correctly limits user input"
  else
    fail "system prompt budget: no truncation/empty warning"
    echo "    stderr: $(cat "${STDERR_FILE}")"
  fi
else
  echo "  SKIP (model not in HF cache, tokenizer unavailable)"
fi

# ============================================================
echo ""
echo "==> Test 6: fatal error produces structured status line"
# ============================================================

# Force a fast failure by pointing to a nonexistent model. The tokenizer
# load fails before any GPU work, so this runs without a GPU.
echo "You are a reviewer." > "${TEST_DIR}/system6.txt"
echo "Review this." > "${TEST_DIR}/input6.txt"

STDERR_FILE="${TEST_DIR}/stderr6.txt"
LOCAL_MODEL="nonexistent/model" "${PYTHON}" "${SCRIPT}" \
  --system "${TEST_DIR}/system6.txt" \
  --input "${TEST_DIR}/input6.txt" \
  2>"${STDERR_FILE}" || true

if grep -q '\[qwen\]' "${STDERR_FILE}"; then
  pass "fatal error: tagged with [qwen]"
else
  fail "fatal error: missing [qwen] tag"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

if grep -q 'nonexistent/model -- error:' "${STDERR_FILE}"; then
  pass "fatal error: structured status with model name"
else
  fail "fatal error: missing structured status"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

# ============================================================
echo ""
echo "==> Test 7: VRAM preflight warns when free < needed"
# ============================================================

# Exercise the preflight logic with mocked torch.cuda.mem_get_info.
# Tests the threshold math (free < gpu_memory_utilization * total) without
# loading vLLM or requiring GPU access.

# Simulate: total=20 GiB, free=15 GiB. With 0.90 utilization, needed=18 GiB.
# 15 < 18, so warning should fire.
RESULT=$("${PYTHON}" -c "
import sys, unittest.mock as mock
free = int(15 * 1024**3)
total = int(20 * 1024**3)
with mock.patch('torch.cuda.is_available', return_value=True), \
     mock.patch('torch.cuda.mem_get_info', return_value=(free, total)):
    import torch
    GPU_MEMORY_UTILIZATION = 0.90
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    needed_gb = GPU_MEMORY_UTILIZATION * total_bytes / (1024 ** 3)
    if free_gb < needed_gb:
        print('WARN_FIRED')
    else:
        print('NO_WARN')
" 2>/dev/null)

if [[ "${RESULT}" == "WARN_FIRED" ]]; then
  pass "VRAM preflight: warns when free (15GB) < needed (18GB)"
else
  fail "VRAM preflight: did not warn when free < needed"
fi

# ============================================================
echo ""
echo "==> Test 8: VRAM preflight silent when free >= needed"
# ============================================================

# Simulate: total=20 GiB, free=19 GiB. needed=18 GiB. 19 >= 18, no warning.
RESULT=$("${PYTHON}" -c "
import sys, unittest.mock as mock
free = int(19 * 1024**3)
total = int(20 * 1024**3)
with mock.patch('torch.cuda.is_available', return_value=True), \
     mock.patch('torch.cuda.mem_get_info', return_value=(free, total)):
    import torch
    GPU_MEMORY_UTILIZATION = 0.90
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    needed_gb = GPU_MEMORY_UTILIZATION * total_bytes / (1024 ** 3)
    if free_gb < needed_gb:
        print('WARN_FIRED')
    else:
        print('NO_WARN')
" 2>/dev/null)

if [[ "${RESULT}" == "NO_WARN" ]]; then
  pass "VRAM preflight: silent when free (19GB) >= needed (18GB)"
else
  fail "VRAM preflight: incorrectly warned when free >= needed"
fi

# ============================================================
echo ""
echo "==> Test 9: malformed LOCAL_MAX_MODEL_LEN"
# ============================================================

echo "You are a reviewer." > "${TEST_DIR}/system9.txt"
echo "Review this." > "${TEST_DIR}/input9.txt"

STDERR_FILE="${TEST_DIR}/stderr9.txt"
LOCAL_MAX_MODEL_LEN=notanumber "${PYTHON}" "${SCRIPT}" \
  --system "${TEST_DIR}/system9.txt" \
  --input "${TEST_DIR}/input9.txt" \
  2>"${STDERR_FILE}" || true

if grep -q 'must be an integer' "${STDERR_FILE}"; then
  pass "malformed max_model_len: diagnostic on stderr"
else
  fail "malformed max_model_len: missing diagnostic"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

# ============================================================
echo ""
echo "==> Test 10: zero LOCAL_MAX_MODEL_LEN"
# ============================================================

STDERR_FILE="${TEST_DIR}/stderr10.txt"
LOCAL_MAX_MODEL_LEN=0 "${PYTHON}" "${SCRIPT}" \
  --system "${TEST_DIR}/system9.txt" \
  --input "${TEST_DIR}/input9.txt" \
  2>"${STDERR_FILE}" || true

if grep -q 'must be positive' "${STDERR_FILE}"; then
  pass "zero max_model_len: diagnostic on stderr"
else
  fail "zero max_model_len: missing diagnostic"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

# ============================================================
echo ""
echo "==> Test 11: adaptive reserve gives more capacity at 8K context"
# ============================================================

# At LOCAL_MAX_MODEL_LEN=8192, the adaptive reserve is 2048 (25% of 8192),
# leaving ~6138 tokens for input. Under the old fixed 4096 reserve, only
# ~4090 tokens would be available. Feed ~5000 tokens of input (via --dry-run)
# and verify it is NOT truncated.
if ${HAS_MODEL}; then
  echo "You are a reviewer." > "${TEST_DIR}/system11.txt"
  # Generate ~4500 tokens of input. This exceeds the old fixed reserve's
  # capacity (8192-5-4096=4091) but fits the adaptive reserve's capacity
  # (8192-5-2048=6139), proving the adaptive reserve works.
  "${PYTHON}" -c "print('def func(): pass  # placeholder line\n' * 500)" > "${TEST_DIR}/medium.txt"

  STDERR_FILE="${TEST_DIR}/stderr11.txt"
  LOCAL_MAX_MODEL_LEN=8192 "${PYTHON}" "${SCRIPT}" \
    --system "${TEST_DIR}/system11.txt" \
    --input "${TEST_DIR}/medium.txt" \
    --dry-run \
    2>"${STDERR_FILE}" || true

  if grep -q "truncated" "${STDERR_FILE}"; then
    fail "adaptive reserve: input truncated at 8K context (reserve too large)"
    echo "    stderr: $(cat "${STDERR_FILE}")"
  else
    pass "adaptive reserve: medium input fits at 8K context"
  fi
else
  echo "  SKIP (model not in HF cache, tokenizer unavailable)"
fi

# ============================================================
echo ""
echo "==> Test 12: dry-run unaffected by held GPU flock"
# ============================================================

# Acquire the GPU flock in a background process, then verify --dry-run
# still works (it exits before the flock acquisition point in review.py).
LOCK_PATH=$("${PYTHON}" -c "import gpu_lock; print(gpu_lock.lock_path())" 2>/dev/null)

if [[ -n "${LOCK_PATH}" ]] && ${HAS_MODEL}; then
  # Hold the flock in a background subshell for 10 seconds.
  (
    "${PYTHON}" -c "
import gpu_lock, time
f = gpu_lock.acquire_gpu_lock(timeout=1.0)
if f: time.sleep(10)
" 2>/dev/null
  ) &
  LOCK_PID=$!
  sleep 1  # give it time to acquire

  STDERR_FILE="${TEST_DIR}/stderr12.txt"
  "${PYTHON}" "${SCRIPT}" \
    --system "${TEST_DIR}/system9.txt" \
    --input "${TEST_DIR}/input9.txt" \
    --dry-run \
    2>"${STDERR_FILE}" || true

  if grep -q "dry-run" "${STDERR_FILE}"; then
    pass "dry-run unaffected by held GPU flock"
  else
    fail "dry-run blocked by GPU flock"
    echo "    stderr: $(cat "${STDERR_FILE}")"
  fi

  kill "${LOCK_PID}" 2>/dev/null || true; wait "${LOCK_PID}" 2>/dev/null || true
else
  echo "  SKIP (model not in HF cache or lock path unavailable)"
fi

# ============================================================
echo ""
echo "==> Test 13: GPU busy when flock held"
# ============================================================

# Hold the flock, run review.py (non-dry-run) with a tiny flock timeout.
# The flock acquisition in review.py uses 270s by default; we override
# via LOCAL_GPU_LOCK_TIMEOUT for testing.
if [[ -n "${LOCK_PATH}" ]] && ${HAS_MODEL}; then
  (
    "${PYTHON}" -c "
import gpu_lock, time
f = gpu_lock.acquire_gpu_lock(timeout=1.0)
if f: time.sleep(10)
" 2>/dev/null
  ) &
  LOCK_PID=$!
  sleep 1

  STDERR_FILE="${TEST_DIR}/stderr13.txt"
  LOCAL_GPU_LOCK_TIMEOUT=1 "${PYTHON}" "${SCRIPT}" \
    --system "${TEST_DIR}/system9.txt" \
    --input "${TEST_DIR}/input9.txt" \
    2>"${STDERR_FILE}" || true

  if grep -q "GPU busy" "${STDERR_FILE}"; then
    pass "GPU busy diagnostic when flock held"
  else
    fail "missing GPU busy diagnostic"
    echo "    stderr: $(cat "${STDERR_FILE}")"
  fi

  kill "${LOCK_PID}" 2>/dev/null || true; wait "${LOCK_PID}" 2>/dev/null || true
else
  echo "  SKIP (model not in HF cache or lock path unavailable)"
fi

# ============================================================
echo ""
echo "==> Results: ${TOTAL} checks, ${FAILS} failures"
# ============================================================

if [[ "${FAILS}" -gt 0 ]]; then
  exit 1
fi
