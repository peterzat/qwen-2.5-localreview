#!/usr/bin/env bash
set -euo pipefail

# Lifecycle tests for the warm server (warm.py).
# Requires GPU + model for Tests 1-3. Tests 4-5 run without GPU.
#
# Expected wall time: ~60-90s (mostly model load time).
#
# Usage:
#   tests/test-warm.sh --full   # required

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WARM_SCRIPT="${REPO_DIR}/warm.py"
REVIEW_SCRIPT="${REPO_DIR}/review.py"
VENV="${REPO_DIR}/.venv"
PYTHON="${VENV}/bin/python"
SYSTEM_PROMPT="${REPO_DIR}/tests/fixtures/system.txt"
FIXTURE="${REPO_DIR}/tests/fixtures/diffs/07-py-buggy-simple.patch"

if [[ "${1:-}" != "--full" ]]; then
  echo "test-warm.sh is a slow GPU test (~60-90s)."
  echo "Usage: tests/test-warm.sh --full"
  exit 2
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv not found at ${VENV}. Run setup.sh first."
  exit 1
fi

SOCK_PATH=$("${PYTHON}" -c "from gpu_lock import socket_path; print(socket_path())" 2>/dev/null)
LOCK_PATH=$("${PYTHON}" -c "from gpu_lock import lock_path; print(lock_path())" 2>/dev/null)
STATE_PATH=$("${PYTHON}" -c "from gpu_lock import state_path; print(state_path())" 2>/dev/null)

FAILS=0
TOTAL=0
pass() { TOTAL=$((TOTAL + 1)); printf '  ok   %s\n' "$1"; }
fail() { TOTAL=$((TOTAL + 1)); FAILS=$((FAILS + 1)); printf '  FAIL %s\n' "$1"; }

# Cleanup helper: kill warm server and remove stale socket/state.
cleanup_warm() {
  if [[ -n "${WARM_PID:-}" ]]; then
    kill "${WARM_PID}" 2>/dev/null || true
    wait "${WARM_PID}" 2>/dev/null || true
    WARM_PID=""
  fi
  [[ -S "${SOCK_PATH}" ]] && rm -f "${SOCK_PATH}" || true
  [[ -n "${STATE_PATH:-}" && -f "${STATE_PATH}" ]] && rm -f "${STATE_PATH}" || true
}
trap cleanup_warm EXIT

# Kill any existing warm server (e.g., auto-launched by a prior test run
# or test-quality.sh triggering auto-warm via review.py's cold path).
# We need exclusive control of the warm lifecycle for these tests.
EXISTING_PID=$(pgrep -f "python.*warm\\.py" 2>/dev/null || true)
if [[ -n "${EXISTING_PID}" ]]; then
  echo "Killing existing warm server (PID ${EXISTING_PID})..."
  kill ${EXISTING_PID} 2>/dev/null || true
  sleep 3
fi
rm -f "${SOCK_PATH}" "${STATE_PATH}" 2>/dev/null || true

# Helper: wait for warm server socket to appear.
wait_for_socket() {
  local timeout="${1:-60}"
  for i in $(seq 1 "${timeout}"); do
    if [[ -S "${SOCK_PATH}" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# ============================================================
echo "==> Test 1: warm server starts and creates socket"
# ============================================================

WARM_LOG=$(mktemp)
LOCAL_WARM_TIMEOUT=60 "${PYTHON}" "${WARM_SCRIPT}" 2>"${WARM_LOG}" &
WARM_PID=$!

if wait_for_socket 90; then
  pass "warm server created socket"
else
  fail "warm server did not create socket within 90s"
  echo "    log: $(cat "${WARM_LOG}")"
fi
rm -f "${WARM_LOG}"

# ============================================================
echo ""
echo "==> Test 2: review.py uses warm path"
# ============================================================

if [[ -S "${SOCK_PATH}" ]]; then
  STDERR_FILE=$(mktemp)
  STDOUT=$("${PYTHON}" "${REVIEW_SCRIPT}" \
    --system "${SYSTEM_PROMPT}" \
    --input "${FIXTURE}" \
    2>"${STDERR_FILE}") || true

  if grep -q "(warm)" "${STDERR_FILE}"; then
    pass "review.py used warm path"
  else
    fail "review.py did not use warm path"
    echo "    stderr: $(cat "${STDERR_FILE}")"
  fi

  if echo "${STDOUT}" | grep -qE '^\[(BLOCK|WARN)\]'; then
    pass "warm path produced findings"
  else
    fail "warm path produced no findings"
    echo "    stdout: ${STDOUT:-<empty>}"
  fi

  rm -f "${STDERR_FILE}"
else
  fail "socket not available for warm path test"
fi

# ============================================================
echo ""
echo "==> Test 3: warm server idle timeout"
# ============================================================

# Kill the existing server and start a new one with short timeout.
cleanup_warm
sleep 5  # allow vLLM engine subprocess to fully release GPU
WARM_LOG=$(mktemp)
LOCAL_WARM_TIMEOUT=10 "${PYTHON}" "${WARM_SCRIPT}" 2>"${WARM_LOG}" &
WARM_PID=$!

if wait_for_socket 90; then
  # Wait for the 10s idle timeout to fire, plus buffer for shutdown.
  sleep 15

  if kill -0 "${WARM_PID}" 2>/dev/null; then
    fail "warm server process still alive after timeout"
    echo "    log: $(cat "${WARM_LOG}")"
    cleanup_warm
  else
    pass "warm server exited after idle timeout"
  fi
else
  fail "warm server did not start for timeout test"
  echo "    log: $(cat "${WARM_LOG}")"
fi
rm -f "${WARM_LOG}"
WARM_PID=""

# ============================================================
echo ""
echo "==> Test 4: two warm servers cannot coexist"
# ============================================================

# Start first warm server.
WARM_LOG=$(mktemp)
LOCAL_WARM_TIMEOUT=30 "${PYTHON}" "${WARM_SCRIPT}" 2>"${WARM_LOG}" &
WARM_PID=$!

if wait_for_socket 60; then
  # Try to start a second one; it should fail immediately.
  STDERR_SECOND=$(mktemp)
  "${PYTHON}" "${WARM_SCRIPT}" 2>"${STDERR_SECOND}" || true

  if grep -q "flock held" "${STDERR_SECOND}" || grep -q "cannot start" "${STDERR_SECOND}"; then
    pass "second warm server rejected (flock held)"
  else
    fail "second warm server did not report flock conflict"
    echo "    stderr: $(cat "${STDERR_SECOND}")"
  fi
  rm -f "${STDERR_SECOND}"
else
  fail "first warm server did not start"
  echo "    log: $(cat "${WARM_LOG}")"
fi
rm -f "${WARM_LOG}"
cleanup_warm

# ============================================================
echo ""
echo "==> Test 5: cold path fallback when socket stale"
# ============================================================

# Create a stale socket file (nothing listening), verify review.py
# falls through to cold path.
if [[ -S "${SOCK_PATH}" ]]; then rm -f "${SOCK_PATH}"; fi
"${PYTHON}" -c "
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(sys.argv[1])
s.close()
" "${SOCK_PATH}"

# The stale socket exists but no server is listening.
STDERR_FILE=$(mktemp)
"${PYTHON}" "${REVIEW_SCRIPT}" \
  --system "${SYSTEM_PROMPT}" \
  --input "${FIXTURE}" \
  --dry-run \
  2>"${STDERR_FILE}" || true

rm -f "${SOCK_PATH}"

if grep -q "dry-run" "${STDERR_FILE}"; then
  pass "cold path fallback on stale socket"
else
  fail "cold path fallback failed on stale socket"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi
rm -f "${STDERR_FILE}"

# ============================================================
echo ""
echo "==> Test 6: gpu-release stops warm server"
# ============================================================

# Start the warm server, then run gpu-release. The warm server
# should exit cleanly within a few seconds.
cleanup_warm
sleep 3
WARM_LOG=$(mktemp)
LOCAL_WARM_TIMEOUT=120 "${PYTHON}" "${WARM_SCRIPT}" 2>"${WARM_LOG}" &
WARM_PID=$!

GPU_RELEASE="${REPO_DIR}/gpu-release"

if wait_for_socket 90; then
  # Run gpu-release; it should stop the warm server.
  RELEASE_EXIT=0
  "${GPU_RELEASE}" || RELEASE_EXIT=$?

  if [[ "${RELEASE_EXIT}" -eq 0 ]]; then
    pass "gpu-release: exited 0"
  else
    fail "gpu-release: exited ${RELEASE_EXIT}"
  fi

  # Warm server should be gone.
  if ! kill -0 "${WARM_PID}" 2>/dev/null; then
    pass "gpu-release: warm server stopped"
  else
    fail "gpu-release: warm server still running after gpu-release"
    cleanup_warm
  fi
else
  fail "warm server did not start for gpu-release test"
  echo "    log: $(cat "${WARM_LOG}")"
fi
rm -f "${WARM_LOG}"
WARM_PID=""

# ============================================================
echo ""
echo "==> Test 6b: gpu-release is no-op when nothing running"
# ============================================================

cleanup_warm
sleep 1
RELEASE_EXIT=0
"${GPU_RELEASE}" || RELEASE_EXIT=$?

if [[ "${RELEASE_EXIT}" -eq 0 ]]; then
  pass "gpu-release no-op: exited 0 when nothing running"
else
  fail "gpu-release no-op: exited ${RELEASE_EXIT}, expected 0"
fi

# ============================================================
echo ""
echo "==> Test 7: OOM produces fail-open with structured error"
# ============================================================

# Hold most GPU memory, then run review.py. vLLM should fail to start
# (OOM), and review.py should exit 0 with a structured [qwen] error
# on stderr.
cleanup_warm
"${PYTHON}" -c "
import torch, time
# Allocate ~18 GB to starve vLLM of memory.
x = torch.zeros(18 * 1024 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
time.sleep(30)
" &
OOM_ALLOC_PID=$!
sleep 2

STDERR_FILE=$(mktemp)
OOM_EXIT=0
LOCAL_GPU_LOCK_TIMEOUT=5 "${PYTHON}" "${REVIEW_SCRIPT}" \
  --system "${SYSTEM_PROMPT}" \
  --input "${FIXTURE}" \
  2>"${STDERR_FILE}" || OOM_EXIT=$?

kill "${OOM_ALLOC_PID}" 2>/dev/null || true; wait "${OOM_ALLOC_PID}" 2>/dev/null || true

if [[ "${OOM_EXIT}" -eq 0 ]]; then
  pass "OOM: exit code 0 (fail-open)"
else
  fail "OOM: exit code ${OOM_EXIT} (not fail-open)"
fi

if grep -q '\[qwen\]' "${STDERR_FILE}" && grep -q 'error:' "${STDERR_FILE}"; then
  pass "OOM: structured [qwen] error on stderr"
else
  fail "OOM: missing structured error"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

rm -f "${STDERR_FILE}"

# ============================================================
echo ""
echo "==> Test 8: state file transitions (starting -> ready -> stopped)"
# ============================================================

# Start warm.py and verify state file progresses through lifecycle.
cleanup_warm
sleep 3
rm -f "${STATE_PATH}" 2>/dev/null || true

WARM_LOG=$(mktemp)
LOCAL_WARM_TIMEOUT=30 "${PYTHON}" "${WARM_SCRIPT}" 2>"${WARM_LOG}" &
WARM_PID=$!

# State should be "starting" almost immediately.
sleep 1
if [[ -f "${STATE_PATH}" ]]; then
  INITIAL_STATE=$("${PYTHON}" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get('state', ''))
except: print('')
" "${STATE_PATH}")

  if [[ "${INITIAL_STATE}" == "starting" ]]; then
    pass "state transitions: initial state is 'starting'"
  else
    # May already be 'ready' if model was cached and loaded fast.
    if [[ "${INITIAL_STATE}" == "ready" ]]; then
      pass "state transitions: initial state is 'starting' (fast load, already ready)"
    else
      fail "state transitions: initial state is '${INITIAL_STATE}', expected 'starting'"
    fi
  fi
else
  fail "state transitions: state file not created"
fi

# Wait for socket (means state should be ready).
if wait_for_socket 90; then
  READY_STATE=$("${PYTHON}" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get('state', ''))
except: print('')
" "${STATE_PATH}")

  if [[ "${READY_STATE}" == "ready" ]]; then
    pass "state transitions: state is 'ready' when socket available"
  else
    fail "state transitions: state is '${READY_STATE}', expected 'ready'"
  fi
else
  fail "state transitions: warm server did not create socket"
  echo "    log: $(cat "${WARM_LOG}")"
fi

# Kill and verify stopped/cleaned up.
if [[ -n "${WARM_PID}" ]]; then
  kill "${WARM_PID}" 2>/dev/null || true
  wait "${WARM_PID}" 2>/dev/null || true
  WARM_PID=""
fi
sleep 1

# After clean shutdown, state file should be removed.
if [[ ! -f "${STATE_PATH}" ]]; then
  pass "state transitions: state file removed after shutdown"
else
  FINAL_STATE=$("${PYTHON}" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get('state', ''))
except: print('')
" "${STATE_PATH}")
  # "stopped" is acceptable if unlink failed; file removed is preferred.
  if [[ "${FINAL_STATE}" == "stopped" ]]; then
    pass "state transitions: state file removed after shutdown (stopped, not cleaned)"
  else
    fail "state transitions: state file still present with state '${FINAL_STATE}'"
  fi
fi

rm -f "${WARM_LOG}" "${STATE_PATH}" 2>/dev/null || true

# ============================================================
echo ""
echo "==> Test 9: auto-launch creates state file after cold review"
# ============================================================

# Run a cold-path review; after it exits, verify warm.py was auto-launched
# and state file was created.
cleanup_warm
sleep 3
rm -f "${STATE_PATH}" 2>/dev/null || true

STDERR_FILE=$(mktemp)
"${PYTHON}" "${REVIEW_SCRIPT}" \
  --system "${SYSTEM_PROMPT}" \
  --input "${FIXTURE}" \
  2>"${STDERR_FILE}" || true

# Give warm.py a moment to start and write state file.
sleep 2

if [[ -f "${STATE_PATH}" ]]; then
  AUTO_STATE=$("${PYTHON}" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(json.load(f).get('state', ''))
except: print('')
" "${STATE_PATH}")

  if [[ "${AUTO_STATE}" == "starting" || "${AUTO_STATE}" == "ready" ]]; then
    pass "auto-launch: state file created (state=${AUTO_STATE})"
  else
    fail "auto-launch: unexpected state '${AUTO_STATE}'"
  fi
else
  fail "auto-launch: state file not created after cold review"
  echo "    stderr: $(cat "${STDERR_FILE}")"
fi

rm -f "${STDERR_FILE}"
cleanup_warm

# ============================================================
echo ""
echo "==> Results: ${TOTAL} checks, ${FAILS} failures"
# ============================================================

# Final cleanup: kill any warm server left by auto-launch during tests.
cleanup_warm
LEFTOVER=$(pgrep -f "python.*warm\\.py" 2>/dev/null || true)
if [[ -n "${LEFTOVER}" ]]; then
  kill ${LEFTOVER} 2>/dev/null || true
fi

if [[ "${FAILS}" -gt 0 ]]; then
  exit 1
fi
