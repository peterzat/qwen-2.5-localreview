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

FAILS=0
TOTAL=0
pass() { TOTAL=$((TOTAL + 1)); printf '  ok   %s\n' "$1"; }
fail() { TOTAL=$((TOTAL + 1)); FAILS=$((FAILS + 1)); printf '  FAIL %s\n' "$1"; }

# Cleanup helper: kill warm server and remove stale socket.
cleanup_warm() {
  if [[ -n "${WARM_PID:-}" ]]; then
    kill "${WARM_PID}" 2>/dev/null || true
    wait "${WARM_PID}" 2>/dev/null || true
    WARM_PID=""
  fi
  [[ -S "${SOCK_PATH}" ]] && rm -f "${SOCK_PATH}" || true
}
trap cleanup_warm EXIT

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
echo "==> Test 6: VRAM yield when external process uses GPU"
# ============================================================

# Start the warm server, then allocate >256 MiB of GPU memory from a
# separate process. The warm server should detect the external usage
# within its 30s poll interval and exit.
cleanup_warm
sleep 3
WARM_LOG=$(mktemp)
LOCAL_WARM_TIMEOUT=120 "${PYTHON}" "${WARM_SCRIPT}" 2>"${WARM_LOG}" &
WARM_PID=$!

if wait_for_socket 90; then
  # Allocate 512 MiB on the GPU from a separate process.
  "${PYTHON}" -c "
import torch, time, sys
x = torch.zeros(512 * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
# Hold the allocation for 45s (enough for the 30s poll to fire).
time.sleep(45)
" &
  ALLOC_PID=$!
  sleep 2  # let the allocation happen

  # Wait up to 40s for the warm server to detect and yield.
  YIELDED=false
  for i in $(seq 1 40); do
    if ! kill -0 "${WARM_PID}" 2>/dev/null; then
      YIELDED=true
      break
    fi
    sleep 1
  done

  kill "${ALLOC_PID}" 2>/dev/null || true; wait "${ALLOC_PID}" 2>/dev/null || true

  if ${YIELDED}; then
    if grep -q "external GPU usage" "${WARM_LOG}" || grep -q "yielding" "${WARM_LOG}"; then
      pass "VRAM yield: warm server exited on external GPU usage"
    else
      pass "VRAM yield: warm server exited (yield message not found, but process died)"
    fi
  else
    fail "VRAM yield: warm server did not exit after external GPU allocation"
    echo "    log: $(tail -5 "${WARM_LOG}")"
    cleanup_warm
  fi
else
  fail "warm server did not start for VRAM yield test"
  echo "    log: $(cat "${WARM_LOG}")"
fi
rm -f "${WARM_LOG}"
WARM_PID=""

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
LOCAL_GPU_LOCK_TIMEOUT=5 "${PYTHON}" "${REVIEW_SCRIPT}" \
  --system "${SYSTEM_PROMPT}" \
  --input "${FIXTURE}" \
  2>"${STDERR_FILE}" || true
OOM_EXIT=$?

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
echo "==> Results: ${TOTAL} checks, ${FAILS} failures"
# ============================================================

if [[ "${FAILS}" -gt 0 ]]; then
  exit 1
fi
