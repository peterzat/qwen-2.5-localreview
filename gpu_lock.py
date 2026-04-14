"""GPU lock and IPC paths for qwen-2.5-localreview.

Shared by review.py (client) and warm.py (server). Provides flock-based
GPU mutex and deterministic paths for the Unix domain socket.
"""

import fcntl
import json
import os
import time

LOCK_FILENAME = "qwen-localreview.lock"
SOCKET_FILENAME = "qwen-localreview.sock"
STATE_FILENAME = "qwen-localreview.state"


def _runtime_dir() -> str:
    """Per-user runtime directory, cleaned on reboot."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        return xdg
    return f"/tmp/qwen-localreview-{os.getuid()}"


def lock_path() -> str:
    return os.path.join(_runtime_dir(), LOCK_FILENAME)


def socket_path() -> str:
    return os.path.join(_runtime_dir(), SOCKET_FILENAME)


def state_path() -> str:
    return os.path.join(_runtime_dir(), STATE_FILENAME)


def write_state(state: str, pid: int = None) -> None:
    """Write warm server state atomically (tmp + rename)."""
    path = state_path()
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump({"state": state, "pid": pid or os.getpid(),
                    "timestamp": time.time()}, f)
    os.rename(tmp, path)


def read_state() -> dict | None:
    """Read warm server state. Returns dict or None if missing/corrupt."""
    try:
        with open(state_path()) as f:
            data = json.load(f)
        if "state" in data and "pid" in data:
            return data
        return None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire_gpu_lock(timeout: float = 270.0):
    """Acquire exclusive flock on the GPU lock file.

    Polls at 1-second intervals until the lock is acquired or timeout
    expires. Returns the open file object (caller holds it open to
    maintain the lock) or None on timeout.

    The lock is released automatically when the file object is closed
    or the process exits.
    """
    path = lock_path()
    # Ensure the runtime dir exists (needed for the /tmp fallback).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.monotonic() >= deadline:
                f.close()
                return None
            time.sleep(1.0)
