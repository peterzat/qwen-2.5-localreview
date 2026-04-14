#!/usr/bin/env python3
"""Keep-warm server: holds the loaded model in VRAM between reviews.

Listens on a Unix domain socket for review requests, avoiding the
30-60s model load cost on each invocation. Exits automatically after
an idle timeout (default 15 min) or when external GPU usage is detected.

Usage:
    .venv/bin/python warm.py              # foreground, default 15 min idle
    LOCAL_WARM_TIMEOUT=60 .venv/bin/python warm.py   # 60s idle for testing
"""

import json
import os
import select
import signal
import socket
import sys
import time

# Reuse review.py's constants and truncation logic.
from review import (
    TAG,
    DEFAULT_MODEL,
    DEFAULT_MAX_MODEL_LEN,
    _output_reserve,
    truncate_to_fit,
)
from gpu_lock import acquire_gpu_lock, socket_path, state_path, write_state

# Suppress vLLM noise, same as review.py.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

# Ensure .venv/bin is on PATH for FlashInfer's ninja JIT, same as review.py.
_venv_bin = os.path.dirname(sys.executable)
if _venv_bin and _venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

import logging
logging.disable(logging.WARNING)

IDLE_TIMEOUT = int(os.environ.get("LOCAL_WARM_TIMEOUT", "900"))
VRAM_YIELD_THRESHOLD = 256 * 1024 * 1024  # 256 MiB
VRAM_POLL_INTERVAL = 30  # seconds
GPU_MEMORY_UTILIZATION = 0.90
MAX_REQUEST_SIZE = 128 * 1024 * 1024  # 128 MiB sanity guard


def _log(msg: str) -> None:
    print(f"[{TAG}] warm: {msg}", file=sys.stderr, flush=True)


def _used_vram_bytes() -> int:
    """Device-level used VRAM via torch.cuda.mem_get_info()."""
    import torch
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return total - free


class WarmServer:
    def __init__(self) -> None:
        self.model = os.environ.get("LOCAL_MODEL", DEFAULT_MODEL)
        self.max_model_len = int(
            os.environ.get("LOCAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
        )
        self.output_reserve = _output_reserve(self.max_model_len)
        self.sock_path = socket_path()
        self.sock = None
        self.lock_file = None
        self.llm = None
        self.tokenizer = None
        self.baseline_vram = 0

    def start(self) -> None:
        # Signal intent before locking. review.py reads this to decide
        # whether to wait for startup vs. falling to cold path.
        write_state("starting")

        # Acquire GPU flock (non-blocking).
        self.lock_file = acquire_gpu_lock(timeout=0.0)
        if self.lock_file is None:
            write_state("failed")
            _log("GPU flock held by another process, cannot start")
            sys.exit(1)

        _log(f"loading {self.model}...")
        from vllm import LLM
        self.llm = LLM(
            model=self.model,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=True,
            kv_cache_dtype="fp8_e4m3",
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.baseline_vram = _used_vram_bytes()

        # Create Unix domain socket.
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)  # stale from prior crash; we hold the flock
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.sock_path)
        self.sock.listen(2)

        write_state("ready")
        _log(f"ready on {self.sock_path} "
             f"(idle={IDLE_TIMEOUT}s, yield={VRAM_YIELD_THRESHOLD // (1024*1024)}MiB)")

    def serve(self) -> None:
        idle_deadline = time.monotonic() + IDLE_TIMEOUT
        last_vram_check = time.monotonic()

        while True:
            remaining = idle_deadline - time.monotonic()
            if remaining <= 0:
                _log(f"idle timeout ({IDLE_TIMEOUT}s), exiting")
                break

            # VRAM yield check.
            now = time.monotonic()
            if now - last_vram_check >= VRAM_POLL_INTERVAL:
                if self._check_vram():
                    _log("external GPU usage detected, yielding")
                    break
                last_vram_check = now

            select_timeout = min(
                remaining,
                VRAM_POLL_INTERVAL - (now - last_vram_check),
            )
            ready, _, _ = select.select([self.sock], [], [], max(0.1, select_timeout))

            if ready:
                try:
                    conn, _ = self.sock.accept()
                    self._handle_request(conn)
                except Exception as e:
                    _log(f"accept error: {e}")
                idle_deadline = time.monotonic() + IDLE_TIMEOUT

    def _handle_request(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(10.0)  # 10s to receive the full request
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk
                if len(data) > MAX_REQUEST_SIZE:
                    self._send_error(conn, "request too large")
                    return

            request = json.loads(data.split(b"\n", 1)[0])
            system_prompt = request["system_prompt"]
            user_input = request["user_input"]

            stderr_lines = []

            # Truncation (same logic as review.py).
            user_input, was_truncated = truncate_to_fit(
                self.tokenizer, system_prompt, user_input,
                self.max_model_len, self.output_reserve,
            )
            if was_truncated:
                user_tokens = len(self.tokenizer.encode(user_input))
                system_tokens = len(self.tokenizer.encode(system_prompt))
                stderr_lines.append(
                    f"[{TAG}] Input truncated to fit context window"
                    f" (system: {system_tokens}, user: {user_tokens},"
                    f" reserve: {self.output_reserve},"
                    f" max: {self.max_model_len})"
                )
            if not user_input:
                stderr_lines.append(
                    f"[{TAG}] Input empty after truncation, skipping"
                )
                self._send_response(conn, {
                    "status": "ok",
                    "output": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed": 0.0,
                    "stderr_lines": stderr_lines,
                })
                return

            # Inference.
            start = time.monotonic()
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=0.2,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                max_tokens=self.output_reserve,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
            outputs = self.llm.chat([messages], sampling_params=params)
            elapsed = time.monotonic() - start

            if not outputs or not outputs[0].outputs:
                stderr_lines.append(f"[{TAG}] No output from model")
                self._send_response(conn, {
                    "status": "ok",
                    "output": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "elapsed": elapsed,
                    "stderr_lines": stderr_lines,
                })
                return

            result = outputs[0].outputs[0]
            prompt_tokens = len(outputs[0].prompt_token_ids)
            completion_tokens = len(result.token_ids)

            self._send_response(conn, {
                "status": "ok",
                "output": result.text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed": elapsed,
                "stderr_lines": stderr_lines,
            })
            _log(f"served: {prompt_tokens} in / {completion_tokens} out / {elapsed:.1f}s")

        except Exception as e:
            _log(f"request error: {e}")
            self._send_error(conn, str(e))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _send_response(self, conn: socket.socket, response: dict) -> None:
        try:
            conn.settimeout(10.0)
            conn.sendall(json.dumps(response).encode() + b"\n")
        except Exception:
            pass

    def _send_error(self, conn: socket.socket, message: str) -> None:
        self._send_response(conn, {
            "status": "error",
            "message": message,
            "stderr_lines": [
                f"[{TAG}] {self.model} -- error: {message} -- 0 in / 0 out -- 0s"
            ],
        })

    def _check_vram(self) -> bool:
        """Returns True if external VRAM usage exceeds the yield threshold."""
        current = _used_vram_bytes()
        external_delta = current - self.baseline_vram
        return external_delta > VRAM_YIELD_THRESHOLD

    def shutdown(self) -> None:
        _log("shutting down")
        try:
            write_state("stopped")
        except Exception:
            pass
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.sock_path and os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except Exception:
                pass
        try:
            p = state_path()
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass
        if self.lock_file:
            try:
                self.lock_file.close()
            except Exception:
                pass


def main() -> int:
    server = WarmServer()

    # Graceful shutdown on signals.
    def _signal_handler(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        server.start()
        server.serve()
    except SystemExit:
        pass
    except Exception as e:
        _log(f"fatal: {e}")
    finally:
        server.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
