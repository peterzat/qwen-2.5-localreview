#!/usr/bin/env python3
"""Local code review via vLLM offline inference. No server, no network."""

import argparse
import os
import sys
import time

# Suppress vLLM's verbose loading output before importing.
# Only structured [qwen] status lines should reach stderr.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

# vLLM's FlashInfer attention backend (selected when kv_cache_dtype is
# fp8_e4m3, set at the LLM() construction below) JIT-compiles kernels
# via ninja on first use. ninja is pip-installed at .venv/bin/ninja by
# the vLLM dependency tree, but invoking .venv/bin/python directly does
# not put .venv/bin on PATH the way `source .venv/bin/activate` would,
# so the inner subprocess.run("ninja", ...) inside FlashInfer fails
# with FileNotFoundError. Prepend the venv's bin dir to PATH so any
# tool installed there is reachable from spawned subprocesses.
_venv_bin = os.path.dirname(sys.executable)
if _venv_bin and _venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

# vLLM pulls in torch which is noisy about CUDA.
import logging
logging.disable(logging.WARNING)

TAG = "qwen"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
DEFAULT_MAX_MODEL_LEN = 32768
def _output_reserve(max_model_len: int) -> int:
    """Adaptive output reserve: up to 25% of context, clamped [256, 4096]."""
    return min(4096, max(256, max_model_len // 4))

# Module-level sentinels for the fatal error handler. main() writes these
# before heavy work so the outer except can emit a structured status line.
_model = DEFAULT_MODEL
_start = None


def truncate_to_fit(tokenizer, system_prompt, user_input, max_model_len, output_reserve):
    """Truncate user_input so system_prompt + user_input + output reserve fits.

    Uses the model's tokenizer for accurate token counting. Returns the
    (possibly truncated) user_input and whether truncation occurred.
    """
    system_tokens = len(tokenizer.encode(system_prompt))
    available = max_model_len - system_tokens - output_reserve
    if available <= 0:
        return "", True

    user_tokens = tokenizer.encode(user_input)
    if len(user_tokens) <= available:
        return user_input, False

    # Truncate at the token level, then decode back to text.
    truncated_tokens = user_tokens[:available]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text, True


def _try_warm_path(system_prompt: str, user_input: str):
    """Attempt inference via the warm server. Returns None on any failure."""
    import json
    import socket as sock_mod
    from gpu_lock import socket_path

    sock_path = socket_path()
    if not os.path.exists(sock_path):
        return None

    try:
        conn = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        conn.settimeout(0.5)  # 500ms connect timeout
        conn.connect(sock_path)

        request = json.dumps({
            "system_prompt": system_prompt,
            "user_input": user_input,
        }).encode() + b"\n"
        conn.settimeout(120.0)  # 120s for inference
        conn.sendall(request)

        data = b""
        while b"\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                return None
            data += chunk
        conn.close()

        response = json.loads(data.split(b"\n", 1)[0])
        if response.get("status") != "ok":
            return None

        return (
            response.get("output", ""),
            response.get("prompt_tokens", 0),
            response.get("completion_tokens", 0),
            response.get("elapsed", 0.0),
            response.get("stderr_lines", []),
        )
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Local code review via vLLM")
    parser.add_argument("--system", required=True, help="Path to system prompt file")
    parser.add_argument("--input", required=True, help="Path to user input file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and exit before loading model")
    args = parser.parse_args()

    try:
        with open(args.system) as f:
            system_prompt = f.read()
        with open(args.input) as f:
            user_input = f.read()
    except (OSError, IOError) as e:
        print(f"[{TAG}] Failed to read input files: {e}", file=sys.stderr)
        return 0

    if not user_input.strip():
        print(f"[{TAG}] Empty input, nothing to review", file=sys.stderr)
        return 0

    global _model, _start
    model = os.environ.get("LOCAL_MODEL", DEFAULT_MODEL)
    _model = model
    raw_max_len = os.environ.get("LOCAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
    try:
        max_model_len = int(raw_max_len)
    except ValueError:
        print(f"[{TAG}] LOCAL_MAX_MODEL_LEN must be an integer, got: {raw_max_len!r}", file=sys.stderr)
        return 0
    if max_model_len <= 0:
        print(f"[{TAG}] LOCAL_MAX_MODEL_LEN must be positive, got: {max_model_len}", file=sys.stderr)
        return 0

    output_reserve = _output_reserve(max_model_len)

    _warm_attempted = False
    # Warm path: try the warm server before loading the tokenizer or model.
    # The server handles tokenization, truncation, and inference internally.
    warm_result = _try_warm_path(system_prompt, user_input)
    if warm_result is not None:
        output_text, prompt_tokens, completion_tokens, elapsed, stderr_lines = warm_result
        for line in stderr_lines:
            print(line, file=sys.stderr)
        print(
            f"[{TAG}] {model} -- {prompt_tokens} in / {completion_tokens} out"
            f" -- {elapsed:.0f}s (warm)",
            file=sys.stderr,
        )
        if output_text:
            print(output_text)
        return 0
    _warm_attempted = True

    # Context limit guard: tokenizer-based, accounts for system prompt.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    user_input, was_truncated = truncate_to_fit(
        tokenizer, system_prompt, user_input, max_model_len, output_reserve,
    )
    if was_truncated:
        user_tokens = len(tokenizer.encode(user_input))
        system_tokens = len(tokenizer.encode(system_prompt))
        print(
            f"[{TAG}] Input truncated to fit context window"
            f" (system: {system_tokens}, user: {user_tokens},"
            f" reserve: {output_reserve}, max: {max_model_len})",
            file=sys.stderr,
        )
    if not user_input:
        print(f"[{TAG}] Input empty after truncation, skipping", file=sys.stderr)
        return 0

    if args.dry_run:
        user_tokens = len(tokenizer.encode(user_input))
        print(f"[{TAG}] dry-run: input valid, {user_tokens} tokens", file=sys.stderr)
        return 0

    # GPU mutex: serialize concurrent invocations via flock.
    # Use a short timeout if the warm path was attempted (the warm server
    # is likely holding the flock and stuck), preserving timeout budget.
    from gpu_lock import acquire_gpu_lock
    default_lock_timeout = "30" if _warm_attempted else "270"
    lock_timeout = float(os.environ.get("LOCAL_GPU_LOCK_TIMEOUT", default_lock_timeout))
    lock_file = acquire_gpu_lock(timeout=lock_timeout)
    if lock_file is None:
        print(f"[{TAG}] GPU busy (another review running), skipping", file=sys.stderr)
        return 0

    # Load model and run inference.
    start = time.monotonic()
    _start = start

    # VRAM preflight: warn early if GPU memory looks too low for vLLM.
    #
    # vLLM's startup check is: free >= gpu_memory_utilization * total.
    # On the 20 GB Ada card with gpu_memory_utilization=0.90, that is
    # 0.90 * 19.55 = 17.6 GB. Rather than hardcoding a threshold, we
    # compute it from the same values vLLM uses, so the preflight
    # mirrors the real check regardless of GPU or utilization setting.
    #
    # Measured (2026-04-13, idle GPU, after torch CUDA init):
    #   total=20019 MiB, free=19853 MiB, needed=18017 MiB
    #   headroom at idle: ~1836 MiB (~1.8 GB)
    #   torch+CUDA context overhead: ~166 MiB
    GPU_MEMORY_UTILIZATION = 0.90  # must match LLM() below
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            needed_gb = GPU_MEMORY_UTILIZATION * total_bytes / (1024 ** 3)
            if free_gb < needed_gb:
                print(
                    f"[{TAG}] Warning: {free_gb:.1f}GB VRAM free,"
                    f" need {needed_gb:.1f}GB"
                    f" ({GPU_MEMORY_UTILIZATION:.0%} of"
                    f" {total_bytes / (1024**3):.1f}GB)."
                    f" May OOM. Try: LOCAL_MAX_MODEL_LEN=8192",
                    file=sys.stderr,
                )
    except Exception:
        pass

    from vllm import LLM, SamplingParams

    # 14B AWQ INT4 weights with FP8 KV cache via Ada's 4th-gen tensor
    # cores. Adopted as the default in commit c53d898 from the
    # abstract-yawning-raven inference experiments (Stage 1, b47ae08):
    # +36% prefill TPS, +58% decode TPS, ~0.9 GB freed VRAM vs the
    # pre-Stage-1 AWQ + FP16 KV path. enforce_eager=True is retained
    # from commit 8321af1 to avoid a CUDA-graph-induced OOM.
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        kv_cache_dtype="fp8_e4m3",
    )

    params = SamplingParams(
        temperature=0.2,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=output_reserve,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    outputs = llm.chat([messages], sampling_params=params)
    elapsed = time.monotonic() - start

    if not outputs or not outputs[0].outputs:
        print(f"[{TAG}] No output from model", file=sys.stderr)
        return 0

    result = outputs[0].outputs[0]
    output_text = result.text

    # Token counts and timing to stderr.
    prompt_tokens = len(outputs[0].prompt_token_ids)
    completion_tokens = len(result.token_ids)
    print(
        f"[{TAG}] {model} -- {prompt_tokens} in / {completion_tokens} out -- {elapsed:.0f}s",
        file=sys.stderr,
    )

    # Findings to stdout.
    print(output_text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        # argparse calls sys.exit(2) on bad args; normalize to fail-open.
        # Exit code 0 from main() passes through unchanged.
        if e.code == 0:
            sys.exit(0)
        print(
            f"[{TAG}] {_model} -- error: bad arguments -- 0 in / 0 out -- 0s",
            file=sys.stderr,
        )
        sys.exit(0)  # fail-open
    except Exception as e:
        elapsed = time.monotonic() - _start if _start is not None else 0
        # Detect CUDA OOM for an actionable message.
        is_oom = "CUDA out of memory" in str(e) or "OutOfMemoryError" in type(e).__name__
        if not is_oom:
            try:
                import torch
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
            except Exception:
                pass
        if is_oom:
            msg = "CUDA OOM. Try: LOCAL_MAX_MODEL_LEN=16384"
        else:
            msg = str(e)
        print(
            f"[{TAG}] {_model} -- error: {msg} -- 0 in / 0 out -- {elapsed:.0f}s",
            file=sys.stderr,
        )
        sys.exit(0)  # fail-open
