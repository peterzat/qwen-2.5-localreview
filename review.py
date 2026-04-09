#!/usr/bin/env python3
"""Local code review via vLLM offline inference. No server, no network."""

import argparse
import os
import sys
import time

# Suppress vLLM's verbose loading output before importing.
# Only structured [qwen] status lines should reach stderr.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

# vLLM pulls in torch which is noisy about CUDA.
import logging
logging.disable(logging.WARNING)

TAG = "qwen"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
DEFAULT_MAX_MODEL_LEN = 32768
OUTPUT_RESERVE_TOKENS = 4096

# Module-level sentinels for the fatal error handler. main() writes these
# before heavy work so the outer except can emit a structured status line.
_model = DEFAULT_MODEL
_start = None


def truncate_to_fit(tokenizer, system_prompt, user_input, max_model_len):
    """Truncate user_input so system_prompt + user_input + output reserve fits.

    Uses the model's tokenizer for accurate token counting. Returns the
    (possibly truncated) user_input and whether truncation occurred.
    """
    system_tokens = len(tokenizer.encode(system_prompt))
    available = max_model_len - system_tokens - OUTPUT_RESERVE_TOKENS
    if available <= 0:
        return "", True

    user_tokens = tokenizer.encode(user_input)
    if len(user_tokens) <= available:
        return user_input, False

    # Truncate at the token level, then decode back to text.
    truncated_tokens = user_tokens[:available]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text, True


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
    max_model_len = int(os.environ.get("LOCAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))

    # Context limit guard: tokenizer-based, accounts for system prompt.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    user_input, was_truncated = truncate_to_fit(
        tokenizer, system_prompt, user_input, max_model_len,
    )
    if was_truncated:
        user_tokens = len(tokenizer.encode(user_input))
        system_tokens = len(tokenizer.encode(system_prompt))
        print(
            f"[{TAG}] Input truncated to fit context window"
            f" (system: {system_tokens}, user: {user_tokens},"
            f" reserve: {OUTPUT_RESERVE_TOKENS}, max: {max_model_len})",
            file=sys.stderr,
        )
    if not user_input:
        print(f"[{TAG}] Input empty after truncation, skipping", file=sys.stderr)
        return 0

    if args.dry_run:
        user_tokens = len(tokenizer.encode(user_input))
        print(f"[{TAG}] dry-run: input valid, {user_tokens} tokens", file=sys.stderr)
        return 0

    # Load model and run inference.
    start = time.monotonic()
    _start = start

    # VRAM preflight: warn early if GPU memory looks too low.
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            if free_gb < 10.0:
                print(
                    f"[{TAG}] Warning: {free_gb:.1f}GB VRAM free, need ~10GB."
                    f" May OOM. Try: LOCAL_MAX_MODEL_LEN=8192",
                    file=sys.stderr,
                )
    except Exception:
        pass

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )

    params = SamplingParams(temperature=0.2, max_tokens=4096)
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
