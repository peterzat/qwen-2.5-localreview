#!/usr/bin/env python3
"""Local code review via vLLM offline inference. No server, no network."""

import argparse
import os
import sys
import time

# Suppress vLLM's verbose loading output before importing.
# Only structured [qwen-2.5-localreview] status lines should reach stderr.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

# vLLM pulls in torch which is noisy about CUDA.
import logging
logging.disable(logging.WARNING)

TAG = "qwen-2.5-localreview"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
DEFAULT_MAX_MODEL_LEN = 32768
CHARS_PER_TOKEN = 3.5
OUTPUT_RESERVE_TOKENS = 4096
SYSTEM_PROMPT_RESERVE_TOKENS = 512


def estimate_max_input_chars(max_model_len: int) -> int:
    """Conservative estimate of max input characters that fit in context."""
    available_tokens = max_model_len - OUTPUT_RESERVE_TOKENS - SYSTEM_PROMPT_RESERVE_TOKENS
    return int(available_tokens * CHARS_PER_TOKEN)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local code review via vLLM")
    parser.add_argument("--system", required=True, help="Path to system prompt file")
    parser.add_argument("--input", required=True, help="Path to user input file")
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

    model = os.environ.get("LOCAL_MODEL", DEFAULT_MODEL)
    max_model_len = int(os.environ.get("LOCAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))

    # Context limit guard.
    max_chars = estimate_max_input_chars(max_model_len)
    if len(user_input) > max_chars:
        print(
            f"[{TAG}] Input too large ({len(user_input)} chars > {max_chars} limit), truncating",
            file=sys.stderr,
        )
        user_input = user_input[:max_chars] + (
            f"\n\n[TRUNCATED: input exceeded {max_chars} chars]"
        )

    # Load model and run inference.
    start = time.monotonic()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.95,
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
        print(f"[qwen-2.5-localreview] Fatal error: {e}", file=sys.stderr)
        sys.exit(0)  # fail-open
