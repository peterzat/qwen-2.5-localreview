#!/usr/bin/env python3
"""Benchmark harness: per-fixture prefill/decode TPS, peak VRAM, wall time.

Slow by design (loads a 14B+ model). Gated behind --full so it cannot
accidentally run from a fast pre-push hook. Reuses the same vllm.LLM()
construction as review.py via tests/_harness.py.

Usage:
    tests/bench.py --config baseline --full
"""

import argparse
import sys

from _harness import CONFIGS, get_config, run_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        choices=sorted(CONFIGS),
                        help="Named inference config from tests/_harness.py")
    parser.add_argument("--full", action="store_true",
                        help="Required: confirms you intend to run a slow GPU benchmark")
    args = parser.parse_args()

    if not args.full:
        print(
            "bench.py is a slow GPU benchmark. Pass --full to run.",
            file=sys.stderr,
        )
        return 2

    config = get_config(args.config)
    print(f"==> bench: config={config.name} model={config.model}", file=sys.stderr)
    print(f"    max_model_len={config.max_model_len} llm_kwargs={config.llm_kwargs}",
          file=sys.stderr)

    run = run_corpus(config)
    results = run.results

    # Markdown table to stdout for easy paste into commit messages.
    print(f"## Bench: {config.name}")
    print()
    print(f"- model: `{config.model}`")
    print(f"- max_model_len: {config.max_model_len}")
    print(f"- llm_kwargs: `{config.llm_kwargs}`")
    print(f"- post-load used VRAM: {run.post_load_used_vram_gb:.2f} GB")
    print(f"- peak used VRAM: {run.peak_used_vram_gb:.2f} GB")
    print()
    print("| fixture | prompt tok | gen tok | prefill TPS | decode TPS | used VRAM (GB) | wall (s) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        print(
            f"| {r.name} | {r.prefill_tokens} | {r.decode_tokens} | "
            f"{r.prefill_tps:.1f} | {r.decode_tps:.1f} | "
            f"{r.used_vram_gb:.2f} | {r.wall_seconds:.1f} |"
        )

    # Aggregate row.
    total_prompt = sum(r.prefill_tokens for r in results)
    total_gen = sum(r.decode_tokens for r in results)
    total_prefill_s = sum(r.prefill_seconds for r in results)
    total_decode_s = sum(r.decode_seconds for r in results)
    total_wall = sum(r.wall_seconds for r in results)
    agg_prefill_tps = total_prompt / total_prefill_s if total_prefill_s > 0 else 0.0
    agg_decode_tps = total_gen / total_decode_s if total_decode_s > 0 else 0.0
    print(
        f"| **total** | **{total_prompt}** | **{total_gen}** | "
        f"**{agg_prefill_tps:.1f}** | **{agg_decode_tps:.1f}** | "
        f"**{run.peak_used_vram_gb:.2f}** | **{total_wall:.1f}** |"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
