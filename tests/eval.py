#!/usr/bin/env python3
"""Eval harness: run the fixture corpus and write raw outputs to disk.

Writes one markdown file per config to tests/results/<config-name>.md so
that adopted vs candidate configs can be diffed side by side. Quality is
judged manually -- there is no automated scoring.

Usage:
    tests/eval.py --config baseline --full
"""

import argparse
import sys

from _harness import CONFIGS, RESULTS_DIR, get_config, run_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        choices=sorted(CONFIGS),
                        help="Named inference config from tests/_harness.py")
    parser.add_argument("--full", action="store_true",
                        help="Required: confirms you intend to run a slow GPU eval")
    args = parser.parse_args()

    if not args.full:
        print(
            "eval.py is a slow GPU eval. Pass --full to run.",
            file=sys.stderr,
        )
        return 2

    config = get_config(args.config)
    print(f"==> eval: config={config.name} model={config.model}", file=sys.stderr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{config.name}.md"

    run = run_corpus(config)

    lines: list[str] = []
    lines.append(f"# Eval results: {config.name}")
    lines.append("")
    lines.append(f"- model: `{config.model}`")
    lines.append(f"- max_model_len: {config.max_model_len}")
    lines.append(f"- llm_kwargs: `{config.llm_kwargs}`")
    lines.append(f"- sampling_kwargs: `{config.sampling_kwargs}`")
    lines.append(f"- post-load used VRAM: {run.post_load_used_vram_gb:.2f} GB")
    lines.append(f"- peak used VRAM: {run.peak_used_vram_gb:.2f} GB")
    lines.append("")
    for r in run.results:
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append(
            f"_{r.prefill_tokens} prompt / {r.decode_tokens} gen tokens, "
            f"{r.wall_seconds:.1f}s wall, used {r.used_vram_gb:.2f} GB_"
        )
        lines.append("")
        lines.append("```")
        lines.append(r.output_text.rstrip())
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"==> wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
