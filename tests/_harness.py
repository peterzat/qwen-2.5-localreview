"""Shared machinery for tests/bench.py and tests/eval.py.

Defines the named inference configurations for the staged Ada-aware
inference experiments and the fixture-loading helpers used by both
harnesses. Stays consistent with review.py's LLM() construction so that
"baseline" measurements actually reflect the production code path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Stay quiet like review.py.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

REPO_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_DIR / "tests" / "fixtures"
DIFFS_DIR = FIXTURES_DIR / "diffs"
RESULTS_DIR = REPO_DIR / "tests" / "results"
SYSTEM_PROMPT_PATH = FIXTURES_DIR / "system.txt"


@dataclass
class InferenceConfig:
    """A named LLM() configuration plus sampling params.

    `llm_kwargs` is passed verbatim to vllm.LLM(). `sampling_kwargs` is
    passed verbatim to vllm.SamplingParams(). Keep these in lockstep with
    review.py for the baseline config so measurements are honest.
    """
    name: str
    model: str
    max_model_len: int
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    sampling_kwargs: dict[str, Any] = field(default_factory=dict)


# Sampling kwargs match review.py:127-133 (commit 2f56c64).
DEFAULT_SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "max_tokens": 4096,
}


CONFIGS: dict[str, InferenceConfig] = {
    # Mirrors review.py:120-125 exactly. This is the reference point for
    # every later stage; do not edit unless review.py changes.
    "baseline": InferenceConfig(
        name="baseline",
        model="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        max_model_len=32768,
        llm_kwargs={
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
        },
        sampling_kwargs=DEFAULT_SAMPLING,
    ),
    # Stage 1: only change is FP8 KV cache. Everything else identical.
    "stage1-fp8kv": InferenceConfig(
        name="stage1-fp8kv",
        model="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        max_model_len=32768,
        llm_kwargs={
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "kv_cache_dtype": "fp8_e4m3",
        },
        sampling_kwargs=DEFAULT_SAMPLING,
    ),
    # Stage 3: 32B AWQ at reduced context, on top of Stage 1's FP8 KV.
    "stage3-32b-16k": InferenceConfig(
        name="stage3-32b-16k",
        model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        max_model_len=16384,
        llm_kwargs={
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "kv_cache_dtype": "fp8_e4m3",
        },
        sampling_kwargs=DEFAULT_SAMPLING,
    ),
    "stage3-32b-8k": InferenceConfig(
        name="stage3-32b-8k",
        model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        max_model_len=8192,
        llm_kwargs={
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "kv_cache_dtype": "fp8_e4m3",
        },
        sampling_kwargs=DEFAULT_SAMPLING,
    ),
}


def load_fixtures() -> list[tuple[str, str]]:
    """Return [(fixture_name, diff_text), ...] sorted by filename."""
    paths = sorted(DIFFS_DIR.glob("*.patch"))
    if not paths:
        raise RuntimeError(f"No .patch fixtures found in {DIFFS_DIR}")
    return [(p.stem, p.read_text()) for p in paths]


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def get_config(name: str) -> InferenceConfig:
    if name not in CONFIGS:
        valid = ", ".join(sorted(CONFIGS))
        raise SystemExit(f"Unknown config '{name}'. Valid: {valid}")
    return CONFIGS[name]


def build_llm(config: InferenceConfig):
    """Construct a vllm.LLM() for the given config. Imports vllm lazily."""
    from vllm import LLM
    return LLM(
        model=config.model,
        max_model_len=config.max_model_len,
        **config.llm_kwargs,
    )


def build_sampling(config: InferenceConfig):
    from vllm import SamplingParams
    return SamplingParams(**config.sampling_kwargs)


@dataclass
class FixtureResult:
    name: str
    prefill_tokens: int
    decode_tokens: int
    prefill_seconds: float
    decode_seconds: float
    wall_seconds: float
    used_vram_bytes: int
    output_text: str

    @property
    def prefill_tps(self) -> float:
        return self.prefill_tokens / self.prefill_seconds if self.prefill_seconds > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0

    @property
    def used_vram_gb(self) -> float:
        return self.used_vram_bytes / (1024 ** 3)


@dataclass
class CorpusRun:
    config_name: str
    results: list[FixtureResult]
    # VRAM after LLM() is built but before any fixture runs. Captures
    # weights + pre-allocated KV pool, which is the dominant difference
    # between configurations.
    post_load_used_vram_bytes: int
    # Highest used VRAM observed across all measurement points. Includes
    # post-load and post-fixture samples; this is the upper-bound figure
    # that matters for "does it fit in 20 GB."
    peak_used_vram_bytes: int

    @property
    def post_load_used_vram_gb(self) -> float:
        return self.post_load_used_vram_bytes / (1024 ** 3)

    @property
    def peak_used_vram_gb(self) -> float:
        return self.peak_used_vram_bytes / (1024 ** 3)


def _used_vram_bytes() -> int:
    """Device-level used VRAM via torch.cuda.mem_get_info().

    vLLM's V1 engine allocates outside PyTorch's caching allocator, so
    torch.cuda.max_memory_allocated() reports 0 here. mem_get_info() reads
    free/total directly from the driver, which catches all allocations
    regardless of source.
    """
    import torch
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return total - free


def run_corpus(config: InferenceConfig) -> CorpusRun:
    """Load model once, run all fixtures, return timing + output + VRAM.

    vLLM 0.19's V1 engine does not populate RequestOutput.metrics for
    synchronous llm.chat() calls, so we cannot read first-token-time
    directly. Instead each fixture is run twice:

      1. max_tokens=1: wall time measures prefill (+ one decode step).
      2. full max_tokens: wall time measures end-to-end. Decode time is
         end-to-end minus the prefill measurement. Decode TPS is computed
         over (gen_tokens - 1) since the first token came from the prefill
         run's accounting.

    Doubles the per-fixture work but gives honest, separable numbers and
    keeps the spec criterion (per-diff prefill TPS and decode TPS) real.
    """
    from vllm import RequestOutput, SamplingParams

    fixtures = load_fixtures()
    system_prompt = load_system_prompt()

    llm = build_llm(config)
    full_params = build_sampling(config)
    # Prefill probe: identical sampling, just one token. Reuse all the
    # other params so we are not measuring sampler overhead differences.
    prefill_probe_kwargs = dict(config.sampling_kwargs)
    prefill_probe_kwargs["max_tokens"] = 1
    prefill_params = SamplingParams(**prefill_probe_kwargs)

    post_load_vram = _used_vram_bytes()
    peak_vram = post_load_vram

    # Untimed warmup so the first measured fixture does not eat the
    # cost of any first-call lazy initialization. Reuse the first
    # fixture's content; the prefix cache is reset right after.
    warmup_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fixtures[0][1]},
    ]
    llm.chat([warmup_messages], sampling_params=prefill_params)
    llm.reset_prefix_cache()

    results: list[FixtureResult] = []
    for fname, diff in fixtures:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": diff},
        ]

        # Probe: prefill + 1 token. Reset prefix cache so this is a cold
        # prefill (otherwise vLLM's prefix cache from the warmup or a
        # prior fixture can short-circuit it).
        llm.reset_prefix_cache()
        probe_start = time.monotonic()
        llm.chat([messages], sampling_params=prefill_params)
        prefill_seconds = time.monotonic() - probe_start

        # Full generation. Reset again so we measure a fresh prefill
        # against the same baseline as the probe.
        llm.reset_prefix_cache()
        wall_start = time.monotonic()
        outputs: list[RequestOutput] = llm.chat([messages], sampling_params=full_params)
        wall_seconds = time.monotonic() - wall_start

        out = outputs[0]
        gen = out.outputs[0]
        prompt_tokens = len(out.prompt_token_ids)
        completion_tokens = len(gen.token_ids)

        decode_seconds = max(wall_seconds - prefill_seconds, 1e-6)
        # decode_tokens excludes the first token, which was part of prefill.
        decode_token_count = max(completion_tokens - 1, 0)

        used_vram = _used_vram_bytes()
        if used_vram > peak_vram:
            peak_vram = used_vram

        results.append(FixtureResult(
            name=fname,
            prefill_tokens=prompt_tokens,
            decode_tokens=decode_token_count,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            wall_seconds=wall_seconds,
            used_vram_bytes=used_vram,
            output_text=gen.text,
        ))

    return CorpusRun(
        config_name=config.name,
        results=results,
        post_load_used_vram_bytes=post_load_vram,
        peak_used_vram_bytes=peak_vram,
    )
