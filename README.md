# qwen-2.5-localreview

Local adversarial code reviewer for use with zat.env's `/codereview` skill. Runs Qwen2.5-Coder-14B-Instruct-AWQ via vLLM offline inference on a local GPU. No server, no network calls, no API costs.

## Prerequisites

- NVIDIA GPU with >= 20GB VRAM (designed for RTX 4000 SFF Ada)
- CUDA drivers and toolkit installed
- Python 3.10+
- ~16GB free disk space (8GB venv + 8GB model in HF cache)

## Setup

```bash
cd ~/src/qwen-2.5-localreview
./setup.sh
```

`setup.sh` is idempotent. It creates the venv, installs vLLM, downloads the model to the shared HuggingFace cache, and adds config entries to `~/.config/claude-reviewers/.env`. Re-run after `git pull` if the script changed (new vLLM version, new model).

## Architecture

```
review-external.sh (zat.env)
  call_openai()  --> curl --> OpenAI API       |
  call_google()  --> curl --> Gemini API       |- parallel
  call_local()   --> python review.py  < NEW   |
                        |
                   vLLM LLM class (in-process)
                   load model --> infer --> print --> exit
                   no server, no TCP/IP
                   VRAM freed on exit
```

Each invocation loads the model (~30-60s), runs inference (~60-120s), then exits and frees all GPU resources. No persistent processes, no reserved VRAM between reviews. The OS page cache keeps model files in RAM (64GB DDR4) so repeat invocations skip most disk I/O.

## Consumer Integration Guide

This section is for zat.env and any project using `review-external.sh`.

**Enabling the local reviewer.** See `integration/integration-guide.md` for the full list of changes needed in `review-external.sh` and `~/.config/claude-reviewers/.env`. The `.env` entries are managed by zat.env's install process, not by this project.

**Behavior.** The reviewer runs in parallel with cloud providers. It does not block or slow down OpenAI/Google reviews. If the GPU is busy or vLLM OOMs, the review completes without local findings (fail-open).

**Finding tags.** The `(qwen)` tag on findings distinguishes them from `(openai)` and `(google)` findings. The 14B model produces more false positives than cloud models; treat `(qwen)` findings as a second opinion, not authoritative.

**Testing in isolation.**

```bash
cd ~/src/qwen-2.5-localreview
echo "You are a code reviewer." > /tmp/test-sys
echo "Review this: eval(input())" > /tmp/test-input
.venv/bin/python review.py --system /tmp/test-sys --input /tmp/test-input
```

**Updating.** `cd ~/src/qwen-2.5-localreview && git pull`. If `setup.sh` changed, re-run it. The consumer config does not change across updates since it points to stable file paths (`review.py`, `.venv`).

## Inference configuration

The default inference path is **14B AWQ INT4 weights with FP8 KV cache
(`kv_cache_dtype="fp8_e4m3"`)**, which uses Ada's 4th-generation tensor
cores for the attention KV path via vLLM's FlashInfer backend. This is
the Stage 1 winning configuration from the
`abstract-yawning-raven` inference experiments (commit `b47ae08`):
+36% prefill TPS, +58% decode TPS, ~0.9 GB freed VRAM, and at least
equal review quality on every fixture vs the pre-Stage-1 baseline.

The legacy path (AWQ INT4 + FP16 KV cache, no `kv_cache_dtype`) is
available behind an environment variable:

```bash
LOCAL_INFERENCE_MODE=legacy .venv/bin/python review.py ...
```

Use this only for one-off comparison or as an escape hatch if a future
regression is suspected. Any other value of `LOCAL_INFERENCE_MODE`
(including unset, `default`, or a typo) selects the FP8 KV default --
silent fallback to legacy is intentionally not supported. The toggle
is exercised by Test 7 in `tests/test-review.sh`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_MODEL` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | HuggingFace model ID |
| `LOCAL_MAX_MODEL_LEN` | `32768` | Max context length in tokens |
| `LOCAL_INFERENCE_MODE` | (unset, behaves as `default`) | `legacy` restores AWQ INT4 + FP16 KV exactly. Any other value selects the FP8 KV default. |
| `LOCAL_REVIEW_SCRIPT` | (none) | Absolute path to `review.py` |
| `LOCAL_REVIEW_VENV` | (none) | Absolute path to project `.venv` |
