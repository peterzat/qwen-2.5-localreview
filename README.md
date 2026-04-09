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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_MODEL` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | HuggingFace model ID |
| `LOCAL_MAX_MODEL_LEN` | `32768` | Max context length in tokens |
| `LOCAL_REVIEW_SCRIPT` | (none) | Absolute path to `review.py` |
| `LOCAL_REVIEW_VENV` | (none) | Absolute path to project `.venv` |
