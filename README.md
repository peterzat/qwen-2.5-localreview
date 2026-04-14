# qwen-2.5-localreview

Local adversarial code reviewer for use with [zat.env](https://github.com/peterzat/zat.env)'s `/codereview` skill. Runs Qwen2.5-Coder-14B-Instruct-AWQ via vLLM offline inference on a local GPU. No server, no network calls, no API costs.

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

`setup.sh` is idempotent. It creates the venv, installs vLLM, downloads the model to the shared HuggingFace cache, and configures git hooks. Re-run after `git pull` if the script changed (new vLLM version, new model).

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

Without the warm server, each invocation loads the model (~30s), runs inference (~1-7s per diff), then exits and frees all GPU resources. The OS page cache keeps model files in RAM (64GB DDR4) so repeat invocations skip most disk I/O. Concurrent invocations serialize via a flock-based GPU mutex (`$XDG_RUNTIME_DIR/qwen-localreview.lock`). After a successful review, the warm server auto-starts in the background so the next review skips the model load (see keep-warm mode below).

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

## Inference configuration: FP8 KV cache on Ada

This section is both a record of why the current configuration was chosen and a short educational piece on why FP8 KV cache is the right lever for this hardware. The full investigation lives in the git history (Stage 0 through done condition, commits `7eb52e6` to `3f1c01c`) and in `tests/results/`. This is the executive summary.

### What we run

`Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` weights (INT4 via the Marlin kernel) with `kv_cache_dtype="fp8_e4m3"` (FP8 KV cache via vLLM's FlashInfer attention backend), `enforce_eager=True`, `gpu_memory_utilization=0.90`, `max_model_len=32768`. The `LLM()` construction is in `review.py`.

### Measured improvements

Bench numbers from a fixed corpus of 4 representative diffs (`tests/fixtures/diffs/`). Baseline = 14B AWQ + FP16 KV. Current = 14B AWQ + FP8 KV. Both runs use the same model, sampling params, and context window.

| metric | baseline | current | delta |
|---|---:|---:|---:|
| prefill TPS (aggregate) | 1114 | 1517 | **+36%** |
| **decode TPS** (aggregate) | 17.8 | 28.2 | **+58%** |
| total wall time (4 fixtures) | 18.9 s | 14.9 s | -21% |
| post-load VRAM | 19.03 GB | 18.12 GB | -0.91 GB |
| review quality | reference | >= baseline on every fixture, strictly better on one | n/a |

In practical terms this is roughly a **1.5x to 1.6x faster local review** with no measurable quality cost. Decode dominates the wall-clock time for review-style outputs (a real review generates 100 to 300 tokens of findings), so the +58% decode rate is the change that actually shows up to users.

### Why it improved: autoregressive decode is memory-bandwidth-bound

The single change is the KV cache dtype: FP16 (2 bytes per element) to FP8 (1 byte per element). Same model weights, same sampling params, same context window, same everything else.

The KV cache is where the attention layers store the keys and values for every token in the context window so future tokens do not have to recompute them. For Qwen2.5-Coder-14B at 32K context, the KV cache is roughly **6 GB in FP16, 3 GB in FP8**.

The big speedup is not the memory savings on its own. The big speedup is that **autoregressive decode is memory-bandwidth-bound, not compute-bound**.

To generate the next token, the model has to read the *entire* KV cache from VRAM through the attention kernel. Halving the KV cache means halving the bytes that have to move across the memory bus on every decode step, and on a memory-bandwidth-limited workload that translates almost linearly into tokens per second. The +58% decode TPS is roughly the bandwidth ratio you would predict from the dtype change alone.

Prefill (+36%) is a smaller win because prefill is more compute-bound: it is one large matrix multiply over the whole prompt, so KV bandwidth matters less. Most of the prefill speedup comes from FlashInfer's attention kernel being better tuned for Ada than the default Flash Attention path.

This is the intuition that drives most "make local LLM inference faster on consumer-ish hardware" decisions. If you can shrink the KV cache (or the weights) without losing quality, you get the speedup almost for free, because the bottleneck is not arithmetic, it is bytes per second across the memory bus.

### How Ada makes this cheap: hardware FP8

`RTX 4000 SFF Ada` is `sm_89`, the 4th generation of NVIDIA tensor cores. The 4th generation added **hardware FP8 support** in two formats: **E4M3** (1 sign bit, 4 exponent bits, 3 mantissa bits, range +/- 448) and **E5M2** (1 sign bit, 5 exponent bits, 2 mantissa bits, larger range, less precision). This is the same FP8 hardware the H100 has. We use E4M3 because the dynamic range is plenty for KV cache values (which are bounded by softmax and normalization upstream) and the extra mantissa bit gives slightly better precision.

Without hardware FP8, the FP8 KV cache path would be a slow software emulation: load the FP8 bytes, dequantize to FP16 in registers, then multiply. With hardware FP8, the dequantize-and-multiply happens in a single tensor-core instruction, which is why the speedup tracks the bandwidth ratio so closely instead of being eaten by extra ALU work.

For the architectural detail: Ada's tensor cores execute FP8 matrix-multiply-accumulate operations with FP16 or FP32 accumulators. The FP8 inputs come from VRAM at FP8 cost (half the bandwidth of FP16), and the math is done in the higher-precision accumulator, so the arithmetic precision of the attention output is essentially unchanged. The only quality cost is the rounding when values are *stored* into the FP8 KV cache.

### How vLLM wires this together

vLLM's default attention backend (Flash Attention) does not support FP8 KV cache. When you set `kv_cache_dtype="fp8_e4m3"`, vLLM auto-routes attention through **FlashInfer**, a separate kernel library with hand-tuned Ada-friendly paths for FP8 KV.

FlashInfer JIT-compiles its attention kernels on first use, via `ninja`. This is the source of one of the gotchas the project hit during development: `ninja` is pip-installed at `.venv/bin/ninja`, but invoking `.venv/bin/python` directly does not put `.venv/bin` on the subprocess `PATH` the way `source .venv/bin/activate` would. So `subprocess.run("ninja", ...)` inside FlashInfer fails with `FileNotFoundError`. Both `review.py` and `tests/_harness.py` prepend `.venv/bin` to `PATH` at import time to work around this.

The AWQ INT4 weights are unchanged across this experiment. The model is still quantized the same way (INT4 weights via the Marlin kernel). Only how the KV cache stores its attention state changed.

### What we tried that did not work

The full investigation went through five stages. The four that did not change the default are useful negative results:

- **Stage 2 (vLLM upgrade evaluation):** discarded. PyPI's JSON API confirmed vLLM 0.19.0 is the latest stable release as of `2026-04-10`. There was no upgrade target to evaluate. Re-check this when you eventually upgrade.
- **Stage 3 (32B AWQ at reduced context):** rejected. The 32B model technically loads on this hardware, but vLLM's own profiler reports a maximum sustainable `max_model_len` of 2336 tokens once weights, activations, and the AWQ kernel scratch space are accounted for. `review.py` needs roughly 4347 tokens minimum to fit a real system prompt plus one diff plus the 4096-token output reserve, and 2336 < 4347. Cutting the output reserve would silently truncate real reviews. The investigation chronology with all six failed configurations is in `tests/results/stage3-32b.md`.
- **Stage 4 (FP8-dynamic 14B weights):** rejected. The same model architecture with FP8 weights instead of INT4 AWQ weights had +19% prefill but **-58% decode** (twice the weight VRAM means twice the bytes to read on every decode step), used 1.18 GB more VRAM, forced the context window down to 8K, and introduced a format-compliance regression on one fixture (the model emitted `BLOCK file:line` without the brackets the system prompt requires, which would have been silently dropped by `integration/call_local.sh`'s line filter).

The general lesson from those failures: on a 20 GB Ada card, doubling weight VRAM is *strictly* worse than halving KV cache bytes. The hardware budget is tight enough that any change that grows weights costs more in decode bandwidth than it buys in anything else. FP8 KV cache wins because it shrinks the dominant per-decode-step memory traffic without touching the weights at all.

### Risks and what to watch

1. **VRAM headroom is tight, just less tight than before.** 18.12 GB used out of 19.55 GB total, roughly **1.4 GB headroom**. The baseline had only ~0.5 GB. Things that could push it over the edge:
   - Another GPU process holding memory between vLLM invocations. The preflight in `review.py` now computes its threshold from `gpu_memory_utilization * total` (0.90 * 19.55 = 17.6 GB), which mirrors vLLM's own startup check. Measured at idle (2026-04-13): ~1.8 GB headroom after torch CUDA init. Any process using more than ~1.8 GB of VRAM will trigger the preflight warning.
   - A future vLLM upgrade with different memory accounting.
   - Re-enabling CUDA graphs (`enforce_eager=False`). The original reason `enforce_eager=True` was set in commit `8321af1` was a CUDA-graph OOM. The new headroom may or may not be enough to fix that; it would need re-measuring before flipping the flag.

2. **FP8 is lossy quantization.** It is empirically below the noise floor of LLM behavior in published work and on the four fixtures we measured, but on a sufficiently long context (~32K tokens of dense code) small errors could accumulate. The way you would find out is a quality regression on a real review, not a unit test. Mitigation: the harness fixtures and `tests/results/` give you a cheap way to re-measure if you suspect a regression.

3. **Quality is measured on four fixtures.** Two synthetic bug fixtures (command injection, off-by-one), one benign (sampling param tuning), one synthetic security regression (path traversal). Notably **the 14B model misses the path-traversal regression on both baseline and FP8 KV**. That gap is real, FP8 KV does not fix it, and a bigger model would. We confirmed the bigger model does not fit on this hardware.

4. **FlashInfer JIT compile is a silent break risk.** If a future environment change removes `.venv/bin/ninja` or breaks the JIT, the FP8 KV path raises `FileNotFoundError`, falls into the fail-open handler in `review.py`, and reviews come back empty with only a `[qwen]` stderr line. Production users would see "no findings" without realizing the inference path is broken. **Test 3 in `tests/test-review.sh --full` catches it on first run** but the fast pre-push hook does not. Worth a periodic `--full` run.

5. **vLLM 0.19.0 pin.** The whole experiment depends on vLLM 0.19.0's specific behavior. `setup.sh` warns on version mismatch but does not enforce it. When you eventually upgrade, the FlashInfer kernel could change, the FP8 layout could change, or the FP8 KV path could regress. Plan to re-run `tests/bench.py --config baseline --full` and `--config stage1-fp8kv --full` against the new version before adopting it.

6. **No regression test for the measurement itself.** The bench and eval harnesses can be re-run on demand, but nothing automatically alerts on a perf or quality regression. Re-running is a manual gesture, and it currently happens only when someone notices something is off.

7. **Format-compliance regression class.** Stage 4 demonstrated that a different-but-similar model can silently emit findings the consumer's tag prepender will drop. Stage 1's actual default does not do this on our fixtures, but the failure mode would not appear in unit tests, only in production output. The mitigation is `integration/call_local.sh`'s defensive filter (which only forwards lines matching `^\[(BLOCK|WARN|NOTE)\]`); we verified it handles unrecognized stdout cleanly.

The biggest practical risk is **#1 (VRAM headroom)** if you ever change anything memory-sensitive. The biggest theoretical risk is **#2 (FP8 quality regression on long contexts)**, which would not manifest until you review a really big diff. The harness gives you a way to investigate if it does.

### How to re-validate

```bash
# Bench and eval the current production config:
.venv/bin/python tests/bench.py --config stage1-fp8kv --full
.venv/bin/python tests/eval.py  --config stage1-fp8kv --full

# Bench and eval the prior baseline (still defined in tests/_harness.py):
.venv/bin/python tests/bench.py --config baseline --full
.venv/bin/python tests/eval.py  --config baseline --full

# Diff tests/results/stage1-fp8kv.md against tests/results/baseline.md
# manually for any regressions in finding quality.
```

The four fixtures, the baseline output, and the FP8 KV output are all committed under `tests/results/` so any future change can be diffed against the historical record.

## Keep-warm mode

After a successful cold-path review, `review.py` automatically launches `warm.py` as a detached background process. The warm server loads the model and waits for the next review. This means:

1. First `/codereview` in a session: cold path (~30-60s). Warm server starts in background.
2. Second `/codereview` (minutes later): warm path (~1-7s). Model already loaded.
3. After 15 minutes of no reviews: warm server exits, VRAM freed.

No manual steps required. The warm server is self-managing.

```bash
# The warm server can also be started manually if desired:
.venv/bin/python warm.py

# Check if a warm server is running:
ls -la $XDG_RUNTIME_DIR/qwen-localreview.sock
```

The warm server:
- Holds the GPU flock for its lifetime (serializes with cold-path reviews)
- Exits after 15 minutes of idle (configurable via `LOCAL_WARM_TIMEOUT`)
- Monitors VRAM every 30 seconds; exits if another process starts using the GPU (>256 MiB external allocation detected)
- Cleans up the socket and releases the flock on shutdown (SIGTERM, SIGINT, idle, or VRAM yield)

If no warm server is running, `review.py` uses the cold path automatically. If the warm server crashes mid-request, `review.py` falls through to the cold path transparently (fail-open preserved). The GPU flock prevents duplicate warm servers: the auto-launch silently exits if one is already running.

The Unix domain socket (`$XDG_RUNTIME_DIR/qwen-localreview.sock`) is local-only and user-owned. No network exposure, no authentication needed (same trust boundary as the filesystem).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCAL_MODEL` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | HuggingFace model ID |
| `LOCAL_MAX_MODEL_LEN` | `32768` | Max context length in tokens |
| `LOCAL_GPU_LOCK_TIMEOUT` | `30` | Seconds to wait for GPU flock before fail-open |
| `LOCAL_WARM_TIMEOUT` | `900` | Warm server idle timeout in seconds |
| `LOCAL_REVIEW_SCRIPT` | (none) | Absolute path to `review.py` |
| `LOCAL_REVIEW_VENV` | (none) | Absolute path to project `.venv` |
