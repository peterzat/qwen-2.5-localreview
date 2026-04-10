# Stage 3 results: Qwen/Qwen2.5-Coder-32B-Instruct-AWQ

**Outcome: rejected.** The 32B AWQ model does not fit usefully on a
20 GB RTX 4000 SFF Ada at any context length the project would actually
use. Investigation chronology below; raw error logs are in
`stage3-32b-16k.bench.stderr.log` and `stage3-32b-8k.bench.stderr.log`.

## Hardware budget recap

- Total VRAM: 19.55 GiB (RTX 4000 SFF Ada)
- 32B AWQ INT4 weights on disk: ~16 GB (15 shards via snapshot_download)
- vLLM 0.19 + PyTorch + CUDA runtime overhead: ~1-2 GB
- Activation peaks during forward pass: ~0.5-1 GB
- KV cache for Qwen2.5-Coder-32B (64 layers, 8 KV heads, 128 head dim)
  in FP8: ~131 KiB/token. 16K context => ~2.1 GB; 8K => ~1 GB; 4K => 0.5 GB

The first three rows already total ~18-19 GB before any KV is allocated.

## Attempts (each on top of the prior, applied to stage3-32b-16k unless
noted)

### Attempt 1: literal spec config (16K, FP8 KV, gpu_memory_utilization=0.90)

Result: **OOM during weight loading**. Error site:
`vllm/model_executor/layers/quantization/awq_marlin.py:468
process_weights_after_loading -> _convert_awq_to_standard_format`.
"Tried to allocate 1.05 GiB. GPU 0 has ... 49 MiB free."

Diagnosis: vLLM auto-selects the awq_marlin path on Ada (Marlin's
min_capability=75 is satisfied; Machete would need 90). The conversion
from packed AWQ format to Marlin's standard format needs ~1 GiB of
scratch on top of the loaded weights. With weights+overhead already at
~17.5 GB out of 19.55, the conversion does not fit.

### Attempt 2: same as Attempt 1 but max_model_len=8192

Result: **same OOM, same error site.** Weight loading is independent
of max_model_len -- the cache pool is sized after weights are loaded,
so reducing context did not help.

### Attempt 3: 16K + quantization="awq" (skip Marlin conversion)

Result: weight loading succeeded; **OOM moved to the dummy profile
forward pass**. Error site:
`vllm/model_executor/layers/quantization/awq.py:273 torch.matmul`
(gate_up_proj inside an MLP layer). "Tried to allocate 864.00 MiB."

Diagnosis: vLLM's `profile_run` issues a dummy forward pass with batch
size = `max_num_batched_tokens` (which defaults to `max_model_len`) to
measure activation memory. At 16K tokens the gate_up_proj output is
exactly 16384 * 27648 * 2 bytes / 2 = 864 MiB, matching the allocation
that failed. This is not a real runtime cost -- it is a sizing probe.

### Attempt 4: 16K + quantization="awq" + max_num_seqs=1

Result: **same OOM in profile_run.** `max_num_seqs` does not limit the
profile pass; it caps concurrent requests, not per-step token batching.

### Attempt 5: 16K + quantization="awq" + max_num_seqs=1 + max_num_batched_tokens=2048

Result: **profile_run succeeded** (gate_up_proj activation drops to
~108 MiB at 2048 batch). New failure point:
`vllm/v1/core/kv_cache_utils.py:626 _check_enough_kv_cache_memory`.
"No available memory for the cache blocks. Try increasing
gpu_memory_utilization."

Diagnosis: weights + activation overhead consumed essentially all of
the 0.90 * 19.55 = 17.6 GB budget, leaving zero room for any KV cache
blocks at all.

### Attempt 6: same as Attempt 5 but gpu_memory_utilization=0.95

Result: **same "no memory for cache blocks" error.** Bumping the
utilization cap by 5 percentage points (~1 GB) was not enough -- the
activation peak from the dummy profile run was already pushing total
process memory above 0.95 * 19.55 too.

### Attempt 7 (8K): max_model_len=8192, gpu_memory_utilization=0.98, max_num_batched_tokens=1024

Result: still rejected, but now with a **precise diagnosis from vLLM
itself**:

> ValueError: To serve at least one request with the model's max seq
> len (8192), 1.0 GiB KV cache is needed, which is larger than the
> available KV cache memory (0.29 GiB). Based on the available memory,
> the estimated maximum model length is **2336**.

So the 32B AWQ model technically fits at `max_model_len <= 2336` on
this card. Anything above that, vLLM refuses to start because the KV
cache cannot hold even one request at the configured length.

## Why 2336 is not adopted as opt-in

`review.py` currently defaults to `max_model_len=32768` and
`OUTPUT_RESERVE_TOKENS=4096`. The minimum `max_model_len` that lets the
context guard accept a non-empty user input is `system_prompt_tokens +
output_reserve + 1`. With the production system prompt (~250 tokens
for the principal-engineer review prompt), that minimum is ~4347
tokens. **2336 < 4347.** A 32B-AWQ vLLM instance on this card cannot
even initialize with review.py's current output reserve.

Lowering `OUTPUT_RESERVE_TOKENS` would let it start, but reviews
routinely produce 1500-2500 output tokens (see baseline fixture 01 at
173 tokens, fixture 02 at 128 tokens; longer multi-finding outputs
exceed 1000). The output reserve was set at 4096 to accommodate that.
Cutting it to fit a 32B-on-tiny-context configuration would silently
truncate real reviews.

## Bench numbers

None for 16K or 8K -- both failed to load. The "estimated maximum
model length is 2336" line in Attempt 7's error is the only useful
quantitative result the experiment produced.

## Eval numbers

None. No fixture was ever processed by the 32B model, so there is no
side-by-side comparison to file against `tests/results/stage1-fp8kv.md`.
The Stage 1 winner (14B AWQ + FP8 KV cache) remains the best
configuration this hardware can run.

## Conclusion

The 32B-AWQ-on-Ada plan rested on the back-of-envelope estimate
"32B AWQ + FP8 KV at 16K ~= 18 GB (tight)." Tight turned out to mean
"does not fit at all once activations and the AWQ kernel scratch space
are accounted for." Larger weight quantization (32B FP8 dynamic) would
be even worse -- those weights are ~32 GB, which is impossible on a
20 GB device. The next reasonable model-size lever for this hardware
would be a 7B model (smaller, weaker) or moving to a 24+ GB card.

**Recorded outcome: rejected -- 32B AWQ does not fit usefully on a
20 GB Ada device.** Stage 1 (14B AWQ + FP8 KV) remains the winning
config; Stage 4 inherits from Stage 1.
