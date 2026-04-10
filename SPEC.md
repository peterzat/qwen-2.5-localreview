## Spec -- 2026-04-10 -- Ada-aware inference experiments (abstract-yawning-raven)

**Goal:** Measure whether Ada-native inference formats (FP8 KV cache, FP8 weights),
a larger model (32B-AWQ), and a newer vLLM buy better adversarial review output on
the RTX 4000 SFF Ada, then either adopt a new default configuration (with the
current config preserved as a legacy toggle) or prove the current configuration
is already optimal and record that finding in CLAUDE.md.

### Acceptance Criteria

- [x] A benchmark harness at `tests/bench.py` loads the configured model, runs a
      fixed corpus of 3-5 representative diffs checked in under `tests/fixtures/diffs/`,
      and reports per-diff prefill tokens/s, decode tokens/s, peak VRAM (via
      `torch.cuda.max_memory_allocated`), and wall time. Takes `--config <name>`
      so all stages reuse it.
- [x] An eval harness at `tests/eval.py` runs the same fixed corpus and writes raw
      model outputs to `tests/results/<config-name>.md` (one file per measured
      config) so configs can be compared side-by-side by reading committed files.
- [x] Both harnesses are gated behind `--full` (or an equivalent slow-test marker)
      so the existing pre-push hook path completes in under 2 seconds. Running
      `tests/test-review.sh` and `tests/test-call-local.sh` without `--full` still
      passes with the new code present.
- [x] A baseline measurement run for the current production config (14B AWQ +
      FP16 KV + `enforce_eager=True` + `gpu_memory_utilization=0.90`) exists as
      `tests/results/baseline.md` plus recorded bench numbers (committed file or
      commit message table). This baseline is the reference for every later stage.
- [x] Stage 1 (FP8 KV cache on current AWQ model) has been run: bench numbers and
      `tests/results/stage1-*.md` exist, and the stage outcome is recorded in the
      commit message as either "adopted as new winning config," "reverted (quality
      regression)," or "reverted (load failure)" with the evidence that drove the
      decision.
- [x] Stage 2 (vLLM upgrade evaluation) has been run: a throwaway venv was used
      (primary `.venv/` untouched unless the upgrade was adopted), bench+eval
      numbers were captured, and the outcome is recorded as either "adopted"
      (with `setup.sh` pin updated and reason in commit message) or "discarded"
      (with reason). If discarded, the throwaway venv is removed.
- [x] Stage 3 (`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` at reduced context) has been
      run: bench+eval numbers exist (or an explicit "failed to load at every
      tested `max_model_len`" record with the OOM evidence), and the outcome is
      recorded as "promoted to default," "documented as opt-in via `LOCAL_MODEL`,"
      or "rejected" with reason.
- [x] Stage 4 (FP8-dynamic 14B weights variant) has been run, or explicitly
      skipped with a recorded reason (e.g., "Stage 2 vLLM upgrade not adopted so
      FP8 weight kernels still unavailable" or "exact HF repo not found"). A skip
      is a valid stage outcome as long as the reason is recorded.
- [x] Stages progressed strictly sequentially: each stage's bench numbers and
      eval side-by-side are captured before the next stage begins, visible as
      either per-stage commits or a per-stage entry in a results log. The
      configuration each stage starts from is the winning config from the prior
      stage (Stage N does not re-run Stage N-1's experiments from scratch).
- [ ] **Done condition (exactly one of A or B is met):**
      - **(A)** `review.py` adopts a new default config that beat baseline on the
        eval harness, AND an environment variable (e.g. `LOCAL_INFERENCE_MODE=legacy`)
        restores the previous `AWQ INT4 + FP16 KV + enforce_eager=True +
        gpu_memory_utilization=0.90` path exactly. Both code paths are exercised
        by a test (unit test or `tests/test-review.sh` extension) that confirms
        the toggle actually selects different `LLM()` constructor arguments. The
        new default and the legacy toggle are documented in `README.md`.
      - **(B)** No stage produced a configuration that beat baseline on the eval
        harness. `CLAUDE.md` gains a dated, concise "Inference config rationale"
        section naming what was tested, what was measured, and why the current
        config won, so the question is not re-investigated in a future session.
- [ ] `review.py` public contract is preserved end-to-end: stdin diff -> stdout
      findings -> stderr `[qwen]` status line -> exit 0 on all paths including
      fatal errors. `tests/test-review.sh` (all 6 tests including the fast path)
      and `tests/test-call-local.sh` pass on the final code.
- [ ] Fail-open behavior verified on the final adopted config: a deliberate OOM
      (e.g., oversized fake diff or a known-bad `LOCAL_MAX_MODEL_LEN`) produces a
      structured `[qwen] ... error: ... -- 0 in / 0 out -- Ns` stderr line and
      exit 0, same as before.
- [ ] No modifications to `~/src/zat.env/` or `~/.config/claude-reviewers/.env`
      were made. Git status outside this repo is unchanged.

### Context

**Source plan (advisory).** `~/.claude/plans/abstract-yawning-raven.md` is the
planning document this spec implements. Treat it as design notes, not
specification. When plan details conflict with observed behavior (e.g., a vLLM
API detail differs from what the plan claims), trust the code.

**Current production configuration** (`review.py:118-133`, commit `8321af1` +
`2f56c64`): `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, `max_model_len=32768`,
`gpu_memory_utilization=0.90`, `enforce_eager=True`, no explicit `kv_cache_dtype`
(implicit FP16), sampling `temperature=0.2, top_p=0.8, top_k=20,
repetition_penalty=1.05, max_tokens=4096`. `enforce_eager=True` exists to avoid
a CUDA-graph-induced OOM fixed in `8321af1`; any re-enablement of CUDA graphs
must re-verify there is no OOM regression.

**Hardware budget.** RTX 4000 SFF Ada, sm_89, 20 GB VRAM, 70 W TDP, CUDA 13.x.
Ada's 4th-gen tensor cores have hardware FP8 (E4M3/E5M2). vLLM 0.19.0 supports
`kv_cache_dtype="fp8_e4m3"`. Machete kernels are gated on `min_capability=90`
in 0.19.0, so sm_89 currently uses Marlin for AWQ. Whether a later vLLM lowers
that gate is one of the Stage 2 questions. Back-of-envelope VRAM: 14B AWQ + FP16
KV at 32K context ~= 15 GB; 14B AWQ + FP8 KV ~= 12 GB; 32B AWQ + FP8 KV at 16K
context ~= 18 GB (tight).

**Stage-pause discipline.** The user explicitly wants a pause and review after
each stage, within a single turn. Checking off a stage's acceptance item before
starting the next is how the pause is made mechanically visible; verifying this
spec means confirming that bench numbers and eval outputs for Stage N exist
before Stage N+1's changes appear in the history.

**Valid stage outcomes.** A stage can legitimately end in "reverted, no code
change" or "skipped with reason." What is not acceptable is starting Stage N+1
without a recorded decision (adopt / revert / skip) on Stage N. Failure to load
a model due to OOM, a vLLM build error, or a missing HF repo is a valid recorded
outcome and lets the spec progress.

**Legacy toggle shape (done condition A).** The legacy toggle restores the
exact pre-change `LLM()` arguments used in the baseline (including
`enforce_eager=True` and absence of `kv_cache_dtype`). A test must confirm the
toggle selects different arguments at the point of `LLM()` construction, not
just a different code branch that happens to produce the same object. Any env
var name is acceptable; `LOCAL_INFERENCE_MODE=legacy` is the suggested shape.

**Done condition B is not a consolation.** If measurement proves the current
config wins, recording that finding in `CLAUDE.md` is a first-class deliverable
and the turn is complete. "No change to review.py" is a legitimate outcome; the
harness plus the recorded rationale are what this turn produced.

**Non-goals.** No changes to the `review.py` CLI surface (stdin/stdout/stderr/
exit contract). No server, no persistent process, no TCP/IP. No modifications to
zat.env or `~/.config/claude-reviewers/.env`. No packaging or versioning changes.
No automated quality scoring of reviews (quality is compared manually via
committed eval outputs).

**Framework practices relevant to this work.**
- *Verification over prompting.* Stage 0 is the load-bearing deliverable: every
  claim about "stronger output" must be grounded in harness measurements, not
  vibes.
- *Small, committable increments.* Each stage should be at least one commit.
  Do not stack untested stage changes.
- *Precision over recall.* This is a reviewer project: false positives erode
  trust. An eval run where the new config produces more findings is not
  automatically "better" -- the findings have to be accurate.
- *Spec is code.* Checking off a criterion requires the criterion to be
  mechanically verifiable. Bench numbers and eval files are the artifacts.
- *Fail-open contract is non-negotiable.* Any experiment that breaks the exit-0-
  on-error path is a regression, not an experiment.

<!-- SPEC_META: {"date":"2026-04-10","title":"Ada-aware inference experiments (abstract-yawning-raven)","criteria_total":13,"criteria_met":9} -->
