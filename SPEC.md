## Spec -- 2026-04-13 -- Cleanup, VRAM preflight, quality validation fixtures

**Goal:** Fix stale code from the toggle add-then-remove sequence, retune the VRAM
preflight threshold from measured data, and add a quality validation fixture corpus
that tests whether the 14B model reliably catches bugs and avoids false positives
across two languages and two difficulty levels.

### Acceptance Criteria

- [x] All actionable code-level findings from the 2026-04-11 code review are
      resolved: stale `build_llm_kwargs` comment (review.py:14), stale
      line-number references and "mirrors exactly" comment
      (tests/_harness.py:52,64), `os.sys.executable` non-public API usage
      (tests/_harness.py:26), and prefill TPS measurement bias
      (tests/_harness.py docstring, per CODEREVIEW NOTE on lines 232-318).
      The remaining WARN (SPEC.md done condition A rewording) is recorded in
      CODEREVIEW.md Accepted Risks.
- [x] review.py's VRAM preflight threshold is updated based on a measured
      analysis of vLLM's actual startup VRAM requirement for the production
      config (14B AWQ + FP8 KV + 32K context + `gpu_memory_utilization=0.90`).
      The analysis derives the threshold from observed values, not
      back-of-envelope estimates, and the reasoning is recorded in a code
      comment at the preflight check. README.md risk #1 (VRAM headroom) is
      updated to reflect the new threshold.
- [x] A test verifies the VRAM preflight code path: warning fires when free
      VRAM is below the threshold, does not fire when above. The test runs
      without loading the vLLM model (no ~30s overhead), keeping it eligible
      for the fast test suite.
- [x] Eight new diff fixtures exist covering a 2x2x2 matrix: {Python, C++} x
      {correct code, buggy code} x {simple, subtle}. "Correct/simple"
      fixtures are obviously clean code. "Correct/subtle" fixtures contain
      code that looks suspicious but has no real defect (false-positive
      resistance). "Buggy/simple" fixtures contain clear bugs. "Buggy/subtle"
      fixtures contain non-obvious bugs the model must work to detect. Each
      fixture has a header comment documenting its matrix position (language,
      condition, difficulty), the expected reviewer behavior, and an explicit
      "intentional test data" marker so review tools do not treat purposeful
      defects as findings to fix.
- [x] A test script runs the quality fixtures through the production config
      and checks: (a) every buggy fixture produces at least one `[BLOCK]` or
      `[WARN]` finding, (b) every correct fixture produces zero `[BLOCK]` or
      `[WARN]` findings (`[NOTE]` is acceptable, "No issues found." is
      acceptable). The script reports pass/fail per fixture with a summary
      line. Fixtures where the model misses the expected behavior are flagged
      as known capability limitations rather than infrastructure failures.
- [x] Quality fixture tests are gated behind `--full` (or equivalent explicit
      opt-in) and do not run in the fast pre-push path. The script header
      documents expected wall-clock run time.
- [x] Quality fixture results for the production config are committed to
      `tests/results/` so future config or prompt changes can be diffed
      against the reference.
- [x] Existing fast tests (test-review.sh, test-call-local.sh) pass with all
      changes in place.

### Context

**Prior turn (abstract-yawning-raven).** Adopted FP8 KV cache as the default
inference config: +58% decode TPS, +36% prefill, -0.91 GB VRAM vs baseline.
Three alternatives rejected with evidence. The post-turn code review (2 WARNs,
7 NOTEs) found the VRAM preflight threshold was too lax (fixed from <10 GB to
<14 GB) plus stale comments left behind by the toggle add-then-remove sequence.
The 2026-04-11 CODEREVIEW.md has the full findings list.

**VRAM preflight background.** The current <14 GB threshold was set in commit
2f6ccaf based on the README's recommendation, but a 16.91 GB OOM was observed
during the review itself (above 14, so the preflight did not fire). vLLM's
startup check is `free >= gpu_memory_utilization * total` (0.90 * 19.55 =
17.6 GB). The analysis this turn should measure the actual free-VRAM-at-startup
values (before torch import, after torch import, etc.) and set the threshold to
catch cases that vLLM will reject.

**Existing fixture corpus.** Four diffs under `tests/fixtures/diffs/` (01 through
04) serve as the bench/eval comparison corpus and have committed baseline and
FP8 KV results. The 14B model catches 01 (cmd injection) and 02 (off-by-one),
passes 03 (benign sampling change), and misses 04 (path traversal) on both
configs. New quality fixtures are additive, not replacements.

**Quality fixture design constraints.** The model runs at temperature=0.2, which
gives fairly stable output. Fixtures should be designed with margin: simple-buggy
should trigger BLOCK/WARN reliably across runs, not just sometimes. Subtle-buggy
may have a lower hit rate, and the committed results document the model's actual
capability. Correct-subtle fixtures (suspicious-looking but sound code) are the
highest-value test: false positives erode trust faster than false negatives.

**Codereview interaction with intentional-bug fixtures.** `/codereview` reviews
the uncommitted diff, not the full repo. Once committed, fixture `.patch` files
are inert: they are not in the diff, so codereview does not evaluate them and
codefix does not attempt to "fix" the intentional bugs. The risk window is the
commit that adds the fixtures. The header comment ("intentional test data")
plus the file location (`tests/fixtures/diffs/`) and extension (`.patch`) are
sufficient for codereview's precision-biased instructions to recognize test data.
No changes to codereview or codefix are needed.

**Subtle-bug calibration.** The existing results give a capability baseline:
the 14B model catches pattern-matchable bugs (command injection via
`os.system`, arithmetic off-by-one) but misses reasoning-dependent bugs
(path traversal guards removed in a diff). "Subtle" fixtures should require
semantic reasoning beyond pattern matching: TOCTOU races, exception swallowing
that masks specific failures, iterator invalidation, move-after-use. The goal
is bugs the model catches sometimes (quality signal across config changes),
not bugs it never catches (ceiling calibration, useful but less actionable).
If a subtle fixture reliably gets missed, document it as a known capability
gap in the committed results rather than weakening the fixture.

**Framework practices relevant to this work.**
- *Precision over recall.* The quality fixtures exist to measure this property:
  does the model flag real bugs (recall) without flagging clean code (precision)?
- *Verification over prompting.* The fixtures are the verification artifact. If
  the model's quality on a fixture is unsatisfactory, the response is better
  fixtures or a better prompt, not a spec change.
- *Small, committable increments.* Three workstreams (cleanup, VRAM, fixtures)
  can be done as separate commits within a single turn.

---
*Prior spec (2026-04-10): Ada-aware inference experiments (abstract-yawning-raven). 13/13 criteria met. Adopted FP8 KV as default, rejected vLLM upgrade, 32B AWQ, and FP8 weights.*

### Proposal (2026-04-13)

**What happened.** This turn cleaned up stale code from the FP8 KV adoption
sequence, retuned the VRAM preflight from measured data, and built a quality
validation corpus. The VRAM preflight now computes its threshold from
`gpu_memory_utilization * total` rather than a hardcoded value, mirroring
vLLM's own startup check (measured: 0.90 * 19.55 = 17.6 GB, ~1.8 GB idle
headroom). Eight quality fixtures in a 2x2x2 matrix ({Python, C++} x
{correct, buggy} x {simple, subtle}) scored 8/8 on the production config:
all four buggy fixtures triggered BLOCK findings (including both "subtle"
cases), all four correct fixtures produced zero false BLOCK/WARN. A polish
pass added env var validation, crash/timeout detection in the quality test
runner, and standardized headers on all 12 fixtures. Fast tests: 12/12 +
12/12.

**Questions and directions.**

- *Integration in zat.env.* This project is functionally complete. The
  consumer side (review-external.sh changes, .env entries, call_local.sh
  wiring) lives in zat.env per the boundary rule. That is the natural
  next step, done from within zat.env itself.

- *System prompt tuning.* The 14B model misses the path-traversal fixture
  (04) on both configs. The quality matrix gives a repeatable way to
  measure whether prompt changes improve recall without introducing false
  positives. A focused prompt engineering turn could target the gap, using
  the 12-fixture corpus as the evaluation harness.

- *vLLM upgrade.* The project is pinned to vLLM 0.19.0. When a newer
  release ships, the bench/eval harness and quality matrix give a
  mechanical way to validate: re-run both, diff results against committed
  baselines, check for regressions. No work until a new version exists.

- *Fixture expansion.* The current matrix covers two languages and two
  difficulty levels. Additional languages (Go, Rust, TypeScript) or
  additional bug classes (concurrency, resource leaks) would broaden
  coverage. Worth doing if/when the system prompt or model changes and
  the existing 12 fixtures no longer discriminate between configs.

<!-- SPEC_META: {"date":"2026-04-13","title":"Cleanup, VRAM preflight, quality validation fixtures","criteria_total":8,"criteria_met":8} -->
