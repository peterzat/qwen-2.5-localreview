## Test Strategy Review -- 2026-04-13

**Summary:** Four bash test suites covering review.py guard logic and warm path (test-review.sh, 18 checks), the call_local integration shim (test-call-local.sh, 12 checks), a GPU-gated quality validation matrix (test-quality.sh, 8 fixtures), and warm server lifecycle tests (test-warm.sh, 9 checks). Pre-push hook runs the two fast suites and passes. A git remote now exists, so the hook fires on actual pushes.

**Test infrastructure found:** bash test scripts (tests/test-review.sh, tests/test-call-local.sh, tests/test-quality.sh, tests/test-warm.sh), Python bench/eval harness (tests/_harness.py, tests/bench.py, tests/eval.py), pre-push git hook (.githooks/pre-push via core.hooksPath), git remote (origin), 12 diff fixtures (tests/fixtures/diffs/), committed results (tests/results/), no CI, no coverage tools, no Makefile

### Findings

```
[WARN] Development loop cadence -- Pre-push hook is 6x slower than documented
  Current state: The pre-push hook takes ~12s wall clock (test-call-local.sh ~2s,
  test-review.sh ~10s). The prior TESTING.md documented "< 2 seconds" total and
  test-review.sh at 1.7s. Since then, test-review.sh grew from 4 checks to 12
  checks (tests 7-10 added in the cleanup/VRAM/quality turn). Tests 4, 5, 6, 9,
  and 10 each invoke review.py, which loads the transformers tokenizer (~2.7s per
  invocation). These tokenizer loads dominate wall time. The hook is still fast
  enough to be non-annoying (12s, not minutes), but the documented claim is wrong
  and the trend is in the wrong direction. Each new test that invokes review.py
  adds another ~2.7s to the fast path.
  Recommendation: Either (a) consolidate tokenizer-loading tests into fewer
  review.py invocations (one invocation that tests multiple conditions via
  separate input files), or (b) add a --fast-only mode that skips
  tokenizer-dependent tests, or (c) accept 12s and update the pre-push hook
  comment from "< 2 seconds" to "< 15 seconds." Option (c) is the simplest and
  honest.
```

```
[NOTE] Test coverage strategy -- Quality matrix adds high-value precision/recall coverage
  Current state: The 2x2x2 quality matrix (8 fixtures: {Python, C++} x {correct,
  buggy} x {simple, subtle}) plus the 4 original bench/eval fixtures give 12
  total diff fixtures. test-quality.sh checks that buggy fixtures trigger
  BLOCK/WARN findings and correct fixtures do not (false-positive resistance).
  Results are committed to tests/results/ for baseline diffing. The quality test
  distinguishes infrastructure failures (FAIL) from model capability gaps (GAP).
  Critical paths (fail-open, output tagging, input truncation, VRAM preflight,
  env var validation, timeout) are all tested. SPEC.md has 8/8 criteria met.
  Recommendation: Nothing to flag.
```

```
[NOTE] Automatic test execution -- Pre-push hook works end-to-end now
  Current state: git remote exists (origin at github.com:peterzat/qwen-2.5-
  localreview.git). core.hooksPath is set to .githooks. The pre-push hook runs
  test-call-local.sh and test-review.sh (both fast suites). test-quality.sh is
  correctly excluded from the pre-push path (GPU-gated, ~2-3 minutes).
  Recommendation: Nothing to flag. Prior WARN resolved.
```

```
[NOTE] CI/CD integration -- No CI, appropriate for current stage
  Current state: Single-developer local tool with a git remote. No deployment
  pipeline. The pre-push hook provides the automatic test gate. No PR workflow
  in use.
  Recommendation: Nothing to flag.
```

```
[NOTE] Test framework choices -- Bash test harness is appropriate
  Current state: Custom pass/fail/gap helpers with summary counts. Three test
  scripts, each self-contained. The bench/eval Python harness (_harness.py) is
  separate infrastructure for performance measurement, not unit testing.
  Recommendation: Nothing to flag.
```

```
[NOTE] Fixture and data management -- Clean, well-isolated
  Current state: Bash scripts use mktemp -d with trap cleanup. test-call-local.sh
  creates mock Python scripts to simulate model behavior. test-quality.sh runs
  real fixtures through the production config. Fixtures have header comments
  documenting matrix position and expected behavior. Results are committed for
  baseline comparison.
  Recommendation: Nothing to flag.
```

```
[NOTE] Flaky test patterns -- None detected
  Current state: No sleep calls (timeout test uses the `timeout` command, not
  sleep), no shared state, no order dependence. GPU-dependent tests (test 3 in
  test-review.sh, test-quality.sh) are cleanly gated. Quality fixtures run at
  temperature=0.2, giving fairly stable output. The GAP classification in
  test-quality.sh handles stochastic model behavior correctly: a missed bug is
  flagged as a capability gap (informational), not a test failure.
  Recommendation: Nothing to flag.
```

```
[NOTE] Missing test categories -- None needed at current scale
  Current state: The project is a single inference script and a shell shim.
  Integration testing with the consumer (review-external.sh) is scoped to
  zat.env per the boundary rule. The quality matrix provides a form of
  acceptance testing for model output quality.
  Recommendation: Nothing to flag.
```

### Status of Prior Recommendations

- **[WARN] No git remote:** Resolved. Remote now configured (origin at github.com:peterzat/qwen-2.5-localreview.git). Pre-push hook fires on actual pushes.
- **[NOTE] Fast path timing:** Regressed. Was 1.7s, now 12s due to new tokenizer-loading tests. Raised as WARN above.

---
*Prior review (2026-04-09): Two bash test suites, pre-push hook working, no git remote. Fast tests completed in ~1.7s. One WARN (no git remote), seven NOTEs (all "nothing to flag").*

<!-- TESTING_META: {"date":"2026-04-13","commit":"afe10c4","block":0,"warn":1,"note":7} -->
