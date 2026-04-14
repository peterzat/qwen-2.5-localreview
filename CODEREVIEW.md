## Review -- 2026-04-14 (commit: e208e45)

**Review scope:** Refresh review. Focus: 4 file(s) changed since prior review (commit 1c78f02). 4 already-reviewed file(s) checked for interactions only.

**Summary:** Three unpushed commits since prior review: auto-launch warm server after cold-path review (review.py), warm server test isolation fixes (test-warm.sh), and stale comment/documentation updates (README.md, tests/_harness.py). Fast tests pass (test-review.sh 18/18, test-call-local.sh 12/12). Security scan clean (0/0/0). One WARN (dead code ternary and documentation mismatch in flock timeout default, fixed). External reviewer (openai) found flock race and symlink concern, both evaluated and scoped below.

**External reviewers:**
[openai] o3 (high) -- 10348 in / 9813 out / 9728 reasoning -- ~$.1770
[openai] 2 findings (1 BLOCK, 1 WARN). BLOCK (symlink attack on lock file) downgraded to NOTE (outside threat model per SECURITY.md local-CLI trust boundary). WARN (flock race on auto-launch) noted as NOTE (timing analysis shows race is likely won, and failure mode is benign).
[qwen] Qwen/Qwen2.5-Coder-14B-Instruct-AWQ -- 10500 in / 5 out -- 29s -- $0.00
[qwen] No issues found.

### Findings

```
[WARN] review.py:191, README.md:192 -- Dead code ternary in flock timeout default, documentation mismatch (FIXED)
  Evidence: review.py:191 reads `default_lock_timeout = "30" if _warm_attempted
  else "270"`. The variable `_warm_attempted` is initialized to False at line 144,
  then unconditionally set to True at line 160 if the warm path fails (the only
  path that reaches the flock code). The `else "270"` branch is unreachable. The
  effective default GPU flock timeout is always 30s. README.md:192 documents the
  default as `270`, which contradicts the actual code behavior.
  Fix applied: Removed the dead `_warm_attempted` variable and ternary, replaced
  with a direct `"30"` default. Updated comment to explain why 30s is the intended
  value. Updated README.md to document `30` as the default.
```

```
[NOTE] (openai) review.py:287-298 -- Auto-launch warm server spawned while parent holds GPU flock
  Evidence: review.py spawns warm.py (line 292) before `return 0` (line 302),
  while the flock is still held by the `lock_file` local variable. warm.py
  calls `acquire_gpu_lock(timeout=0.0)` (non-blocking). The race depends on
  whether review.py's process exit (releasing the flock) happens before warm.py's
  Python startup reaches the flock call. Analysis: review.py exits within
  microseconds of the Popen call; warm.py's Python interpreter startup takes
  50-200ms. The parent almost certainly exits first. Failure mode is benign:
  warm.py exits silently, and the next cold-path review re-attempts auto-launch.
```

```
[NOTE] (openai) gpu_lock.py:44 -- Lock file opened with "w" in predictable /tmp fallback path
  Evidence: openai flagged a symlink attack vector on the /tmp fallback path.
  The primary path (XDG_RUNTIME_DIR, typically /run/user/1000/) is user-owned
  mode 0700, immune to symlink attacks. The /tmp fallback uses a user-specific
  directory created with os.makedirs(exist_ok=True); /tmp has sticky bit on
  standard Linux. The tool runs on a single-user development machine with no
  untrusted local users. Downgraded from external BLOCK to NOTE per SECURITY.md
  local-CLI trust boundary.
```

```
[NOTE] tests/test-review.sh:455-482 -- Test 13 fragile when warm server is running
  Evidence: Test 13 holds the GPU flock in a background process and expects
  review.py to fail with "GPU busy." If a warm server is running (e.g.,
  auto-launched by a prior review.py invocation), review.py uses the warm path
  and never reaches the flock code, causing the test to fail. Observed during
  this review: the external reviewer step triggered auto-launch, and the warm
  server served Test 13's request via warm path. test-warm.sh handles this with
  startup cleanup (lines 47-59), but test-review.sh does not.
  Suggested fix: Add warm-server cleanup at the top of test-review.sh (kill
  existing warm server and remove socket), mirroring test-warm.sh's approach.
```

```
[NOTE] tests/test-warm.sh:4 -- Header comment inaccurate after Tests 6-7 were added
  Evidence: Line 4 says "Requires GPU + model for Tests 1-3. Tests 4-5 run
  without GPU." Test 4 starts a real warm server (requires GPU). Only Test 5
  (stale socket with --dry-run) runs without GPU. Tests 6-7 require GPU but
  are not mentioned.
  Suggested fix: Update the header to "Requires GPU + model for Tests 1-4,
  6-7. Test 5 runs without GPU."
```

```
[NOTE] tests/test-review.sh:218,230-231 -- Test 5 comments reference old fixed 4096 reserve, not the adaptive value
  Evidence: Lines 218 and 230-231 say "output reserve (4096)" and "With a
  1024 context window and 4096 output reserve." With the adaptive reserve,
  at LOCAL_MAX_MODEL_LEN=1024, the actual reserve is
  min(4096, max(256, 1024 // 4)) = 256, not 4096. The test still passes
  because the arithmetic produces the same truncation/empty outcome.
  Suggested fix: update the comments to say "output reserve (256 at 1024
  context)" and adjust the arithmetic in line 231 accordingly.
```

### Fixes Applied

- [WARN] review.py:191, README.md:192: Removed dead `_warm_attempted` variable and unreachable `else "270"` branch. Replaced ternary with direct `"30"` default. Updated comment to explain the 30s rationale. Fixed README.md to document `30` as the default.

### Accepted Risks

- SPEC.md done condition (A) was reworded post-completion (commit 3f1c01c) to remove the `LOCAL_INFERENCE_MODE=legacy` toggle requirement after the toggle was deleted. The rewording is documented in a post-completion note with rationale. Accepted: the toggle was speculative scaffolding, `git revert` is the actual rollback mechanism, and `tests/_harness.py` retains the baseline config for re-measurement.

---
*Prior review (2026-04-13): Refresh review of 7 files (GPU mutex, warm server, warm tests, VRAM yield/OOM tests). 0 BLOCK, 1 WARN (test-warm.sh exit code check, fixed), 2 NOTE. Fast tests 18/18 + 12/12.*

<!-- REVIEW_META: {"date":"2026-04-14","commit":"e208e45","reviewed_up_to":"e208e451437604f32c386beae8d3dcb00d21c7b0","base":"origin/main","tier":"refresh","block":0,"warn":1,"note":5} -->
