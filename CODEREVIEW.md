## Review -- 2026-04-16 (commit: 8591da1)

**Review scope:** Refresh review. Focus: 5 file(s) changed since prior review (commit fb85346). 0 already-reviewed file(s) (all prior changes are in the focus set).

**Summary:** One unpushed commit replaces the broken VRAM yield polling mechanism with an explicit gpu-release script. warm.py drops ~30 lines of dead polling code and the serve() loop simplifies to idle-timeout-only select. New gpu-release bash script reads the state file, sends SIGTERM, and waits up to 5s. Tests updated: VRAM yield test replaced with gpu-release lifecycle tests (6, 6b). CLAUDE.md and integration guide document the preemption model. Fast tests pass (test-review.sh 22/22, test-call-local.sh 12/12). Security scan clean (0/0/0). No BLOCK or WARN findings.

**External reviewers:**
[openai] o3 (high) -- 3423 in / 8393 out / 8256 reasoning -- ~$.1400
[openai] 3 findings (1 BLOCK, 1 WARN, 1 NOTE). BLOCK (PID reuse SIGTERM) downgraded to NOTE (4M PID space, single-user dev machine, warm.py finally-block cleanup). WARN (STATE_PATH fallback never fires) is a false positive (verified with shell tests). NOTE (GPU_MEMORY_UTILIZATION unused) is a false positive (used at warm.py:83).
[qwen] Qwen/Qwen2.5-Coder-14B-Instruct-AWQ -- 3544 in / 5 out -- 25s -- $0.00
[qwen] No issues found.

### Findings

```
[NOTE] (openai) gpu-release:46-55 -- PID reuse could SIGTERM an unrelated process
  Evidence: If warm.py is killed by SIGKILL/OOM-killer (bypassing the finally
  block that unlinks the state file), and the OS reuses the PID for a different
  process, gpu-release will send SIGTERM to the wrong target. The kill -0 check
  at line 50 confirms the PID is alive but not that it belongs to warm.py.
  Mitigants: 4M PID space (kernel.pid_max=4194304), single-user dev machine,
  warm.py's finally block cleans up state file on normal exit/SIGTERM/SIGINT.
  The failure requires SIGKILL or OOM-killer, then PID reuse before the next
  gpu-release call. Probability is negligible for the intended deployment.
  Suggested fix: If this ever matters, check /proc/$PID/cmdline for "warm.py".
```

```
[NOTE] (openai) gpu_lock.py:40 -- State file tmp path in predictable location
  Evidence: write_state() creates path + ".tmp" in XDG_RUNTIME_DIR or /tmp
  fallback. Same class as the prior lock file finding. XDG_RUNTIME_DIR is
  user-owned mode 0700. /tmp fallback uses user-specific directory. Single-user
  dev machine, no untrusted local users. Downgraded from external BLOCK per
  SECURITY.md local-CLI trust boundary.
```

```
[NOTE] (openai) gpu_lock.py:65 -- pid_alive() returns False on PermissionError
  Evidence: os.kill(pid, 0) raises PermissionError (subclass of OSError) for
  PIDs owned by other users. The except clause catches OSError, so pid_alive
  returns False (false negative). In practice, the state file only contains
  same-user PIDs (written by warm.py with os.getpid()). On a multi-user system,
  PID reuse across users would cause review.py to fall to cold path (safe).
```

```
[NOTE] (openai) gpu_lock.py:48 -- dict | None union syntax requires Python 3.10+
  Evidence: The return type annotation uses the PEP 604 union syntax introduced
  in Python 3.10. The target machine runs Python 3.10 per CLAUDE.md constraints.
  Not a compatibility issue for the intended deployment.
```

```
[NOTE] tests/test-review.sh:455-482 -- Test 13 fragile when warm server is running
  Evidence: Test 13 holds the GPU flock and expects review.py to fail with
  "GPU busy." If a warm server is running with a state file (state=ready),
  review.py uses the warm path and never reaches the flock code. test-warm.sh
  handles this with startup cleanup, but test-review.sh does not.
  Suggested fix: Add warm-server cleanup at the top of test-review.sh.
```

```
[NOTE] tests/test-warm.sh:5 -- Header comment inaccurate after Tests 6-9
  Evidence: Line 5 says "Tests 4-5 run without GPU." Test 4 starts a real
  warm server (requires GPU). Tests 6, 6b, 7, 8, 9 require GPU. Only Test 5
  runs without GPU.
  Suggested fix: "Requires GPU + model for Tests 1-4, 6-9. Test 5 runs without GPU."
```

```
[NOTE] tests/test-review.sh:218,230-231 -- Test 5 comments reference old fixed 4096 reserve
  Evidence: Comments say "output reserve (4096)" and "With a 1024 context window
  and 4096 output reserve." With adaptive reserve at LOCAL_MAX_MODEL_LEN=1024,
  the actual reserve is min(4096, max(256, 1024 // 4)) = 256.
  Suggested fix: Update comments to reflect adaptive reserve value.
```

### Fixes Applied

None.

### Accepted Risks

- SPEC.md done condition (A) was reworded post-completion (commit 3f1c01c) to remove the `LOCAL_INFERENCE_MODE=legacy` toggle requirement after the toggle was deleted. The rewording is documented in a post-completion note with rationale. Accepted: the toggle was speculative scaffolding, `git revert` is the actual rollback mechanism, and `tests/_harness.py` retains the baseline config for re-measurement.

---
*Prior review (2026-04-14): Refresh review of warm.py fixes. 0 BLOCK, 0 WARN, 6 NOTE. write_state moved after flock, shutdown guarded. Fast tests 22/22 + 12/12.*

<!-- REVIEW_META: {"date":"2026-04-16","commit":"8591da1","reviewed_up_to":"8591da1ad37a2189f5132b13b83ff94cf0a672de","base":"origin/main","tier":"refresh","block":0,"warn":0,"note":7} -->
