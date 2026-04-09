## Review -- 2026-04-09 (commit: c70996c)

**Summary:** Refresh review of uncommitted changes: VRAM preflight check, structured fatal error handler with OOM detection, memory tuning (gpu_memory_utilization 0.90, enforce_eager), new LOCAL_MAX_MODEL_LEN env var, and corresponding test and docs updates. All 20 tests pass.

**External reviewers:**
[openai] o3 (high) -- 1766 in / 4944 out / 4864 reasoning -- ~$.0819
[qwen] Qwen/Qwen2.5-Coder-14B-Instruct-AWQ -- 1825 in / 5 out -- 24s -- $0.00

### Findings

```
[NOTE] (openai) review.py:177 -- sys.exit(0) in fatal error handler masks failures from callers
  Evidence: The except handler on line 159-177 always exits 0 regardless of error type.
  Context: This is intentional fail-open design, documented in CLAUDE.md ("Fail-open:
  review.py exits 0 on all errors, warnings to stderr"). The consumer (call_local.sh)
  depends on this behavior. Errors are reported via structured stderr lines for diagnosis.
```

### Fixes Applied

None.

### Accepted Risks

None.

---
*Prior review (2026-04-09): Full review of all 5 initial commits. Found 2 WARNs (wrong directory name and hardcoded username in documentation paths). Both auto-fixed.*

<!-- REVIEW_META: {"date":"2026-04-09","commit":"c70996c","reviewed_up_to":"c70996c3fdd92a1f875eb96a656b06e5b575dcfe","base":"origin/main","tier":"refresh","block":0,"warn":0,"note":1} -->
