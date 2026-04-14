## Security Review -- 2026-04-13 (scope: paths)

**Summary:** No security issues identified. review.py reads local files and passes content to vLLM offline inference via in-process LLM() or a local Unix domain socket (warm path). LLM output is printed as text, never executed. _harness.py reads hardcoded fixture paths and constructs vLLM configs with no external input. test-warm.sh uses properly quoted shell variables and communicates only with local processes. No secrets, no PII, no attacker-reachable code paths in the reviewed files.

### Findings

No security issues identified.

### Accepted Risks

(none)

---
*Prior review (2026-04-13): No security issues across five files (review.py, warm.py, gpu_lock.py, test-review.sh, test-warm.sh). Local-CLI trust boundary, no network surface, LLM output never executed.*

<!-- SECURITY_META: {"date":"2026-04-13","commit":"e208e451437604f32c386beae8d3dcb00d21c7b0","scope":"paths","scanned_files":["review.py","tests/_harness.py","tests/test-warm.sh"],"block":0,"warn":0,"note":0} -->
