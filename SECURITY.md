## Security Review -- 2026-04-16 (scope: paths)

**Summary:** No security issues identified in gpu-release, warm.py, or tests/test-warm.sh. The new gpu-release script reads a PID from a state file in the user-owned XDG_RUNTIME_DIR and sends SIGTERM. No user-controlled input reaches shell commands or filesystem paths. warm.py changes removed dead VRAM polling code (net reduction in attack surface). All IPC remains via Unix domain socket with filesystem-permission access control. No secrets, no PII, no attacker-reachable code paths.

### Findings

No security issues identified.

### Accepted Risks

(none)

---
*Prior review (2026-04-14): No issues in warm.py and gpu_lock.py. Unix socket in user-owned XDG_RUNTIME_DIR, JSON-parsed requests with size bounds, LLM output never executed, atomic state file writes.*

<!-- SECURITY_META: {"date":"2026-04-16","commit":"8591da1ad37a2189f5132b13b83ff94cf0a672de","scope":"paths","scanned_files":["gpu-release","tests/test-warm.sh","warm.py"],"block":0,"warn":0,"note":0} -->
