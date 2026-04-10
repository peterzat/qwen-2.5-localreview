# Eval results: stage4-14b-fp8

- model: `RedHatAI/Qwen2.5-Coder-14B-Instruct-FP8-dynamic`
- max_model_len: 8192
- llm_kwargs: `{'gpu_memory_utilization': 0.9, 'enforce_eager': True, 'kv_cache_dtype': 'fp8_e4m3'}`
- sampling_kwargs: `{'temperature': 0.2, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05, 'max_tokens': 4096}`
- post-load used VRAM: 17.97 GB
- peak used VRAM: 17.98 GB

## 01-cmd-injection

_408 prompt / 105 gen tokens, 6.9s wall, used 17.98 GB_

```
[BLOCK] tools/runner.py:7 -- Potential command injection vulnerability due to user-supplied input in `target`.

[BLOCK] tools/runner.py:14 -- Missing error handling for `os.system` which can mask failures.

[WARN] tools/runner.py:11 -- Hardcoded path to `build.sh` may not be portable across different environments.

[NOTE] tools/runner.py:13 -- Consider using `subprocess.run` instead of `os.system` for better control and security.
```

## 02-off-by-one

_483 prompt / 127 gen tokens, 8.4s wall, used 17.98 GB_

```
BLOCK src/pagination.py:17 -- Off-by-one error in calculating the end index, which can lead to incorrect slicing.
BLOCK src/pagination.py:18 -- Incorrect conversion to 0-indexed offset, which will cause the function to skip the first item on the first page.
WARN src/pagination.py:19 -- Unnecessary calculation of `total` since `len(items)` is called inside the slice operation, which can be inefficient for large lists.
NOTE src/pagination.py:10 -- Consider adding a check for `page_size` being too large relative to the number of items to avoid unnecessary processing.
```

## 03-sampling-params

_425 prompt / 4 gen tokens, 0.5s wall, used 17.98 GB_

```
No issues found.
```

## 04-path-traversal

_498 prompt / 4 gen tokens, 0.5s wall, used 17.98 GB_

```
No issues found.
```
