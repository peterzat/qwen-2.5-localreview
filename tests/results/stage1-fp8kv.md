# Eval results: stage1-fp8kv

- model: `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
- max_model_len: 32768
- llm_kwargs: `{'gpu_memory_utilization': 0.9, 'enforce_eager': True, 'kv_cache_dtype': 'fp8_e4m3'}`
- sampling_kwargs: `{'temperature': 0.2, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05, 'max_tokens': 4096}`
- post-load used VRAM: 18.12 GB
- peak used VRAM: 18.12 GB

## 01-cmd-injection

_408 prompt / 177 gen tokens, 6.5s wall, used 18.12 GB_

```
[BLOCK] tools/runner.py:7 -- Potential security vulnerability: The command is constructed using string interpolation without validation, which could lead to command injection if `target` contains malicious input.

[BLOCK] tools/runner.py:14 -- Potential security vulnerability: The script does not validate the `target` argument, which could lead to path traversal if `target` contains unexpected values.

[WARN] tools/runner.py:14 -- Missing error handling: If `sys.argv[1]` is not provided, the script will default to "all" without any indication that the default was used.

[NOTE] tools/runner.py:14 -- Consider using `argparse` for better argument parsing and help message generation.

[NOTE] tools/runner.py:14 -- Consider adding logging to track which build target was selected and the result of the build process.
```

## 02-off-by-one

_483 prompt / 202 gen tokens, 7.5s wall, used 18.12 GB_

```
[BLOCK] src/pagination.py:17 -- The calculation of `start` is incorrect. It should be `(page - 1) * page_size` instead of `page * page_size` to maintain 1-indexed page numbers.
[BLOCK] src/pagination.py:18 -- The calculation of `end` is incorrect. It should be `start + page_size` instead of `start + page_size + 1` to ensure the correct slice size.
[WARN] src/pagination.py:19 -- The use of `min(end, total)` is necessary to prevent index out of range errors, but it should be accompanied by a check to ensure that `start` is not greater than `total`.
[NOTE] src/pagination.py:14 -- The comment "Convert to 0-indexed offset." is misleading because the function still uses 1-indexed page numbers. Consider renaming the variable `page` to `page_1_indexed` for clarity.
```

## 03-sampling-params

_425 prompt / 4 gen tokens, 0.4s wall, used 18.12 GB_

```
No issues found.
```

## 04-path-traversal

_498 prompt / 4 gen tokens, 0.5s wall, used 18.12 GB_

```
No issues found.
```
