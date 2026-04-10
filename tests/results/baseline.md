# Eval results: baseline

- model: `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
- max_model_len: 32768
- llm_kwargs: `{'gpu_memory_utilization': 0.9, 'enforce_eager': True}`
- sampling_kwargs: `{'temperature': 0.2, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05, 'max_tokens': 4096}`
- post-load used VRAM: 19.41 GB
- peak used VRAM: 19.52 GB

## 01-cmd-injection

_408 prompt / 173 gen tokens, 8.8s wall, used 19.52 GB_

```
[BLOCK] tools/runner.py:7 -- Potential security vulnerability: The command is constructed using string interpolation without validation, which could lead to command injection if `target` contains malicious input.

[BLOCK] tools/runner.py:14 -- Potential security vulnerability: The script does not validate the `target` argument, which could lead to path traversal if `target` is crafted to access arbitrary files.

[WARN] tools/runner.py:14 -- Missing error handling: If `sys.argv[1]` is not provided, the script will default to "all" without any indication that the default was used.

[NOTE] tools/runner.py:14 -- Consider using `argparse` for better argument parsing and help message generation.

[NOTE] tools/runner.py:14 -- Consider adding logging to track which build target was executed.
```

## 02-off-by-one

_483 prompt / 128 gen tokens, 7.9s wall, used 19.52 GB_

```
[WARN] src/pagination.py:17 -- The calculation of `start` should be `(page - 1) * page_size` instead of `page * page_size` to maintain 1-indexed page numbers.
[WARN] src/pagination.py:18 -- The calculation of `end` should be `start + page_size` instead of `start + page_size + 1` to correctly slice the list.
[NOTE] src/pagination.py:19 -- The use of `min(end, total)` is correct but could be simplified by ensuring `end` does not exceed the list length before slicing.
```

## 03-sampling-params

_425 prompt / 4 gen tokens, 0.6s wall, used 19.52 GB_

```
No issues found.
```

## 04-path-traversal

_498 prompt / 4 gen tokens, 0.7s wall, used 19.52 GB_

```
No issues found.
```
