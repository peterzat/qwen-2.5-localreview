## Bench: stage4-14b-fp8

- model: `RedHatAI/Qwen2.5-Coder-14B-Instruct-FP8-dynamic`
- max_model_len: 8192
- llm_kwargs: `{'gpu_memory_utilization': 0.9, 'enforce_eager': True, 'kv_cache_dtype': 'fp8_e4m3'}`
- post-load used VRAM: 19.30 GB
- peak used VRAM: 19.30 GB

| fixture | prompt tok | gen tok | prefill TPS | decode TPS | used VRAM (GB) | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-cmd-injection | 408 | 105 | 1448.1 | 9.8 | 19.30 | 11.0 |
| 02-off-by-one | 483 | 127 | 1536.6 | 14.1 | 17.98 | 9.3 |
| 03-sampling-params | 425 | 4 | 2131.9 | 14.2 | 17.98 | 0.5 |
| 04-path-traversal | 498 | 4 | 2397.5 | 14.4 | 17.98 | 0.5 |
| **total** | **1814** | **240** | **1808.3** | **11.8** | **19.30** | **21.3** |
