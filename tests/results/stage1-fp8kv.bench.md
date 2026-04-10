## Bench: stage1-fp8kv

- model: `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
- max_model_len: 32768
- llm_kwargs: `{'gpu_memory_utilization': 0.9, 'enforce_eager': True, 'kv_cache_dtype': 'fp8_e4m3'}`
- post-load used VRAM: 18.12 GB
- peak used VRAM: 18.12 GB

| fixture | prompt tok | gen tok | prefill TPS | decode TPS | used VRAM (GB) | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-cmd-injection | 408 | 177 | 1536.8 | 28.3 | 18.12 | 6.5 |
| 02-off-by-one | 483 | 202 | 1529.1 | 28.2 | 18.12 | 7.5 |
| 03-sampling-params | 425 | 4 | 1501.5 | 26.1 | 18.12 | 0.4 |
| 04-path-traversal | 498 | 4 | 1502.2 | 25.7 | 18.12 | 0.5 |
| **total** | **1814** | **387** | **1516.8** | **28.2** | **18.12** | **14.9** |
