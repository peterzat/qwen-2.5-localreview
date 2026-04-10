## Bench: baseline

- model: `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
- max_model_len: 32768
- llm_kwargs: `{'gpu_memory_utilization': 0.9, 'enforce_eager': True}`
- post-load used VRAM: 19.03 GB
- peak used VRAM: 19.03 GB

| fixture | prompt tok | gen tok | prefill TPS | decode TPS | used VRAM (GB) | wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01-cmd-injection | 408 | 173 | 1133.5 | 17.9 | 19.03 | 10.0 |
| 02-off-by-one | 483 | 128 | 1106.6 | 18.0 | 19.03 | 7.6 |
| 03-sampling-params | 425 | 4 | 1050.4 | 18.2 | 19.03 | 0.6 |
| 04-path-traversal | 498 | 4 | 1164.5 | 13.9 | 19.03 | 0.7 |
| **total** | **1814** | **309** | **1113.8** | **17.8** | **19.03** | **18.9** |
