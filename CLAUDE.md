# qwen-2.5-localreview

Local adversarial code reviewer using Qwen2.5-Coder-14B-Instruct-AWQ via vLLM offline inference.

## Design Couplings

This project is tightly coupled to:

- **zat.env's usage patterns**: adversarial code review via `review-external.sh`, fail-open provider model, `(qwen)` finding tags, stdin diff / stdout findings / stderr status contract.
- **Target machine configuration**: RTX 4000 SFF Ada (20GB VRAM, 70W TDP), CUDA 13.x, 64GB DDR4, fast NVMe, Python 3.10, Ubuntu Linux.

It may be portable to similar setups but those are the intentional design constraints, not accidents.

## Local-Repo Contract

This project lives at `~/src/qwen-2.5-localreview`. Consumer projects (zat.env, any downstream user) reference the repo by absolute path. There is no install step beyond `setup.sh`, no packaging, no versioning scheme. Updates are manual: `cd ~/src/qwen-2.5-localreview && git pull`. The config in `~/.config/claude-reviewers/.env` points directly to files in this repo (`review.py`, `.venv`). These paths are stable across updates.

## Boundary with zat.env

This project never modifies zat.env. All integration (changes to `review-external.sh`, test updates, lint checks, `.env` configuration) is developed, integrated, and tested from within zat.env itself. This project provides the inference script and integration guide; zat.env owns the consumer side.

This includes `~/.config/claude-reviewers/.env`, which is created and managed by zat.env's install process. This project must not read, write, or append to that file. The integration guide documents what entries zat.env should add.

## Conventions

- Python venv at `.venv/`. Never pip install outside the venv.
- Model downloaded to shared HuggingFace cache (`~/.cache/huggingface`) per ml-gpu.md convention. Never override `HF_HOME`.
- vLLM pinned to a specific version in `setup.sh`. Update manually when needed.
- All inference is pure offline (vLLM `LLM` class). No network server, no TCP/IP, no ports. Optional local Unix domain socket for keep-warm mode (`warm.py`).
- GPU mutex via flock (`$XDG_RUNTIME_DIR/qwen-localreview.lock`). Concurrent invocations serialize automatically.
- Fail-open: review.py exits 0 on all errors, warnings to stderr.
