#!/usr/bin/env bash
set -euo pipefail

# Idempotent setup for qwen-2.5-localreview.
# Creates venv, installs vLLM, downloads model, configures git hooks.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
VLLM_VERSION="0.19.0"

# --- Step 1: Create venv ---

if [[ -d "${VENV_DIR}" ]]; then
  echo "[ok] venv exists at ${VENV_DIR}"
else
  echo "[..] Creating venv..."
  python3 -m venv "${VENV_DIR}"
  echo "[ok] venv created"
fi

# --- Step 2: Install vLLM ---

if "${VENV_DIR}/bin/pip" show vllm >/dev/null 2>&1; then
  INSTALLED_VERSION=$("${VENV_DIR}/bin/pip" show vllm 2>/dev/null | grep -i '^Version:' | awk '{print $2}')
  echo "[ok] vLLM ${INSTALLED_VERSION} installed"
  if [[ "${INSTALLED_VERSION}" != "${VLLM_VERSION}" ]]; then
    echo "[!!] Pinned version is ${VLLM_VERSION}, installed is ${INSTALLED_VERSION}"
    echo "     Run: ${VENV_DIR}/bin/pip install vllm==${VLLM_VERSION} to update"
  fi
else
  echo "[..] Installing vLLM ${VLLM_VERSION} (this may take a few minutes)..."
  "${VENV_DIR}/bin/pip" install "vllm==${VLLM_VERSION}"
  echo "[ok] vLLM installed"
fi

# --- Step 3: Download model to shared HF cache ---
# Per ml-gpu.md: "Shared HF cache: ~/.cache/huggingface. Never override
# HF_HOME per-project; all projects share the same downloaded models."

HF_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--${MODEL//\//--}"
if [[ -d "${HF_CACHE_DIR}" ]]; then
  echo "[ok] Model ${MODEL} already in HF cache"
else
  echo "[..] Downloading ${MODEL} to shared HF cache (~8GB)..."
  "${VENV_DIR}/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}')
"
  echo "[ok] Model downloaded"
fi

# --- Step 4: Configure git hooks ---

if [[ "$(git -C "${SCRIPT_DIR}" config --get core.hooksPath 2>/dev/null)" == ".githooks" ]]; then
  echo "[ok] git hooks configured"
else
  git -C "${SCRIPT_DIR}" config core.hooksPath .githooks
  echo "[ok] git hooks configured"
fi

# --- Done ---

echo ""
echo "Setup complete."
echo "  venv:  ${VENV_DIR}"
echo "  model: ${MODEL} (in ~/.cache/huggingface)"
echo ""
echo "Next steps:"
echo "  1. Integrate with review-external.sh (see integration/integration-guide.md)"
echo "  2. Test: .venv/bin/python review.py --system <file> --input <file>"
