#!/usr/bin/env bash
# Create the isolated conda environment "fastvlm" and install dependencies.
set -euo pipefail

ENV_NAME="fastvlm"
PY_VERSION="3.11"

source /home/play/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

# CPU-only PyTorch
conda run -n "${ENV_NAME}" python -m pip install \
  torch --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies (transformers pinned for the model's remote code)
conda run -n "${ENV_NAME}" python -m pip install \
  "numpy<2" \
  "transformers==4.39.3" \
  timm \
  accelerate \
  peft \
  bitsandbytes \
  pillow \
  safetensors \
  sentencepiece

echo "En/home/play/build/forward/原始系统提示词/核心.mdvironment '${ENV_NAME}' is ready."
