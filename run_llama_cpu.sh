#!/usr/bin/env zsh
set -euo pipefail

# Project runner for Llama 3.2 1B Instruct on CPU using GGUF Q4 weights.
# Adjust WEIGHT_PATHS if you switch to a different repo/file (e.g., Q4_0 or Q6_K).

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)

# GGUF weights (Q4_K_M → q4_k). Leave ENCODING unset; MAX infers from filename.
# Switch quant by editing WEIGHT_PATHS inside the script:
#   Q4_0: .../Llama-3.2-1B-Instruct-Q4_0.gguf
#   Q6_K: .../Llama-3.2-1B-Instruct-Q6_K.gguf
export WEIGHT_PATHS='bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
unset ENCODING

# Use venv Python; falls back to system python if venv missing.
VENV_PY="$SCRIPT_DIR/.venv/quickstart/bin/python"
if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
else
  echo "Warning: venv Python not found, using 'python' on PATH" >&2
  PY=python
fi

exec "$PY" "$SCRIPT_DIR/offline-inference.py"
