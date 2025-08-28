#!/usr/bin/env zsh
set -euo pipefail

# Project runner for Llama 3.2 1B Instruct on CPU using GGUF Q4 weights.
# Adjust WEIGHT_PATHS if you switch to a different repo/file (e.g., Q4_0 or Q6_K).

SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)

# Load .env if present (do not commit real tokens; .gitignore already excludes .env)
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
fi

# If no token in env, try the CLI token cache
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HUGGINGFACEHUB_API_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
# Map to other env names some libs use (only if not already set)
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACEHUB_API_TOKEN:-}}"

# Required: GGUF/safetensors weights. Keep ENCODING empty for GGUF so MAX can infer (e.g., Q4_K_M → q4_k)
: "${WEIGHT_PATHS:?WEIGHT_PATHS is required (path to GGUF/safetensors). Set in .env or export before running.}"

# Defaults (can be overridden in .env)
export DEVICE="${DEVICE:-cpu}"
export ENCODING="${ENCODING:-}"

# Use venv Python; falls back to system python if venv missing.
VENV_PY="$SCRIPT_DIR/.venv/quickstart/bin/python"
if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
else
  echo "Warning: venv Python not found, using 'python' on PATH" >&2
  PY=python
fi

exec "$PY" "$SCRIPT_DIR/offline-inference.py"
