# Modular MAX quickstart: Llama 3.2 1B Instruct (CPU/GPU)

Run Meta Llama 3.2 1B Instruct locally with MAX. On CPU, use GGUF Q4 weights; on GPU, use bf16 safetensors.

## Prereqs
- Python venv (this repo uses `./.venv/quickstart`).
- Hugging Face account + token and access to the gated model:
  - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (accept terms)
- Authenticate once (recommended):

```zsh
hf auth login
```

Or export your token for the session:

```zsh
export HUGGINGFACEHUB_API_TOKEN='hf_...'
# Optional aliases some libs use:
export HUGGINGFACE_HUB_TOKEN="$HUGGINGFACEHUB_API_TOKEN"
export HF_TOKEN="$HUGGINGFACEHUB_API_TOKEN"
```

## Files
- `offline-inference.py` — uses the official repo for architecture/tokenizer and lets you set `WEIGHT_PATHS` to GGUF or safetensors weights.

## Run on CPU with GGUF Q4
The official Meta repo ships bf16 safetensors (GPU-oriented). For CPU, point to a GGUF Q4 file from a community “-GGUF” repo.

```zsh
# 1) Ensure token is available (via hf auth login or env exports)
# export HUGGINGFACEHUB_API_TOKEN='hf_...'

# 2) Provide a GGUF Q4 file (example path from bartowski)
export WEIGHT_PATHS='bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf'

# 3) Let MAX infer encoding from the filename
unset ENCODING

# 4) Run with the venv Python
./.venv/quickstart/bin/python offline-inference.py
```

Notes:
- Keep `model_path` set to the official repo so MAX recognizes the architecture.
- If you see a bfloat16-on-CPU error, you’re loading GPU-only weights; use Q4/Q6/float32 for CPU.
- If you see an architecture-not-available error for a community GGUF repo, don’t switch `model_path`; only set `WEIGHT_PATHS`.

## Run on GPU with bf16 safetensors

```zsh
export DEVICE='gpu'
export ENCODING='bfloat16'
./.venv/quickstart/bin/python offline-inference.py
```

## Environment variables
- Auth: `HUGGINGFACEHUB_API_TOKEN` (preferred), `HUGGINGFACE_HUB_TOKEN`, or `HF_TOKEN`.
- Weights: `WEIGHT_PATHS` — comma-separated paths; accepts local paths or `org/repo/file` on the Hub.
- Device/encoding:
  - `DEVICE`: `cpu` (default) or `gpu`.
  - `ENCODING` (optional override). CPU: `q4_k`, `q4_0`, `q6_k`, `float32`. GPU: `bfloat16`, `float8_e4m3fn`.

## Troubleshooting
- 401 Unauthorized / gated repo: ensure terms are accepted and token is active (`hf auth login`).
- Quantization not supported by repo: your repo doesn’t host that encoding; provide GGUF Q4 in `WEIGHT_PATHS` and let MAX infer it.
- Architecture not available: keep the official repo in `model_path` and use `WEIGHT_PATHS` for external weights.
- Python 3.13 issues: if packages fail to install/run, try Python 3.10–3.12.

## Learn more
- HF Course: https://huggingface.co/learn
- Hub docs: https://huggingface.co/docs/hub/index
- Transformers: https://huggingface.co/docs/transformers/index
- GGUF / llama.cpp: https://github.com/ggerganov/llama.cpp#gguf
- safetensors: https://huggingface.co/docs/safetensors
- MAX docs: https://docs.modular.com/max/
