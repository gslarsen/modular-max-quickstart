# from max.entrypoints.llm import LLM
# from max.pipelines import PipelineConfig

import os
import importlib


# ➜ source .venv/quickstart/bin/activate
# if desired (otherwise will authorize from cache): export HUGGINGFACE_HUB_TOKEN="<your_token>"
# ➜ ./run_llama_cpu.sh in project root and it will set up the environment and execute this file
#
# else, to run non-Hugging Face gated model, set the model_path and run:
# ➜ python offline-inference.py
def main():
    # Specify the direct path to the GGUF file within the repo.
    # The 'gguf' format is automatically recognized.
    # Use the official repo for architecture/tokenizer; provide GGUF weights via WEIGHT_PATHS env.
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    """Modular's model path from online tutorial"""
    # model_path = "modularai/Llama-3.1-8B-Instruct-GGUF"

    # Set your Hugging Face token as an environment variable
    # Do NOT hard-code secrets. Ensure HF_TOKEN is set in your shell before running.
    # export HF_TOKEN=...   (or set HUGGINGFACEHUB_API_TOKEN)
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        print("HF token not set via env; relying on cached login (hf auth login).")
    # Allow overriding device/encoding via env vars. Defaults are CPU + q4_k for CPU compatibility.
    device = os.getenv("DEVICE", "cpu").strip().lower()  # 'cpu' or 'gpu'
    encoding_env = os.getenv("ENCODING", "").strip().lower()
    weight_paths_env = os.getenv("WEIGHT_PATHS", "").strip()

    # Import MAX components lazily to avoid static import resolution issues
    driver = importlib.import_module("max.driver")
    config_enums = importlib.import_module("max.pipelines.lib.config_enums")
    pipelines_config = importlib.import_module("max.pipelines.lib.config")
    entry_llm = importlib.import_module("max.entrypoints.llm")

    DeviceSpec = getattr(driver, "DeviceSpec")
    SupportedEncoding = getattr(config_enums, "SupportedEncoding")
    PipelineConfig = getattr(pipelines_config, "PipelineConfig")
    LLM = getattr(entry_llm, "LLM")

    # Build kwargs for PipelineConfig -> MAXModelConfig
    model_kwargs = {"model_path": model_path}

    # Optional explicit weight paths (comma-separated). Accepts local paths or
    # repo-prefixed paths like "org/repo/file.safetensors" or GGUF files.
    if weight_paths_env:
        model_kwargs["weight_path"] = [
            p.strip() for p in weight_paths_env.split(",") if p.strip()
        ]

    # Device selection
    if device == "gpu":
        model_kwargs["device_specs"] = [DeviceSpec.gpu()]
    else:
        model_kwargs["device_specs"] = [DeviceSpec.cpu()]

    # Encoding selection
    if encoding_env:
        # If user specifies, honor it explicitly
        try:
            model_kwargs["quantization_encoding"] = SupportedEncoding(encoding_env)
        except ValueError as e:
            raise ValueError(
                f"ENCODING env value '{encoding_env}' is not valid. Use one of: "
                f"{', '.join(v.value for v in SupportedEncoding)}"
            ) from e
    else:
        # If user didn't specify encoding, consider a safe default only when
        # no explicit weight files are provided (so MAX can infer from names when present).
        if device != "gpu" and not model_kwargs.get("weight_path"):
            # q4_k is a good default for CPU; alternatively: q4_0, q6_k, or float32
            model_kwargs["quantization_encoding"] = SupportedEncoding.q4_k

    try:
        pipeline_config = PipelineConfig(**model_kwargs)
    except ValueError as e:
        # Provide a clearer hint when users hit encoding/device incompatibilities or missing weights
        hint = (
            "\nHint: On CPU, use one of {q4_k, q4_0, q6_k, float32}. "
            "If your chosen Hugging Face repo doesn't host those weights (e.g., official bf16 safetensors), "
            "switch to a GGUF repo that includes Q4 weights or set weight_path to specific Q4 files in that repo.\n"
        )
        raise RuntimeError(f"Failed to build PipelineConfig: {e}{hint}") from e

    # *** Modular specific code for online tutorial from here on ***
    # pipeline_config = PipelineConfig(model_path=model_path)
    llm = LLM(pipeline_config)

    prompts = [
        "In the beginning, there was",
        "I believe the meaning of life is",
        "The fastest way to learn python is",
    ]

    print("Generating responses...")
    responses = llm.generate(prompts, max_new_tokens=50)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"========== Response {i} ==========")
        print(prompt + response)
        print()


if __name__ == "__main__":
    main()
