"""
LLM engine factory.

Keeps the choice of local inference engine in ONE place so every front-end
(generate.py, chat.py, app.py) is engine-agnostic. Swap engines by setting the
LLM_BACKEND environment variable -- no code changes needed.

    LLM_BACKEND=ollama    (default)  -> ChatOllama, talks to the Ollama server
    LLM_BACKEND=llamacpp             -> ChatLlamaCpp, loads the GGUF in-process

Everything is local/offline either way.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
GGUF_PATH = (
    HERE / "models" / "qwen3-coder-gguf"
    / "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
)

# --- Tunables (overridable via env) -----------------------------------------

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder")

# Low temperature: we want faithful, grounded explanations of real code,
# not creative prose. Bump toward 0.7 for chattier answers.
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Context window. THE #1 local-RAG footgun: engines default this small (~4096)
# and silently truncate your retrieved chunks. Keep it generous.
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "16384"))

# Cap generation length (mainly for llama.cpp, which needs an explicit bound).
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# llama.cpp only: how many layers to offload to GPU (-1 = as many as fit).
N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", "-1"))


def build_llm():
    """Return a LangChain chat model for the configured backend."""
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
        )

    if LLM_BACKEND == "llamacpp":
        # Requires: pip install llama-cpp-python
        from langchain_community.chat_models import ChatLlamaCpp

        if not GGUF_PATH.exists():
            raise FileNotFoundError(f"GGUF not found at {GGUF_PATH}")
        return ChatLlamaCpp(
            model_path=str(GGUF_PATH),
            temperature=TEMPERATURE,
            n_ctx=NUM_CTX,
            max_tokens=MAX_TOKENS,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )

    raise ValueError(
        f"Unknown LLM_BACKEND={LLM_BACKEND!r}. Use 'ollama' or 'llamacpp'."
    )
