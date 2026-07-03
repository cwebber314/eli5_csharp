"""
LLM engine factory.

Keeps the model configuration in ONE place so every front-end (generate.py,
chat.py, app.py) stays model-agnostic. Generation runs locally through Ollama.
"""

import os

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder-7b")

# Low temperature: we want faithful, grounded explanations of real code,
# not creative prose. Bump toward 0.7 for chattier answers.
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Context window. THE #1 local-RAG footgun: Ollama defaults this small (~4096)
# and silently truncates your retrieved chunks. Keep it generous.
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "16384"))


def build_llm():
    """Return a LangChain chat model backed by the local Ollama server."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_ctx=NUM_CTX,
    )
