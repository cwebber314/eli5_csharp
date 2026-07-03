"""
Step 4 -- Gradio web chat UI.

A browser-based chat over the same RAG core, with session memory and
history-aware retrieval. Runs fully locally.

Usage:
    python app.py
    # then open the printed http://127.0.0.1:7860 URL

Uses the OpenAI-style "messages" history format from Gradio's ChatInterface,
which we convert into the (user, assistant) turns the RAG core expects.
"""

import gradio as gr

from rag import DEFAULT_K, stream_answer, unique_sources

# The C# repo you ingested -- shown in the header for context.
TITLE = "Explain it to me like I'm 5 — C# edition"
DESCRIPTION = (
    "Ask questions about the ingested C# codebase. Answers are grounded in "
    "retrieved code chunks and cite their source files."
)


def _to_turns(history) -> list[tuple[str, str]]:
    """Convert Gradio 'messages' history into (user, assistant) turns."""
    turns: list[tuple[str, str]] = []
    pending_user = None
    for msg in history:
        role, content = msg["role"], msg["content"]
        if role == "user":
            pending_user = content
        elif role == "assistant":
            turns.append((pending_user or "", content))
            pending_user = None
    return turns


def respond(message, history):
    """Streaming response function for Gradio's ChatInterface."""
    turns = _to_turns(history)
    tokens, docs = stream_answer(message, history=turns, k=DEFAULT_K, history_aware=True)

    partial = ""
    for token in tokens:
        partial += token
        yield partial

    sources = unique_sources(docs)
    if sources:
        partial += "\n\n---\n*Sources: " + ", ".join(sources) + "*"
        yield partial


demo = gr.ChatInterface(
    fn=respond,
    title=TITLE,
    description=DESCRIPTION,
    examples=[
        "What does this codebase do?",
        "Explain the main class like I'm five.",
        "How is a graph traversal implemented?",
    ],
)


if __name__ == "__main__":
    demo.launch()
