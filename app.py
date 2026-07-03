"""
Step 4 -- Gradio web chat UI, with a source-code viewer.

Layout: chat on the left, a source panel on the right. After each answer, the
files that were retrieved to ground it appear in a dropdown; selecting one shows
that file with syntax highlighting + line numbers, and reports which lines the
retrieved chunk covers. This makes the RAG story tangible: "here's the answer,
and here's exactly the code it came from."

Deliberately scoped for a demo: it shows the *cited* sources, not a full repo
file tree. Runs fully locally.

Usage:
    python app.py
    # then open the printed http://127.0.0.1:7860 URL

Finding files on disk: chunks store a repo-relative `source` path. We locate the
real file by matching that suffix under ./repos. If your repo lives elsewhere,
set SOURCE_ROOT to the same path you passed to `ingest.py --source`.
"""

import os
from pathlib import Path

import gradio as gr

from rag import DEFAULT_K, stream_answer, unique_sources

HERE = Path(__file__).resolve().parent
REPOS_DIR = HERE / "repos"

# gr.Code has no "csharp" language; "cpp" highlights C# well (same C-family
# keywords, strings, // and /* */ comments, braces).
CODE_LANGUAGE = "cpp"

TITLE = "Explain it to me like I'm 5 — C# edition"
DESCRIPTION = (
    "Ask questions about the ingested C# codebase. Answers are grounded in "
    "retrieved code chunks; click a cited source on the right to read it."
)

_path_cache: dict[str, Path | None] = {}


# --- Locating source files on disk ------------------------------------------

def resolve_source_path(source_rel: str) -> Path | None:
    """Map a stored repo-relative path to an actual file under ./repos.

    Honors SOURCE_ROOT if set, otherwise matches the path suffix under ./repos
    so it works regardless of which repo/subdir was ingested."""
    if source_rel in _path_cache:
        return _path_cache[source_rel]

    result: Path | None = None
    posix = source_rel.replace("\\", "/")

    env_root = os.environ.get("SOURCE_ROOT")
    if env_root:
        candidate = Path(env_root) / source_rel
        if candidate.exists():
            result = candidate

    if result is None and REPOS_DIR.exists():
        for candidate in REPOS_DIR.rglob(Path(source_rel).name):
            if candidate.as_posix().endswith(posix):
                result = candidate
                break

    _path_cache[source_rel] = result
    return result


def _chunk_line_span(text: str, chunk: str) -> tuple[int, int] | None:
    """Find where a retrieved chunk sits in the file, as (start, end) lines."""
    needle = chunk.strip()[:200]
    if not needle:
        return None
    idx = text.find(needle)
    if idx < 0:
        return None
    start = text.count("\n", 0, idx) + 1
    end = start + chunk.strip().count("\n")
    return start, end


def read_source(source_rel: str, docs) -> tuple[str, str]:
    """Return (file_text, info_line) for the given cited source."""
    path = resolve_source_path(source_rel)
    if path is None:
        msg = (
            f"// Could not locate '{source_rel}' under ./repos.\n"
            f"// Set the SOURCE_ROOT env var to your ingested repo root."
        )
        return msg, f"⚠️ {source_rel} not found on disk"

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    spans = []
    for d in docs or []:
        if d.metadata.get("source") == source_rel:
            span = _chunk_line_span(text, d.page_content)
            if span:
                spans.append(span)

    if spans:
        parts = ", ".join(f"lines {a}–{b}" for a, b in sorted(set(spans)))
        info = f"**{source_rel}** — retrieved chunk(s) at {parts}"
    else:
        info = f"**{source_rel}**"
    return text, info


# --- Chat plumbing -----------------------------------------------------------

def _text(content) -> str:
    """Gradio 6 message content may be a string, a list of parts, or a dict.
    Flatten it to plain text (used for building history turns)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("value") or ""))
        return " ".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or "")
    return str(content)


def _to_turns(history) -> list[tuple[str, str]]:
    """Convert Gradio 'messages' history into (user, assistant) turns."""
    turns: list[tuple[str, str]] = []
    pending_user = None
    for msg in history:
        role, content = msg["role"], _text(msg["content"])
        if role == "user":
            pending_user = content
        elif role == "assistant":
            turns.append((pending_user or "", content))
            pending_user = None
    return turns


def on_submit(message, history):
    """Append the user's message, stash the raw question, lock the input.

    We keep the raw textbox string (a guaranteed str) and feed *that* to the
    RAG core, rather than reading it back out of the chatbot -- Gradio 6 may
    hand message content back as a list, which the embedder can't consume."""
    if not message.strip():
        return history, gr.update(), ""
    history = history + [{"role": "user", "content": message}]
    return history, gr.update(value="", interactive=False), message


def on_respond(history, question):
    """Stream the assistant's answer; expose retrieved docs + cited sources."""
    if not question or not question.strip():
        yield history, [], gr.update()
        return

    turns = _to_turns(history[:-1])
    tokens, docs = stream_answer(question, history=turns, k=DEFAULT_K)

    sources = unique_sources(docs)
    dropdown = gr.update(choices=sources, value=(sources[0] if sources else None))

    history = history + [{"role": "assistant", "content": ""}]
    for token in tokens:
        history[-1]["content"] += token
        yield history, docs, dropdown


def load_code(source, docs):
    """Load a cited source into the code viewer + info line."""
    if not source:
        return gr.update(value=""), ""
    text, info = read_source(source, docs)
    return gr.update(value=text), info


def do_clear():
    return [], [], gr.update(choices=[], value=None), gr.update(value=""), ""


# --- UI ----------------------------------------------------------------------

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    docs_state = gr.State([])
    pending_q = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=560)
            msg = gr.Textbox(placeholder="Ask about the C# code…", show_label=False)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

        with gr.Column(scale=1):
            sources_dd = gr.Dropdown(
                label="Cited sources (this answer)", choices=[], interactive=True
            )
            chunk_info = gr.Markdown()
            code_view = gr.Code(
                label="Source",
                language=CODE_LANGUAGE,
                lines=28,
                show_line_numbers=True,
            )

    for trigger in (msg.submit, send.click):
        (
            trigger(on_submit, [msg, chatbot], [chatbot, msg, pending_q])
            .then(on_respond, [chatbot, pending_q], [chatbot, docs_state, sources_dd])
            .then(load_code, [sources_dd, docs_state], [code_view, chunk_info])
            .then(lambda: gr.update(interactive=True), None, [msg])
        )

    # Clicking a different cited source swaps the viewer.
    sources_dd.change(load_code, [sources_dd, docs_state], [code_view, chunk_info])

    clear.click(
        do_clear, None, [chatbot, docs_state, sources_dd, code_view, chunk_info]
    )


if __name__ == "__main__":
    demo.launch()
