"""
Shared RAG core: retrieve -> prompt -> answer.

Every front-end (one-shot CLI, REPL, Gradio) calls into this module, so the
actual retrieval + prompting logic lives in exactly one place. This is the
"answer(question, history)" seam we designed toward.

Includes the history-aware query rewriting used for multi-turn follow-ups.
"""

from typing import Iterator, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Reuse the exact embedder + store config from ingestion so query and document
# vectors share one space.
from ingest import CHROMA_DIR, COLLECTION_NAME, build_embedder
from llm import build_llm

DEFAULT_K = 5

# A conversation turn is a (user_text, assistant_text) pair.
Turn = Tuple[str, str]


# --- Prompts -----------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly C# tutor for an "explain it like I'm 5" \
project. Your job is to explain real C# code from the provided context in \
plain, simple language.

Rules:
- Answer ONLY using the provided code context. If the context does not contain \
the answer, say so plainly -- never invent code or APIs.
- Cite the source file(s) you used inline, like [src/Foo/Bar.cs].
- Prefer short paragraphs, simple words, and everyday analogies over jargon.
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}\n\nCode context:\n{context}"),
])

# Step 3: rewrite a follow-up into a standalone query BEFORE retrieval, so
# "does it handle nulls?" becomes something the vector search can actually use.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation so far and a follow-up question, rewrite the "
     "follow-up as a standalone question that includes any context needed to "
     "understand it on its own. Return ONLY the rewritten question with no "
     "preamble. If it is already standalone, return it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# --- Lazy singletons (avoid reloading the embedder / model per call) ---------

_store = None
_llm = None


def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=build_embedder(),
            persist_directory=str(CHROMA_DIR),
        )
    return _store


def get_llm():
    global _llm
    if _llm is None:
        _llm = build_llm()
    return _llm


# --- Helpers -----------------------------------------------------------------

def to_messages(history: List[Turn]) -> List[BaseMessage]:
    """Convert (user, assistant) turns into LangChain messages."""
    msgs: List[BaseMessage] = []
    for user, assistant in history:
        if user:
            msgs.append(HumanMessage(user))
        if assistant:
            msgs.append(AIMessage(assistant))
    return msgs


def format_context(docs: List[Document]) -> str:
    """Render retrieved chunks with their source paths for the prompt."""
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        idx = d.metadata.get("chunk_index", "?")
        blocks.append(f"// File: {src} (chunk {idx})\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def unique_sources(docs: List[Document]) -> List[str]:
    """Deduplicated list of source files, preserving retrieval order."""
    seen: List[str] = []
    for d in docs:
        src = d.metadata.get("source")
        if src and src not in seen:
            seen.append(src)
    return seen


def rewrite_query(question: str, history: List[Turn]) -> str:
    """History-aware query condensation. No-op when there's no history."""
    if not history:
        return question
    prompt = CONDENSE_PROMPT.invoke({
        "history": to_messages(history),
        "question": question,
    })
    return get_llm().invoke(prompt).content.strip()


# --- Core: retrieve + build the answer prompt --------------------------------

def _prepare(question: str, history: List[Turn], k: int, history_aware: bool):
    """Shared work for both streaming and non-streaming answers."""
    search_query = (
        rewrite_query(question, history) if history_aware else question
    )
    docs = get_store().similarity_search(search_query, k=k)
    prompt = ANSWER_PROMPT.invoke({
        "history": to_messages(history),
        "question": question,
        "context": format_context(docs),
    })
    return prompt, docs, search_query


def answer(
    question: str,
    history: List[Turn] | None = None,
    k: int = DEFAULT_K,
    history_aware: bool = True,
) -> Tuple[str, List[Document]]:
    """Non-streaming: return (answer_text, retrieved_docs)."""
    history = history or []
    prompt, docs, _ = _prepare(question, history, k, history_aware)
    text = get_llm().invoke(prompt).content
    return text, docs


def stream_answer(
    question: str,
    history: List[Turn] | None = None,
    k: int = DEFAULT_K,
    history_aware: bool = True,
) -> Tuple[Iterator[str], List[Document]]:
    """Streaming: return (token_generator, retrieved_docs).

    Docs are resolved up front (retrieval happens before generation), so
    callers can show citations immediately and stream the text as it arrives.
    """
    history = history or []
    prompt, docs, _ = _prepare(question, history, k, history_aware)
    llm = get_llm()

    def tokens() -> Iterator[str]:
        for chunk in llm.stream(prompt):
            yield chunk.content

    return tokens(), docs
