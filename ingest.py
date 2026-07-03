"""
Ingest a C# codebase into a local Chroma vector store using the offline
bge-small-en-v1.5 embedding model.

This is step 1 of a learning RAG pipeline: load -> chunk -> embed -> store.
Retrieval / generation (with the local Qwen3-Coder model) come later.

Usage:
    python ingest.py --source path/to/csharp/repo
    python ingest.py --source ./repos/mathnet-numerics --reset

Everything runs fully offline: the embedding model is read from ./models.
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# --- Configuration -----------------------------------------------------------

HERE = Path(__file__).resolve().parent

# Local, offline embedding model (already downloaded under ./models).
EMBED_MODEL_PATH = HERE / "models" / "bge-small-en-v1.5"

# Where Chroma persists its data on disk.
CHROMA_DIR = HERE / "chroma_db"
COLLECTION_NAME = "csharp_code"

# Chunking. C#-aware splitting keeps methods/classes together where possible.
# ~1000 chars with overlap is a reasonable starting point for code RAG.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# How many chunks to embed + write to Chroma per batch. Larger batches embed
# faster on multi-core CPUs; smaller batches give more frequent progress ticks.
BATCH_SIZE = 64

# Directories and files we never want to embed.
SKIP_DIRS = {"bin", "obj", ".git", ".vs", "packages", "node_modules", "TestResults"}
SKIP_FILE_SUFFIXES = (".Designer.cs", ".g.cs", ".g.i.cs", "AssemblyInfo.cs")


# --- Loading -----------------------------------------------------------------

def find_cs_files(source: Path) -> list[Path]:
    """Walk the source tree and collect .cs files worth embedding."""
    files: list[Path] = []
    for path in source.rglob("*.cs"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.endswith(SKIP_FILE_SUFFIXES):
            continue
        files.append(path)
    return files


def load_documents(source: Path) -> list[Document]:
    """Read each C# file into a Document with useful metadata."""
    docs: list[Document] = []
    for path in find_cs_files(source):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        rel = path.relative_to(source).as_posix()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": rel,
                    "filename": path.name,
                    "extension": ".cs",
                },
            )
        )
    return docs


# --- Chunking ----------------------------------------------------------------

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split files along C# syntactic boundaries (classes, methods, etc.)."""
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CSHARP,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    # Tag each chunk with its position within its source file.
    per_file_index: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = per_file_index.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        per_file_index[src] = idx + 1
    return chunks


# --- Embedding + storage -----------------------------------------------------

def build_embedder() -> HuggingFaceEmbeddings:
    """Load bge-small from the local folder.

    BGE recommends normalized embeddings, and a query-side instruction for
    retrieval. The deprecated HuggingFaceBgeEmbeddings did the instruction
    automatically; here we reproduce it explicitly via query_encode_kwargs'
    `prompt` (documents get no instruction, queries do -- so a store built by
    the old class stays compatible)."""
    if not EMBED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Embedding model not found at {EMBED_MODEL_PATH}")
    # Force offline so HF never tries to reach the network.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return HuggingFaceEmbeddings(
        model_name=str(EMBED_MODEL_PATH),
        model_kwargs={"device": "cpu"},
        # Documents: normalize, batch, no instruction.
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
        # Queries: same, plus BGE's retrieval instruction prompt.
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Represent this sentence for searching relevant passages: ",
        },
    )


def ingest(source: Path, reset: bool) -> None:
    print(f"Source repo : {source}")
    print(f"Chroma dir  : {CHROMA_DIR}")
    print(f"Embed model : {EMBED_MODEL_PATH}\n")

    if reset and CHROMA_DIR.exists():
        import shutil
        print("Resetting existing Chroma store...")
        shutil.rmtree(CHROMA_DIR)

    print("Loading C# files...")
    docs = load_documents(source)
    print(f"  {len(docs)} files loaded")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks produced")

    if not chunks:
        print("Nothing to ingest. Check the --source path.")
        return

    print("Loading embedding model (first call may take a moment)...")
    embedder = build_embedder()

    # Create an (empty) store, then add chunks in batches so we can show
    # progress. from_documents() would do it in one opaque blocking call.
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Embedding + writing to Chroma ({BATCH_SIZE} chunks/batch)...")
    batches = range(0, len(chunks), BATCH_SIZE)
    for start in tqdm(batches, total=len(batches), unit="batch", desc="Ingesting"):
        batch = chunks[start:start + BATCH_SIZE]
        store.add_documents(batch)

    print(f"\nDone. {len(chunks)} chunks stored in collection '{COLLECTION_NAME}'.")


# --- Quick sanity check ------------------------------------------------------

def sample_query(query: str) -> None:
    """Run a similarity search to confirm the store works."""
    embedder = build_embedder()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(CHROMA_DIR),
    )
    results = store.similarity_search(query, k=4)
    print(f"\nTop matches for: {query!r}\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata.get('source')} (chunk {doc.metadata.get('chunk_index')})")
        snippet = doc.page_content.strip().replace("\n", " ")[:160]
        print(f"    {snippet}...\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a C# repo into Chroma.")
    parser.add_argument(
        "--source",
        type=Path,
        default=HERE / "repos" / "mathnet-numerics",
        help="Path to the C# repository to ingest.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma store before ingesting.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Skip ingestion; run a test similarity search against the store.",
    )
    args = parser.parse_args()

    if args.query:
        sample_query(args.query)
        return

    ingest(args.source, args.reset)


if __name__ == "__main__":
    main()
