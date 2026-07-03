"""
Retrieval step of the C# RAG pipeline.

Loads the Chroma store built by ingest.py, embeds a query with the same
offline bge-small model, and prints the nearest code chunks with their
similarity scores and source locations.

This is deliberately minimal and inspectable: the whole point is to *see*
what the vector search returns before any LLM is involved.

Usage:
    python retrieve.py "how do I compute an SVD"
    python retrieve.py "matrix multiplication" --k 8
"""

import argparse
from pathlib import Path

from langchain_chroma import Chroma

# Reuse the exact same embedder + paths as ingestion so query and document
# vectors live in the same space.
from ingest import CHROMA_DIR, COLLECTION_NAME, build_embedder


def retrieve(query: str, k: int) -> None:
    if not Path(CHROMA_DIR).exists():
        raise SystemExit(
            f"No Chroma store at {CHROMA_DIR}. Run ingest.py first."
        )

    embedder = build_embedder()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(CHROMA_DIR),
    )

    # similarity_search_with_relevance_scores returns (Document, score) pairs.
    # Score is normalized to [0, 1] where higher = more relevant.
    results = store.similarity_search_with_relevance_scores(query, k=k)

    if not results:
        print("No results. Is the store populated?")
        return

    print(f"\nQuery: {query!r}   (top {len(results)} of collection '{COLLECTION_NAME}')\n")
    print("=" * 80)
    for rank, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        idx = doc.metadata.get("chunk_index", "?")
        print(f"\n[{rank}] score={score:.4f}  {src}  (chunk {idx})")
        print("-" * 80)
        print(doc.page_content.strip())
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve C# chunks from Chroma.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument(
        "--k", type=int, default=4, help="Number of chunks to retrieve (default 4)."
    )
    args = parser.parse_args()
    retrieve(args.query, args.k)


if __name__ == "__main__":
    main()
