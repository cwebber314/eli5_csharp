"""
Step 1 -- one-shot CLI.

Ask a single question, get one grounded answer streamed to the terminal, exit.
No conversation memory: each run is independent. Simplest possible front-end
over the RAG core, good for isolating retrieval + generation behavior.

Usage:
    python generate.py "how does the BFS algorithm work?"
    python generate.py "what does the Matrix class do?" --k 8
"""

import argparse
import sys

from rag import DEFAULT_K, stream_answer, unique_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one question of the C# codebase.")
    parser.add_argument("question", type=str, help="Your question (quote it).")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Chunks to retrieve (default {DEFAULT_K}).")
    args = parser.parse_args()

    # No history in one-shot mode, so history-aware rewriting is off.
    tokens, docs = stream_answer(args.question, history=[], k=args.k, history_aware=False)

    for token in tokens:
        sys.stdout.write(token)
        sys.stdout.flush()

    sources = unique_sources(docs)
    if sources:
        print("\n\nSources:")
        for src in sources:
            print(f"  - {src}")
    print()


if __name__ == "__main__":
    main()
