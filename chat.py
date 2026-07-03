"""
Step 2 -- terminal REPL with session memory.
Step 3 -- history-aware retrieval for follow-up questions.

A real chat loop in the terminal: it remembers the current conversation (in
RAM, lost on exit) and, by default, rewrites follow-up questions into
standalone queries before retrieving.

Toggle the rewriting off to *feel* what step 3 buys you: ask a follow-up like
"does it handle nulls?" with and without --no-history-aware and compare which
code gets retrieved.

Usage:
    python chat.py
    python chat.py --k 8
    python chat.py --no-history-aware     # disable follow-up query rewriting
"""

import argparse
import sys

from rag import DEFAULT_K, Turn, stream_answer, unique_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the C# codebase.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Chunks to retrieve (default {DEFAULT_K}).")
    parser.add_argument("--no-history-aware", action="store_true",
                        help="Disable follow-up query rewriting (step 3).")
    args = parser.parse_args()
    history_aware = not args.no_history_aware

    print("Chatting with your C# codebase.")
    print(f"  history-aware retrieval: {'on' if history_aware else 'off'}")
    print("  type 'exit' or Ctrl-C to quit.\n")

    history: list[Turn] = []

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        tokens, docs = stream_answer(
            question, history=history, k=args.k, history_aware=history_aware
        )

        sys.stdout.write("bot> ")
        answer_text = ""
        for token in tokens:
            sys.stdout.write(token)
            sys.stdout.flush()
            answer_text += token
        print()

        sources = unique_sources(docs)
        if sources:
            print("     sources: " + ", ".join(sources))
        print()

        # Remember this turn so the next question has context.
        history.append((question, answer_text))


if __name__ == "__main__":
    main()
