"""Stage 3: ask a question, get a cited answer.

    python ask.py "what did we decide about the Atlas rollback"
    python ask.py --show-prompt "..."      # print the prompt, contact no chat server
    python ask.py --bm25-only "..."        # skip the embedding server
    python ask.py --since 01.01.26 "..."   # your date formats work here
    python ask.py                          # interactive

Every claim in the answer carries a [n] marker; the numbers map to the source
list printed underneath, as path:line. Nothing here opens the vault.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from obsidian_rag import config, embed, generate, search, store


def print_sources(answer, show_all: bool) -> None:
    """The source list, with what the answer actually leaned on marked.

    Uncited sources are hidden by default. They are not evidence for anything
    the model said, and printing eight paths under a three-line answer trains
    you to stop reading the list -- which defeats the point of having one.
    """
    shown = [s for s in answer.sources if show_all or s.n in answer.cited]
    if not shown:
        return
    print("\nSources")
    for source in shown:
        mark = "*" if source.n in answer.cited else " "
        kind = "  (transcluded)" if source.kind == "transclusion" else ""
        print(f" {mark}[{source.n}] {source.citation}{kind}")
        print(f"      {source.breadcrumb}")
    hidden = len(answer.sources) - len(shown)
    if hidden:
        print(f"      ... and {hidden} retrieved but uncited (--all-sources to see)")


def print_warnings(answer) -> None:
    # stdout is buffered and stderr is not, so without this the warnings
    # overtake the answer they are about and appear above it.
    sys.stdout.flush()
    if answer.bogus:
        numbers = ", ".join(f"[{n}]" for n in answer.bogus)
        verb = "does" if len(answer.bogus) == 1 else "do"
        print(f"\n! The answer cites {numbers}, which {verb} not exist. "
              f"Only {len(answer.sources)} sources were provided -- treat the "
              f"whole answer with suspicion.", file=sys.stderr)
    if answer.text.strip() and not answer.cited:
        print("\n! The answer cites nothing. Either the model ignored the "
              "sources, or it is telling you the notes do not cover this.",
              file=sys.stderr)
    if answer.dropped:
        print(f"\n! {answer.dropped} retrieved chunk(s) did not fit the context "
              f"budget and were left out. Lower -k, or raise -c on the server.",
              file=sys.stderr)


def show_prompt(answer) -> None:
    for message in answer.messages:
        print(f"===== {message['role'].upper()} " + "=" * 50)
        print(message["content"])
    print("=" * 62)
    print(f"~{answer.prompt_tokens} prompt tokens, {len(answer.sources)} sources, "
          f"{answer.dropped} dropped")


def run(conn, searcher, args, question: str) -> None:
    started = time.time()
    hits = searcher.search(question, **search.options_from_args(args))
    retrieved = time.time() - started
    if not hits:
        print("Nothing retrieved -- no answer attempted.\n"
              "Try search.py --explain on the same question to see why.")
        return

    if args.show_prompt:
        result = generate.answer(conn, hits, question, send=False,
                                 max_tokens=args.max_tokens, n_ctx=args.n_ctx,
                                 expand_transclusions=not args.no_transclusions)
        show_prompt(result)
        return

    stream = config.CHAT_STREAM and not args.no_stream
    printer = (lambda piece: print(piece, end="", flush=True)) if stream else None

    result = generate.answer(
        conn, hits, question, stream=stream, on_token=printer,
        temperature=args.temperature, max_tokens=args.max_tokens,
        n_ctx=args.n_ctx, expand_transclusions=not args.no_transclusions)

    if not result.sources:
        print("Every retrieved chunk was too large for the context budget.")
        return
    if not stream:
        print(result.text)
    else:
        print()

    print_sources(result, args.all_sources)
    print_warnings(result)
    print(f"\n{len(hits)} chunks retrieved in {retrieved * 1000:.0f} ms, "
          f"~{result.prompt_tokens} prompt tokens, "
          f"{time.time() - started:.1f}s total")


def repl(conn, searcher, args) -> int:
    print("Ask a question, or Ctrl-C to quit.")
    while True:
        try:
            question = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        try:
            run(conn, searcher, args, question)
        except (search.SearchError, generate.ServerError) as exc:
            print(f"\n{exc}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Answer a question from the indexed notes, with citations.")
    p.add_argument("question", nargs="*", help="omit for interactive mode")
    search.add_retrieval_args(p)
    p.add_argument("--show-prompt", action="store_true",
                   help="print the assembled prompt and stop, contacting no chat server")
    p.add_argument("--all-sources", action="store_true",
                   help="list retrieved sources the answer did not cite")
    p.add_argument("--no-stream", action="store_true", help="wait for the whole answer")
    p.add_argument("--no-transclusions", action="store_true",
                   help="do not follow ![[embeds]] out of retrieved chunks")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--n-ctx", type=int, default=None,
                   help="override the server's context size when budgeting")
    args = p.parse_args()

    args.since = search.parse_date_arg(args.since, "since")
    args.until = search.parse_date_arg(args.until, "until")

    index_path = Path(args.index or config.INDEX_PATH)
    if not index_path.exists():
        print(f"No index at {index_path}. Run index.py first.", file=sys.stderr)
        return 2

    try:
        conn = store.connect(index_path)
        searcher = search.Searcher(conn)

        if not args.bm25_only and not searcher.has_vectors():
            print("! No vectors in this index -- retrieving by keyword only.\n"
                  "  Start the embedding server and re-run index.py for the hybrid.",
                  file=sys.stderr)
            args.bm25_only = True

        # Ask the server how much context it actually has, rather than
        # trusting config.CHAT_N_CTX to match how you launched it.
        if args.n_ctx is None and not args.show_prompt:
            args.n_ctx = generate.server_context_size()

        question = " ".join(args.question).strip()
        if not question:
            return repl(conn, searcher, args)
        run(conn, searcher, args, question)
        return 0
    except (store.SchemaError, search.SearchError, generate.ServerError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
