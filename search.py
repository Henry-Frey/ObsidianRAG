"""Stage 2: query the index.

    python search.py "how do we handle releases"
    python search.py --bm25-only "acme pricing"     # no llama-server needed
    python search.py --explain "rollback"           # both lists before fusion
    python search.py --since 01.01.26 "standup"     # your date formats work here
    python search.py                                # interactive, matrix loaded once

Nothing here opens the vault. Search reads the index file and nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from obsidian_rag import config, embed, search, store


def snippet(body: str, terms: list[str], width: int = 320) -> str:
    """A window of the body centred on the first query term that appears.

    Showing the first 300 characters of a chunk is a poor preview: heading
    chunks often open with context and the matching sentence is further down,
    so every result looks equally irrelevant.
    """
    flat = " ".join(body.split())
    low = flat.lower()
    pos = -1
    for term in terms:
        found = low.find(term)
        if found >= 0 and (pos < 0 or found < pos):
            pos = found
    if len(flat) <= width:
        return flat
    if pos < 0:
        return flat[:width] + " ..."
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    return ("... " if start else "") + flat[start:end] + (" ..." if end < len(flat) else "")


def render(hits, query: str, full: bool) -> None:
    if not hits:
        print("No matches.")
        return
    terms = search.query_terms(query)
    for i, hit in enumerate(hits, start=1):
        date = f"  [{hit.date}]" if hit.date else ""
        print(f"\n{i:2}. {hit.citation}{date}")
        print(f"    {hit.breadcrumb}")
        print(f"    score {hit.score:.4f}   {hit.why}")
        text = hit.body if full else snippet(hit.body, terms)
        for line in text.splitlines() if full else [text]:
            print(f"    | {line}")
        for extra in hit.context:
            label = extra["heading"] or f"chunk {extra['ord']}"
            body = extra["body"] if full else snippet(extra["body"], terms, 200)
            print(f"    + context ({label}): {body}")


def as_json(hits) -> str:
    return json.dumps([{
        "rank": i,
        "citation": hit.citation,
        "path": hit.path,
        "line": hit.start_line,
        "breadcrumb": hit.breadcrumb,
        "date": hit.date,
        "score": round(hit.score, 6),
        "ranks": hit.ranks,
        "boosted": hit.boosted,
        "body": hit.body,
        "context": [c["body"] for c in hit.context],
    } for i, hit in enumerate(hits, start=1)], indent=2, ensure_ascii=False)


def do_explain(searcher, query: str, candidates: int, args) -> None:
    """Print each retriever's list separately.

    This is the tool for deciding whether the weights are doing anything. If
    the two lists agree completely, the hybrid is buying you nothing; if they
    barely overlap, fusion is doing all the work.
    """
    out = searcher.explain(query, candidates,
                           use_vector=not args.bm25_only,
                           use_bm25=not args.vector_only)
    rows = out["rows"]
    print(f"FTS5 MATCH: {out['match']}\n")
    for name in ("bm25", "vector"):
        ids = out[name]
        print(f"--- {name} ({len(ids)} hits) " + "-" * 40)
        if not ids:
            print("    (nothing)")
        for rank, cid in enumerate(ids[:15], start=1):
            row = rows.get(cid)
            if row:
                print(f"  {rank:3}. {row['path']}:{row['start_line']}  {row['breadcrumb']}")
        print()
    both = set(out["bm25"]) & set(out["vector"])
    print(f"overlap: {len(both)} of {candidates} candidates found by both")


def run(searcher, args, query: str) -> None:
    started = time.time()
    hits = searcher.search(query, **search.options_from_args(args))
    if args.json:
        print(as_json(hits))
        return
    render(hits, query, args.full)
    plural = "" if len(hits) == 1 else "s"
    print(f"\n{len(hits)} result{plural} in {(time.time() - started) * 1000:.0f} ms")


def repl(searcher, args) -> int:
    print("Type a query, or Ctrl-C to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        try:
            run(searcher, args, query)
        except (search.SearchError, embed.ServerError) as exc:
            print(f"\n{exc}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description="Search the index (never reads the vault).")
    p.add_argument("query", nargs="*", help="the question; omit for interactive mode")
    search.add_retrieval_args(p)
    p.add_argument("--explain", action="store_true",
                   help="show both ranked lists before fusion, and their overlap")
    p.add_argument("--full", action="store_true", help="print whole chunks, not snippets")
    p.add_argument("--json", action="store_true", help="machine-readable output")
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
            print("! No vectors in this index -- falling back to keyword search.\n"
                  "  Start the embedding server and re-run index.py for the hybrid.",
                  file=sys.stderr)
            args.bm25_only = True

        query = " ".join(args.query).strip()
        if not query:
            return repl(searcher, args)
        if args.explain:
            do_explain(searcher, query, args.candidates or config.CANDIDATES, args)
            return 0
        run(searcher, args, query)
        return 0
    except (store.SchemaError, search.SearchError, embed.ServerError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
