"""Stage 4: measure whether retrieval actually works.

    python evaluate.py                          # run the default set
    python evaluate.py --set eval/mine.jsonl -v # per-case detail
    python evaluate.py --answer                 # also grade generated answers
    python evaluate.py --save runs/before.json
    python evaluate.py --baseline runs/before.json   # show what changed

WHY THIS EXISTS

Every other test in this project checks that the machinery is correct: that
RRF adds fractions properly, that bm25()'s sign is handled, that streaming
does not drop tokens. None of them can tell you whether the answers are any
good. Without that, tuning FTS_WEIGHTS or turning on --link-boost is guesswork
dressed up as engineering -- you change a number, the output looks different,
and you have no way to say whether it improved.

WHAT IS MEASURED, AND WHY THESE METRICS

  recall@k   Is the note that should have been found in the top k? This is the
             ceiling on everything else: if the note is not retrieved, no
             amount of prompt tuning will produce a correct answer. recall@1
             and recall@k matter for different reasons -- @k is "could the
             model possibly get this right", @1 is "did we put it first".

  MRR        The reciprocal of the rank of the first correct note, averaged.
             Distinguishes "scraped in at rank 8" from "nailed it at rank 1",
             which recall@8 alone cannot.

  per-retriever  The same set run through bm25-only, vector-only and the
             hybrid. This is the number that justifies -- or refutes -- the
             existence of half this codebase. If the hybrid does not beat both
             single retrievers, the fusion is not earning its complexity.

WHAT IS DELIBERATELY NOT MEASURED

Answer quality by LLM-as-judge. It needs the chat model for every case, it is
slow, and it is noisy enough that small real differences vanish into it. The
answer-side checks here are all deterministic: did it cite anything, did it
cite something that exists, did it cite the right note, and -- for questions
the vault genuinely cannot answer -- did it correctly decline.

That last one matters more than it looks. A personal notes assistant that
invents a plausible answer when you have not written the note is worse than
useless, because you will believe it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from obsidian_rag import config, embed, generate, search, store

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- the set
def load_cases(path: Path) -> list[dict]:
    """One JSON object per line. Blank lines and # comments are skipped.

    JSONL rather than one big JSON array so you can append a case with a
    single line, and a syntax error costs you one case instead of the file.

        {"q": "who owns kestrel cache", "expect": ["projects/Kestrel Cache.md"]}
        {"q": "what is my bank PIN", "expect": []}    <- unanswerable, must refuse
    """
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: {exc}")
        if "q" not in case:
            raise SystemExit(f"{path}:{number}: every case needs a 'q'")
        case.setdefault("expect", [])
        cases.append(case)
    return cases


def check_paths(conn, cases: list[dict]) -> list[str]:
    """An expected path that is not in the index is a typo, not a failure.

    Worth catching loudly: a mistyped path scores 0 forever and looks exactly
    like a retrieval problem you cannot fix.
    """
    known = {r["path"] for r in conn.execute("SELECT path FROM notes")}
    missing = []
    for case in cases:
        for path in case["expect"]:
            if path not in known:
                missing.append(f"{path!r}  (in: {case['q'][:50]})")
    return missing


# ------------------------------------------------------------------ metrics
def rank_of(hits, expected: list[str]) -> int | None:
    """1-based rank of the first hit from an expected note, or None."""
    wanted = set(expected)
    for position, hit in enumerate(hits, start=1):
        if hit.path in wanted:
            return position
    return None


class Scores:
    def __init__(self, label: str, cutoffs: tuple[int, ...]):
        self.label = label
        self.cutoffs = cutoffs
        self.ranks: list[int | None] = []
        self.elapsed = 0.0

    def add(self, rank: int | None) -> None:
        self.ranks.append(rank)

    def recall(self, k: int) -> float:
        if not self.ranks:
            return 0.0
        return sum(1 for r in self.ranks if r is not None and r <= k) / len(self.ranks)

    def mrr(self) -> float:
        if not self.ranks:
            return 0.0
        return sum(1 / r for r in self.ranks if r is not None) / len(self.ranks)

    def row(self) -> dict:
        out = {"label": self.label, "n": len(self.ranks), "mrr": round(self.mrr(), 4),
               "ms_per_query": round(self.elapsed * 1000 / max(len(self.ranks), 1), 1)}
        for k in self.cutoffs:
            out[f"r@{k}"] = round(self.recall(k), 4)
        return out


MODES = {
    "hybrid":      dict(use_bm25=True,  use_vector=True),
    "bm25 only":   dict(use_bm25=True,  use_vector=False),
    "vector only": dict(use_bm25=False, use_vector=True),
}


def run_retrieval(searcher, cases, options, modes, cutoffs, verbose):
    answerable = [c for c in cases if c["expect"]]
    results = {}
    for name in modes:
        scores = Scores(name, cutoffs)
        started = time.time()
        for case in answerable:
            hits = searcher.search(case["q"], **{**options, **MODES[name]})
            scores.add(rank_of(hits, case["expect"]))
        scores.elapsed = time.time() - started
        results[name] = scores

    if verbose and "hybrid" in results:
        print("\nper case (hybrid):")
        for case, rank in zip(answerable, results["hybrid"].ranks):
            mark = f"{rank:>4}" if rank else "MISS"
            print(f"  {mark}  {case['q'][:66]}")
            if rank is None:
                print(f"        expected {', '.join(case['expect'])}")
        print()
    return results, answerable


# ------------------------------------------------------------ answer checks
def run_answers(conn, searcher, cases, options, verbose):
    """Deterministic answer-side checks. Needs the chat server."""
    tally = {"asked": 0, "cited_something": 0, "bogus": 0,
             "expected_cited": 0, "answerable": 0, "declined_wrongly": 0,
             "refused_correctly": 0, "unanswerable": 0}
    for case in cases:
        hits = searcher.search(case["q"], **options)
        result = generate.answer(conn, hits, case["q"], stream=False)
        tally["asked"] += 1
        cited_paths = {s.path for s in result.sources if s.n in result.cited}
        if result.cited:
            tally["cited_something"] += 1
        if result.bogus:
            tally["bogus"] += 1

        if case["expect"]:
            tally["answerable"] += 1
            hit = bool(cited_paths & set(case["expect"]))
            tally["expected_cited"] += hit
            if not hit and not result.cited:
                # Declining a question the vault CAN answer is a retrieval
                # failure -- the right note never reached the model. Citing
                # the wrong note is a different bug with a different fix, so
                # they are counted apart.
                tally["declined_wrongly"] += 1
            if verbose and not hit:
                if result.cited:
                    print(f"  wrong source : {case['q'][:56]}")
                    print(f"      cited {sorted(cited_paths)}")
                else:
                    print(f"  declined     : {case['q'][:56]}")
                    print(f"      but {', '.join(case['expect'])} exists -- "
                          f"retrieval did not surface it")
        else:
            tally["unanswerable"] += 1
            # Nothing cited is the correct behaviour: the sources given to the
            # model do not support an answer, and it should say so.
            if not result.cited:
                tally["refused_correctly"] += 1
            elif verbose:
                print(f"  should have declined: {case['q'][:52]}")
                print(f"        answered: {result.text[:100]}")
    return tally


# -------------------------------------------------------------------- output
def print_table(results, cutoffs) -> None:
    heads = ["retriever", *[f"R@{k}" for k in cutoffs], "MRR", "ms"]
    widths = [13, *[6] * len(cutoffs), 6, 7]
    print("".join(h.ljust(w) for h, w in zip(heads, widths)))
    for scores in results.values():
        row = scores.row()
        cells = [scores.label,
                 *[f"{row[f'r@{k}']:.2f}" for k in cutoffs],
                 f"{row['mrr']:.2f}", f"{row['ms_per_query']:.1f}"]
        print("".join(c.ljust(w) for c, w in zip(cells, widths)))


def print_verdict(results, cutoffs) -> None:
    """The line the whole exercise exists to produce."""
    if len(results) < 3:
        return
    k = cutoffs[0]
    hybrid = results["hybrid"].recall(k)
    best_single = max(results["bm25 only"].recall(k), results["vector only"].recall(k))
    delta = hybrid - best_single
    if delta > 0.01:
        print(f"\nThe hybrid beats the better single retriever by "
              f"{delta:+.2f} R@{k}. Fusion is earning its keep.")
    elif delta < -0.01:
        print(f"\nThe hybrid is {delta:+.2f} R@{k} WORSE than the better single "
              f"retriever.\nCheck RRF_WEIGHTS and FTS_WEIGHTS -- one retriever is "
              f"dragging the other down.")
    else:
        print(f"\nThe hybrid matches the better single retriever at R@{k}. On this "
              f"set\nfusion is not adding anything -- try harder questions, or ones "
              f"phrased\ndifferently from how the notes are written.")


def print_delta(current: dict, baseline: dict) -> None:
    print("\nvs baseline:")
    for label, row in current.items():
        old = baseline.get(label)
        if not old:
            continue
        bits = []
        for key in row:
            if key in ("label", "n") or key not in old:
                continue
            change = row[key] - old[key]
            if abs(change) > 0.0001:
                bits.append(f"{key} {change:+.3f}")
        print(f"  {label:13} {', '.join(bits) if bits else 'unchanged'}")


# ---------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description="Measure retrieval and answer quality.")
    p.add_argument("--set", dest="path", default=str(ROOT / "eval" / "dummy-vault.jsonl"),
                   help="question set (JSONL)")
    search.add_retrieval_args(p)
    p.add_argument("--answer", action="store_true",
                   help="also grade generated answers (needs the chat server)")
    p.add_argument("--save", metavar="FILE", help="write the metrics as JSON")
    p.add_argument("--baseline", metavar="FILE", help="compare against a saved run")
    p.add_argument("-v", "--verbose", action="store_true", help="per-case detail")
    args = p.parse_args()

    args.since = search.parse_date_arg(args.since, "since")
    args.until = search.parse_date_arg(args.until, "until")

    case_path = Path(args.path)
    if not case_path.exists():
        print(f"No question set at {case_path}.\n"
              f"Copy eval/dummy-vault.jsonl and rewrite it for your own vault.",
              file=sys.stderr)
        return 2

    index_path = Path(args.index or config.INDEX_PATH)
    if not index_path.exists():
        print(f"No index at {index_path}. Run index.py first.", file=sys.stderr)
        return 2

    try:
        conn = store.connect(index_path)
        searcher = search.Searcher(conn)
        cases = load_cases(case_path)

        missing = check_paths(conn, cases)
        if missing:
            print("These expected paths are not in the index -- typos score zero "
                  "forever and look\nexactly like a retrieval failure:", file=sys.stderr)
            for line in missing:
                print(f"  {line}", file=sys.stderr)
            return 2

        modes = list(MODES)
        if not searcher.has_vectors():
            print("! No vectors in this index -- keyword only, so there is nothing "
                  "to fuse.\n  The comparison that makes this worth running needs "
                  "the embedding server.\n", file=sys.stderr)
            modes = ["bm25 only"]
        elif args.bm25_only:
            modes = ["bm25 only"]
        elif args.vector_only:
            modes = ["vector only"]

        options = search.options_from_args(args)
        top_k = options["top_k"] or config.TOP_K
        cutoffs = tuple(sorted({1, 3, top_k}))

        answerable = sum(1 for c in cases if c["expect"])
        print(f"{case_path.name}: {answerable} answerable, "
              f"{len(cases) - answerable} unanswerable, k={top_k}\n")

        results, _ = run_retrieval(searcher, cases, options, modes, cutoffs,
                                   args.verbose)
        print_table(results, cutoffs)
        print_verdict(results, cutoffs)

        rows = {label: s.row() for label, s in results.items()}

        if args.answer:
            print("\nanswers (deterministic checks only):")
            tally = run_answers(conn, searcher, cases, options, args.verbose)
            print(f"  cited at least one source : {tally['cited_something']}/{tally['asked']}")
            print(f"  invented a citation       : {tally['bogus']}/{tally['asked']}")
            if tally["answerable"]:
                print(f"  cited the expected note   : "
                      f"{tally['expected_cited']}/{tally['answerable']}")
                print(f"  declined a real question  : "
                      f"{tally['declined_wrongly']}/{tally['answerable']}"
                      f"   (retrieval missed it)")
            if tally["unanswerable"]:
                print(f"  correctly declined        : "
                      f"{tally['refused_correctly']}/{tally['unanswerable']}")
            rows["_answers"] = tally

        if args.baseline:
            print_delta(rows, json.loads(Path(args.baseline).read_text(encoding="utf-8")))

        if args.save:
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"\nsaved to {out}")
        return 0
    except (store.SchemaError, search.SearchError, generate.ServerError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
