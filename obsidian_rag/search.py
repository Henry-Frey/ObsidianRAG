"""Stage 2: hybrid retrieval -- BM25 and vector search, fused by rank.

WHY TWO RETRIEVERS

They fail in opposite directions, and your vault sits exactly on the fault line.

    BM25    finds exact strings. Proper nouns, surnames, version numbers,
            error codes, the one German word in an English note. It is
            helpless when you ask about "rolling back a deploy" and the note
            says "reverting a release" -- no shared token, no match, no hit.

    Vector  finds paraphrase, and handles that query easily. It misses the
            first case, because "Kubernetes 1.29" and "Kubernetes 1.31" embed
            almost identically: the two tokens you care about are, to the
            model, noise.

Running only one of these is a mistake in a vault full of jargon and names.

HOW THE FUSION WORKS

Reciprocal Rank Fusion. Each retriever returns a ranked list; a chunk's score
is the sum over lists of  weight / (RRF_K + rank).

The alternative -- normalise both score scales and add them -- is worse here,
for two concrete reasons:

  * The scales are not comparable and not stable. Cosine similarity from
    BGE-M3 lives in a narrow band (related text clusters around 0.6-0.9), while
    BM25 is unbounded and depends on corpus statistics that shift every time
    you add notes. Min-max normalising either one makes the fused score of your
    best result depend on the score of the worst result in the list.

  * Ranks are comparable by construction. "Third best thing BM25 found" means
    the same as "third best thing the embedder found", whatever the units.

RRF also sidesteps a trap that has bitten most people who write this code by
hand: SQLite's bm25() returns a NEGATIVE number, and more negative means a
better match. Ranks have no sign, so fusion never has to know.

WHAT THIS MODULE DOES NOT TOUCH

The vault. Search reads only the index file. The read-only guarantee is not
even in play here -- there is no filesystem access at all.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from . import config, embed, store


class SearchError(RuntimeError):
    pass


# --------------------------------------------------------------- query terms
# Letters and digits, underscore excluded, allowing internal apostrophes so
# "o'brien" stays one term. Plain \w would swallow underscores.
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def query_terms(text: str, limit: int = 32) -> list[str]:
    """Lower-cased, de-duplicated search terms, in order of first appearance."""
    seen: set[str] = set()
    terms: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) >= limit:
            break
    return terms


def fts_query(text: str) -> str:
    """Turn a natural-language question into a safe FTS5 MATCH expression.

    Three things this has to survive, all of which raise rather than degrade:

    1. PUNCTUATION. "What about the API?" is a syntax error -- FTS5 does not
       quietly ignore the '?', it refuses the whole query.
    2. OPERATOR WORDS. A question containing "and", "or", "not" or "near" would
       be parsed as an operator rather than as something to search for.
    3. LEADING HYPHENS. "-foo" is a column filter, not a word.

    Quoting every term as a literal string defuses all three.

    The terms are joined with OR, not AND. A ten-word question ANDed together
    matches nothing; BM25 already rewards documents containing more of the
    *rare* terms, so an OR query self-sorts. That is also why there is no
    stopword list: "the" appears in every note, so its IDF is near zero and it
    contributes almost nothing to the score. A stopword list would be a second
    place to be wrong about language, for no gain.
    """
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in query_terms(text))


# --------------------------------------------------------------------- fusion
def rrf(lists: dict[str, list[int]], k: int, weights: dict[str, float]):
    """Reciprocal Rank Fusion. Returns (scores, per-retriever ranks)."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in lists.items():
        weight = weights.get(name, 1.0)
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            ranks.setdefault(cid, {})[name] = rank
    return scores, ranks


def _in_range(date: str | None, since: str | None, until: str | None) -> bool:
    """ISO date strings compare correctly as plain strings -- that is the point
    of storing them normalised.

    An undated note fails any date filter. That is a decision, not an
    oversight: --since is a claim about when something happened, and a note
    with no date cannot support that claim. Ask without the filter to see them.
    """
    if not date:
        return False
    if since and date < since:
        return False
    if until and date > until:
        return False
    return True


def _link_neighbourhood(conn: sqlite3.Connection, paths: list[str]) -> set[str]:
    """Notes one hop from `paths` in either direction."""
    if not paths:
        return set()
    holes = ",".join("?" * len(paths))
    near = set()
    for row in conn.execute(
            f"SELECT dst FROM links WHERE dst IS NOT NULL AND src IN ({holes})", paths):
        near.add(row["dst"])
    for row in conn.execute(f"SELECT src FROM links WHERE dst IN ({holes})", paths):
        near.add(row["src"])
    return near


# --------------------------------------------------------------------- result
@dataclass
class Hit:
    chunk_id: int
    path: str
    ord: int
    breadcrumb: str
    heading: str
    start_line: int
    body: str
    date: str | None
    score: float
    ranks: dict = field(default_factory=dict)
    boosted: bool = False
    context: list = field(default_factory=list)

    @property
    def why(self) -> str:
        """Compact provenance, e.g. 'b3 v11 link' -- which retriever found this."""
        bits = [f"{name[0]}{rank}" for name, rank in sorted(self.ranks.items())]
        if self.boosted:
            bits.append("link")
        return " ".join(bits)

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}"


# -------------------------------------------------------------------- search
class Searcher:
    """Holds the loaded vector matrix so repeated queries do not reload it."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        # From the index, not from the server: knowing which model's vectors
        # are stored costs no network call, and search must work with both
        # llama-servers switched off when --bm25-only is used.
        self.model = store.get_meta(conn, "embed_model")
        self._ids: np.ndarray | None = None
        self._mat: np.ndarray | None = None
        self._row_of: dict[int, int] = {}

    # -- vectors ----------------------------------------------------------
    def _load(self) -> bool:
        """Load every chunk vector into one matrix. False if there are none.

        Brute force, deliberately. Measured on this schema: 17,500 chunks at
        1024 dims is a 0.9 ms dot product. An ANN index would add a
        dependency, a build step, a second structure to keep in sync with the
        chunks table, and an approximation -- to save well under a
        millisecond. Revisit somewhere past 100k chunks; 250 notes is roughly
        800, which is two orders of magnitude away.
        """
        if self._mat is not None:
            return self._mat.shape[0] > 0

        rows = []
        if self.model:
            rows = self.conn.execute(
                """SELECT c.id, e.vec
                     FROM chunks c
                     JOIN embed_cache e
                       ON e.text_hash = c.text_hash AND e.model = ?
                 ORDER BY c.id""",
                (self.model,)).fetchall()

        if not rows:
            self._ids = np.zeros(0, dtype=np.int64)
            self._mat = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
            return False

        dim = len(rows[0]["vec"]) // 4          # float32
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, row in enumerate(rows):
            vec = np.frombuffer(row["vec"], dtype=np.float32)
            if vec.shape[0] != dim:
                raise SearchError(
                    f"Chunk {row['id']} has a {vec.shape[0]}-dim vector while the "
                    f"rest of the index uses {dim}. Two embedding models have been "
                    f"mixed; re-run index.py --rebuild.")
            mat[i] = vec
        self._ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
        self._mat = mat
        self._row_of = {int(cid): i for i, cid in enumerate(self._ids)}
        return True

    def has_vectors(self) -> bool:
        """Whether this index can do semantic search at all."""
        return self._load()

    def vector_search(self, query: str, n: int) -> list[int]:
        if not self._load():
            return []
        qvec = embed.embed([config.QUERY_PREFIX + query])[0]
        # Both sides are L2-normalised at write time, so the dot product IS
        # the cosine similarity. No division, no per-query normalisation.
        scores = self._mat @ qvec
        order = np.argsort(-scores)[:n]
        # Vector search has no concept of "no match": ask about something the
        # vault has never heard of and it still hands back its 50 least-bad
        # guesses. The floor is the only defence, and it is off by default
        # because the right value is model- and vault-specific -- measure with
        # search.py --explain before setting config.MIN_VECTOR_SCORE.
        floor = config.MIN_VECTOR_SCORE
        return [int(self._ids[i]) for i in order if scores[i] >= floor]

    def bm25_search(self, query: str, n: int) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        weights = ", ".join(f"{w:g}" for w in config.FTS_WEIGHTS)
        sql = (
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            # bm25() returns a NEGATIVE number and more negative is better, so
            # ascending order is best-first. Writing DESC here -- the intuitive
            # thing, and the most common bug in hand-rolled FTS5 code --
            # returns the worst matches in the corpus, which look plausible
            # enough that you may never notice.
            f"ORDER BY bm25(chunks_fts, {weights}) LIMIT ?"
        )
        try:
            rows = self.conn.execute(sql, (match, n)).fetchall()
        except sqlite3.OperationalError as exc:
            raise SearchError(f"FTS5 rejected the query {match!r}: {exc}") from exc
        return [int(r["chunk_id"]) for r in rows]

    # -- support ----------------------------------------------------------
    def _hydrate(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        out: dict[int, sqlite3.Row] = {}
        for start in range(0, len(ids), 400):       # SQLite variable limit
            batch = ids[start:start + 400]
            holes = ",".join("?" * len(batch))
            for row in self.conn.execute(
                    f"""SELECT c.id, c.path, c.ord, c.breadcrumb, c.heading,
                               c.start_line, c.body, n.date
                          FROM chunks c JOIN notes n ON n.path = c.path
                         WHERE c.id IN ({holes})""", batch):
                out[row["id"]] = row
        return out

    def _near_duplicate(self, cid: int, chosen: list[Hit], threshold: float) -> bool:
        """True if this chunk already said its piece under another name.

        Content hashing gives you incremental indexing; it cannot give you
        de-duplication, because it only catches byte-identical text. Two notes
        covering the same subject in different words share no hash and produce
        chunks at cosine ~0.97. Without this check they take turns filling the
        result list and the answer gets built from one idea repeated.

        A chunk with no vector (a BM25-only hit in a not-yet-embedded index) is
        never called a duplicate -- there is nothing to compare it against.
        """
        if threshold >= 1.0 or not chosen:
            return False
        i = self._row_of.get(cid)
        if i is None:
            return False
        vec = self._mat[i]
        for hit in chosen:
            j = self._row_of.get(hit.chunk_id)
            if j is not None and float(vec @ self._mat[j]) >= threshold:
                return True
        return False

    def _attach_context(self, hits: list[Hit], span: int) -> None:
        """Pull the chunks either side of each hit back in.

        Heading chunking splits at real topic boundaries, but the sentence
        that answers the question can still sit two paragraphs past the
        heading that matched. This restores continuity at read time instead of
        making every chunk bigger at index time.
        """
        if span <= 0:
            return
        chosen = {h.chunk_id for h in hits}
        for hit in hits:
            rows = self.conn.execute(
                """SELECT id, ord, heading, body FROM chunks
                    WHERE path = ? AND ord BETWEEN ? AND ? ORDER BY ord""",
                (hit.path, hit.ord - span, hit.ord + span)).fetchall()
            hit.context = [dict(r) for r in rows if r["id"] not in chosen]

    # -- the whole pipeline -----------------------------------------------
    def search(self, query: str, top_k: int | None = None,
               candidates: int | None = None, use_vector: bool = True,
               use_bm25: bool = True, since: str | None = None,
               until: str | None = None, link_boost: float | None = None,
               per_note: int | None = None, dedupe: float | None = None,
               neighbours: int | None = None,
               weights: dict[str, float] | None = None) -> list[Hit]:
        top_k = config.TOP_K if top_k is None else top_k
        candidates = config.CANDIDATES if candidates is None else candidates
        link_boost = config.LINK_BOOST if link_boost is None else link_boost
        per_note = config.MAX_CHUNKS_PER_NOTE if per_note is None else per_note
        dedupe = config.DEDUPE_COSINE if dedupe is None else dedupe
        neighbours = config.NEIGHBOUR_CONTEXT if neighbours is None else neighbours
        weights = config.RRF_WEIGHTS if weights is None else weights

        # A query with no searchable term is not a query. BM25 returns nothing
        # for one, but vector search always returns *something* -- embed an
        # empty string and you get 50 arbitrary chunks with a straight face.
        if not query_terms(query):
            return []

        lists: dict[str, list[int]] = {}
        if use_bm25:
            lists["bm25"] = self.bm25_search(query, candidates)
        if use_vector:
            lists["vector"] = self.vector_search(query, candidates)
        if not any(lists.values()):
            return []

        scores, ranks = rrf(lists, config.RRF_K, weights)
        rows = self._hydrate(list(scores))
        # A chunk can disappear between retrieval and hydration only if the
        # index is being rebuilt underneath us. Drop it rather than crash.
        scores = {cid: s for cid, s in scores.items() if cid in rows}

        if since or until:
            scores = {cid: s for cid, s in scores.items()
                      if _in_range(rows[cid]["date"], since, until)}
        if not scores:
            return []

        boosted: set[int] = set()
        if link_boost:
            ordered = sorted(scores, key=lambda c: -scores[c])
            seeds: list[str] = []
            for cid in ordered:
                path = rows[cid]["path"]
                if path not in seeds:
                    seeds.append(path)
                if len(seeds) >= config.LINK_SEEDS:
                    break
            near = _link_neighbourhood(self.conn, seeds) - set(seeds)
            # Expressed as a fraction of one first-place vote, so the number in
            # config means something: 1.0 would promote a linked note as hard
            # as being ranked first by an entire retriever.
            bonus = link_boost / (config.RRF_K + 1)
            for cid in scores:
                if rows[cid]["path"] in near:
                    scores[cid] += bonus
                    boosted.add(cid)

        # Stable ordering: score, then path, then position in the note, so two
        # runs over an unchanged index print identical output.
        ordered = sorted(scores, key=lambda c: (-scores[c], rows[c]["path"], rows[c]["ord"]))

        hits: list[Hit] = []
        seen_per_note: dict[str, int] = {}
        for cid in ordered:
            if len(hits) >= top_k:
                break
            row = rows[cid]
            if seen_per_note.get(row["path"], 0) >= per_note:
                continue
            if self._near_duplicate(cid, hits, dedupe):
                continue
            seen_per_note[row["path"]] = seen_per_note.get(row["path"], 0) + 1
            hits.append(Hit(
                chunk_id=cid, path=row["path"], ord=row["ord"],
                breadcrumb=row["breadcrumb"], heading=row["heading"],
                start_line=row["start_line"], body=row["body"], date=row["date"],
                score=scores[cid], ranks=ranks.get(cid, {}), boosted=cid in boosted,
            ))

        self._attach_context(hits, neighbours)
        return hits

    def explain(self, query: str, candidates: int | None = None,
                use_vector: bool = True, use_bm25: bool = True) -> dict:
        """The two ranked lists before fusion, for comparing retrievers."""
        candidates = config.CANDIDATES if candidates is None else candidates
        out = {
            "match": fts_query(query),
            "bm25": self.bm25_search(query, candidates) if use_bm25 else [],
            "vector": self.vector_search(query, candidates) if use_vector else [],
        }
        out["rows"] = self._hydrate(list({*out["bm25"], *out["vector"]}))
        return out


# ------------------------------------------------------------------ CLI glue
# These live here, rather than being written out twice, so that a default
# cannot drift between "what search.py shows you" and "what ask.py answers
# from". Two front-ends disagreeing for reasons neither one prints is a
# genuinely nasty thing to debug.
def parse_date_arg(value: str | None, label: str) -> str | None:
    """Accept ISO or either day-first format on the command line."""
    from . import dates
    if not value:
        return None
    iso = dates.find_date(value.strip())
    if not iso:
        raise SystemExit(
            f"--{label}: could not read {value!r} as a date. "
            f"Try 2026-01-01, 01-01-2026 or 01.01.26.")
    return iso


def add_retrieval_args(parser) -> None:
    parser.add_argument("--index", help="index file (default: config.INDEX_PATH)")
    parser.add_argument("-k", "--top-k", type=int, default=None,
                        help="chunks to retrieve")
    parser.add_argument("--candidates", type=int, default=None,
                        help="how deep each retriever digs before fusion")
    parser.add_argument("--bm25-only", action="store_true",
                        help="keyword search only -- needs no embedding server")
    parser.add_argument("--vector-only", action="store_true",
                        help="semantic search only")
    parser.add_argument("--since", help="only notes dated on or after this")
    parser.add_argument("--until", help="only notes dated on or before this")
    parser.add_argument("--link-boost", type=float, default=None,
                        help="promote notes linked to the top hits (try 0.5; 0 = off)")
    parser.add_argument("--per-note", type=int, default=None,
                        help="max chunks from any one note")
    parser.add_argument("--neighbours", type=int, default=None,
                        help="also pull N chunks either side of each hit")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep near-identical chunks")
    # Tunable from the command line because the whole point of evaluate.py is
    # to try a value and measure it. Editing config.py between runs makes it
    # far too easy to compare two numbers produced by two different configs.
    parser.add_argument("--weight-bm25", type=float, default=None,
                        help="RRF weight for keyword search (default 1.0)")
    parser.add_argument("--weight-vector", type=float, default=None,
                        help="RRF weight for semantic search (default 1.0)")


def options_from_args(args) -> dict:
    """Turn parsed CLI flags into Searcher.search() keyword arguments."""
    return dict(
        top_k=args.top_k,
        candidates=args.candidates,
        use_vector=not args.bm25_only,
        use_bm25=not args.vector_only,
        since=args.since,
        until=args.until,
        link_boost=args.link_boost,
        per_note=args.per_note,
        dedupe=1.0 if args.no_dedupe else None,
        neighbours=args.neighbours,
        weights=_weights_from_args(args),
    )


def _weights_from_args(args) -> dict[str, float] | None:
    bm25 = getattr(args, "weight_bm25", None)
    vector = getattr(args, "weight_vector", None)
    if bm25 is None and vector is None:
        return None
    base = dict(config.RRF_WEIGHTS)
    if bm25 is not None:
        base["bm25"] = bm25
    if vector is not None:
        base["vector"] = vector
    return base
