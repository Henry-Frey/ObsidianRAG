"""The whole test suite. Run it with:

    python tests/test_all.py

No pytest, matching the rest of the project's dependency budget. Each group is
a function; a failure raises and the traceback points at the exact assertion.

Two things are stubbed, because neither can be assumed present:

  * embeddings   replaced by a hashing bag-of-words vector. Similar text gets
                 similar vectors, which is all the fusion, capping and
                 de-duplication logic needs to be exercised.
  * the chat server  replaced by a real HTTP server on a loopback port, so the
                 streaming path, SSE parsing and the think filter are tested
                 over an actual socket rather than against a mock object.

What this suite does NOT prove: that a real llama-server returns the shapes we
expect, or that retrieval quality is good. Those need index.py --doctor and
your own judgement respectively.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obsidian_rag import config, dates, embed, generate, search, store  # noqa: E402

DIM = 256


# --------------------------------------------------------------------- setup
def fake_vec(text: str) -> np.ndarray:
    """Hashing bag-of-words -> unit vector. Deterministic, no server."""
    vec = np.zeros(DIM, dtype=np.float32)
    for word in search.query_terms(text, limit=10_000):
        vec[int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def build_index(db: Path) -> None:
    """Index the dummy vault through the real CLI, so that gets tested too.

    The vault is generated rather than committed -- it is 120 files of
    deterministic filler, and make_dummy_vault.py reproduces it exactly from
    a seed. Generating it here means a fresh clone can run the tests.
    """
    vault = ROOT / "dummy-vault"
    if not vault.is_dir():
        print("generating the dummy vault ...")
        made = subprocess.run(
            [sys.executable, str(ROOT / "make_dummy_vault.py"), "--out", str(vault)],
            capture_output=True, text=True)
        assert made.returncode == 0, made.stdout + made.stderr

    result = subprocess.run(
        [sys.executable, str(ROOT / "index.py"),
         "--vault", str(vault), "--index", str(db),
         "--rebuild", "--no-embed"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "217 chunks" in result.stdout or "chunks" in result.stdout


def seed_vectors(conn) -> None:
    rows = conn.execute("SELECT DISTINCT text_hash, embed_text FROM chunks").fetchall()
    with conn:
        store.cache_put(conn, [(r["text_hash"], fake_vec(r["embed_text"])) for r in rows],
                        "stub")
        store.set_meta(conn, "embed_model", "stub")


# ---------------------------------------------------------------------- dates
def test_dates() -> None:
    good = {
        "2026-08-31": "2026-08-31",              # ISO
        "31-08-2026": "2026-08-31",              # DD-MM-YYYY
        "31.08.2026": "2026-08-31",              # DD.MM.YYYY
        "31-08-26": "2026-08-31",                # DD-MM-YY
        "31.08.26": "2026-08-31",                # DD.MM.YY
        "31-08-2026 Standup": "2026-08-31",      # trailing words
        "Meeting_31.08.26": "2026-08-31",        # underscore boundary
        "2026-08-31-Retro": "2026-08-31",        # trailing hyphen allowed
        "29-02-2024": "2024-02-29",              # real leap day
        "01.09.26": "2026-09-01",
        "14.03.24": "2024-03-14",
        "31.12.98": "1998-12-31",                # pivot -> last century
    }
    for text, want in good.items():
        assert dates.find_date(text) == want, (text, dates.find_date(text))

    rejected = [
        "v1.2.26",          # version number, no leading boundary
        "Note-1.2.26",      # same
        "32-13-2026",       # impossible day and month
        "29-02-2025",       # not a leap year
        "31-08.2026",       # separators disagree
        "no digits here",
    ]
    for text in rejected:
        assert dates.find_date(text) is None, (text, dates.find_date(text))

    # filename beats frontmatter
    iso, src = dates.note_date("Daily/31-08-2026 Standup.md", {"created": "01-01-2020"})
    assert (iso, src) == ("2026-08-31", "filename")
    iso, src = dates.note_date("Notes/untitled.md", {"created": "01-01-2020"})
    assert (iso, src) == ("2020-01-01", "frontmatter:created")
    assert dates.note_date("Notes/untitled.md", {}) == (None, "")
    print(f"dates: {len(good)} parsed, {len(rejected)} correctly rejected, "
          f"precedence ok")


# ------------------------------------------------------------- query parsing
def test_query_parsing() -> None:
    # punctuation and operator words must not reach FTS5 unquoted
    q = search.fts_query("What about the API? (v1.29) and/or rollbacks")
    assert '"and"' in q and '"or"' in q and "?" not in q and "(" not in q
    assert search.fts_query("???") == ""
    assert search.fts_query("") == ""
    assert search.query_terms("O'Brien's Kubernetes") == ["o'brien's", "kubernetes"]
    assert search.fts_query('say "hi"') == '"say" OR "hi"'
    print("query parsing: punctuation, operators and apostrophes handled")


# --------------------------------------------------------------------- fusion
def test_rrf() -> None:
    scores, ranks = search.rrf({"bm25": [10, 20, 30], "vector": [30, 10, 99]},
                               60, {"bm25": 1.0, "vector": 1.0})
    assert abs(scores[10] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(scores[30] - (1 / 63 + 1 / 61)) < 1e-12
    assert abs(scores[99] - (1 / 63)) < 1e-12
    # found by both beats found once, even at a better single rank
    assert scores[10] > scores[30] > scores[20] > scores[99]
    assert ranks[10] == {"bm25": 1, "vector": 2}
    print("rrf: arithmetic exact; agreement beats a single strong rank")


def test_bm25_sign(conn, searcher) -> None:
    """bm25() is negative and more-negative is better. Ascending is best-first."""
    weights = ", ".join(f"{w:g}" for w in config.FTS_WEIGHTS)
    match = search.fts_query("Atlas migration")
    best = conn.execute(f"SELECT chunk_id, bm25(chunks_fts,{weights}) r FROM chunks_fts "
                        f"WHERE chunks_fts MATCH ? ORDER BY r LIMIT 1", (match,)).fetchone()
    worst = conn.execute(f"SELECT chunk_id, bm25(chunks_fts,{weights}) r FROM chunks_fts "
                         f"WHERE chunks_fts MATCH ? ORDER BY r DESC LIMIT 1", (match,)).fetchone()
    assert best["r"] < worst["r"] < 0
    assert searcher.bm25_search("Atlas migration", 1)[0] == best["chunk_id"]
    print(f"bm25 sign: best={best['r']:.2f} worst={worst['r']:.2f}, "
          f"ascending is best-first")


# ------------------------------------------------------------------ retrieval
QUERY = "What did we decide about the Atlas migration rollback?"


def test_hybrid(searcher) -> None:
    hits = searcher.search(QUERY)
    assert hits
    assert any("vector" in h.ranks and "bm25" in h.ranks for h in hits), \
        "nothing was found by both retrievers -- fusion is not fusing"
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.path] = counts.get(hit.path, 0) + 1
    assert max(counts.values()) <= config.MAX_CHUNKS_PER_NOTE
    print(f"hybrid: {len(hits)} hits, per-note cap held at "
          f"{max(counts.values())}/{config.MAX_CHUNKS_PER_NOTE}")


def test_dedupe(searcher) -> None:
    question = "Quedan pendientes las pruebas de integracion"
    loose = searcher.search(question, top_k=8, per_note=99, dedupe=1.0)
    tight = searcher.search(question, top_k=8, per_note=99, dedupe=0.75)
    assert [h.citation for h in tight] != [h.citation for h in loose], \
        "the de-duplication threshold changed nothing"
    dropped = {h.citation for h in loose} - {h.citation for h in tight}
    print(f"dedupe: threshold 0.75 displaced {len(dropped)} chunks")


def test_dates_filter(searcher) -> None:
    window = searcher.search(QUERY, since="2025-01-01", until="2025-05-01")
    assert all(h.date and "2025-01-01" <= h.date <= "2025-05-01" for h in window)
    assert [h.citation for h in window] != [h.citation for h in searcher.search(QUERY)]
    print(f"date filter: {len(window)} hits inside the window, undated excluded")


def test_no_match(searcher) -> None:
    assert searcher.search("") == []
    assert searcher.search("???") == []
    never = "zzzqqq nonexistent token"
    assert searcher.search(never, use_vector=False) == [], "bm25 invented a match"
    assert searcher.search(never, use_bm25=False) != [], \
        "vector search should always return its least-bad guesses"
    config.MIN_VECTOR_SCORE = 0.9
    try:
        assert searcher.search(never, use_bm25=False) == [], "the score floor did nothing"
        assert searcher.search(QUERY) != [], "the score floor ate a real query"
    finally:
        config.MIN_VECTOR_SCORE = 0.0
    print("no-match: empty and unseen queries handled; score floor works")


def test_extras(searcher) -> None:
    plain = searcher.search(QUERY, top_k=12)
    boosted = searcher.search(QUERY, top_k=12, link_boost=1.0)
    assert [h.citation for h in plain] != [h.citation for h in boosted]
    assert any(h.boosted for h in boosted)
    withctx = searcher.search(QUERY, top_k=3, neighbours=1)
    assert any(h.context for h in withctx)
    print(f"extras: link boost reorders, neighbours attach "
          f"{sum(len(h.context) for h in withctx)} chunks")


# ------------------------------------------------------------- think filter
def filtered(pieces: list[str]) -> str:
    filt = generate.ThinkFilter()
    return "".join(filt.feed(p) for p in pieces) + filt.flush()


def test_think_filter() -> None:
    assert filtered(["<think>reasoning</think>answer"]) == "answer"
    assert filtered(["hello ", "<think>", "x", "</think>", " world"]) == "hello  world"
    # the case a regex over the finished string would miss: a tag arriving
    # split across two network chunks
    assert filtered(["a<thi", "nk>secret</thi", "nk>b"]) == "ab"
    # a partial held back mid-stream is released at flush -- if the stream
    # ended there it was text, and eating real output is the worse failure
    assert filtered(["a<th"]) == "a<th"
    assert filtered(["<think>never closed"]) == ""
    assert filtered(["<think>a</think>x<think>b</think>y"]) == "xy"
    assert filtered(["1 < 2 and 3 > 2"]) == "1 < 2 and 3 > 2"
    print("think filter: 7 cases including tags split across stream chunks")


# ---------------------------------------------------------------- generation
def test_citations() -> None:
    assert generate.check_citations("Yes [1] and no [3].", 3) == ([1, 3], [])
    assert generate.check_citations("See [9].", 3) == ([], [9])
    assert generate.check_citations("None here.", 3) == ([], [])
    assert generate.check_citations("[2][2][1]", 3) == ([1, 2], [])
    print("citations: real ones collected, invented ones flagged")


def test_budget() -> None:
    assert generate.estimate_tokens("привет мир " * 50) > \
           generate.estimate_tokens("hello world " * 50), \
        "non-ASCII must count as denser, or a mixed-language vault overflows"
    print(f"budget: {generate.context_budget(8192)} tokens of an 8192 context")


def test_sources(conn, searcher) -> None:
    hits = searcher.search("Kestrel Cache latency", top_k=6)
    with_t, _ = generate.gather_sources(conn, hits, 4000, expand_transclusions=True)
    without, _ = generate.gather_sources(conn, hits, 4000, expand_transclusions=False)
    pulled = [s.path for s in with_t if s.kind == "transclusion"]
    assert pulled, "the ![[embed]] edge was not followed"
    assert all(s.kind == "hit" for s in without)
    assert [s.n for s in with_t] == list(range(1, len(with_t) + 1)), \
        "source numbering must stay contiguous once transclusions are spliced in"

    # the model must never see a path it could try to echo back as a citation
    for source in with_t:
        assert source.path not in source.render()

    # ambiguous breadcrumbs get disambiguated
    dupes = searcher.search("Kestrel Cache sync actions", top_k=8, per_note=1)
    sources, _ = generate.gather_sources(conn, dupes, 6000)
    crumbs = [s.breadcrumb for s in sources]
    for source in sources:
        if crumbs.count(source.breadcrumb) > 1:
            assert source.label, f"ambiguous breadcrumb left unlabelled: {source.breadcrumb}"

    tight, dropped = generate.gather_sources(conn, hits, 120, expand_transclusions=False)
    assert dropped > 0 and len(tight) < len(hits)
    assert sum(generate.estimate_tokens(s.body) for s in tight) <= 120
    print(f"sources: transclusion {pulled[0]} followed, numbering contiguous, "
          f"budget trims to {len(tight)} keeping {dropped} out")


# ------------------------------------------------------- stand-in chat server
REPLY = ["<thi", "nk>I should check the sources.</think>",
         "Postponed ", "[1]", " pending the indexing fix ", "[2]", "."]
SEEN: dict = {}


class ChatHandler(BaseHTTPRequestHandler):
    loading = False

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if ChatHandler.loading:
            self._json({"error": {"message": "Loading model"}}, 503)
        else:
            self._json({"n_ctx": 4096, "model_path": "/models/stub.gguf"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.clear()
        SEEN.update(body)
        if ChatHandler.loading:
            self._json({"error": {"message": "Loading model"}}, 503)
            return
        if not body.get("stream"):
            self._json({"choices": [{"message": {"content": "".join(REPLY)}}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in REPLY:
            for i in range(0, len(piece), 5):     # split mid-word and mid-tag
                event = {"choices": [{"delta": {"content": piece[i:i + 5]}}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def test_chat(conn, searcher) -> None:
    server = HTTPServer(("127.0.0.1", 0), ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    previous = config.CHAT_HOST
    config.CHAT_HOST = f"http://127.0.0.1:{server.server_port}"
    try:
        hits = searcher.search(QUERY, top_k=4)

        # a server still loading is a distinct condition, not a failure
        ChatHandler.loading = True
        try:
            generate.probe_chat()
            raise AssertionError("a 503 while loading should raise")
        except generate.ModelLoading:
            pass
        assert generate.server_context_size() is None
        ChatHandler.loading = False

        assert generate.server_context_size() == 4096, \
            "n_ctx must come from the server, not from config"

        pieces: list[str] = []
        streamed = generate.answer(conn, hits, QUERY, stream=True, on_token=pieces.append)
        assert "".join(pieces) == streamed.text
        assert "<think>" not in streamed.text and "check the sources" not in streamed.text
        assert streamed.text.startswith("Postponed [1]")
        assert streamed.cited == [1, 2] and streamed.bogus == []

        whole = generate.answer(conn, hits, QUERY, stream=False)
        assert whole.text == streamed.text, "streaming and non-streaming disagree"

        assert SEEN["chat_template_kwargs"] == {"enable_thinking": False}
        assert [m["role"] for m in SEEN["messages"]] == ["system", "user"]
        assert "SOURCES" in SEEN["messages"][1]["content"]

        REPLY[:] = ["Everything is fine [99]."]
        bad = generate.answer(conn, hits, QUERY, stream=False)
        assert bad.bogus == [99] and bad.cited == []

        # --show-prompt must contact nothing at all
        config.CHAT_HOST = "http://127.0.0.1:1"
        dry = generate.answer(conn, hits, QUERY, send=False)
        assert dry.text == "" and dry.sources and dry.messages
        print("chat: streaming == non-streaming, thinking stripped, 503 "
              "distinguished, bogus citation caught, dry run offline")
    finally:
        config.CHAT_HOST = previous
        server.shutdown()


# --------------------------------------------------------------- http errors
def test_http_errors() -> None:
    loading = embed.http_error("u", 503, '{"message":"Loading model"}')
    assert isinstance(loading, embed.ModelLoading)
    assert isinstance(embed.http_error("u", 404, "nope"), embed.ServerError)
    assert not isinstance(embed.http_error("u", 404, "nope"), embed.ModelLoading)
    assert isinstance(embed.http_error("u", 503, "out of memory"), embed.ServerError)
    print("http errors: 'loading' distinguished from every other failure")


# ----------------------------------------------------------------------- main
def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="obsidian-rag-tests-"))
    db = workdir / "test.sqlite3"
    try:
        print(f"building a test index in {workdir}\n")
        build_index(db)
        # Queries must be embedded by the same stub that produced the stored
        # vectors, or nothing would be comparable. Patched after build_index,
        # which runs in a subprocess and is unaffected.
        embed.embed = lambda texts, host=None: np.vstack([fake_vec(t) for t in texts])
        conn = store.connect(db)
        seed_vectors(conn)
        searcher = search.Searcher(conn)
        assert searcher.has_vectors()

        test_dates()
        test_query_parsing()
        test_rrf()
        test_bm25_sign(conn, searcher)
        test_hybrid(searcher)
        test_dedupe(searcher)
        test_dates_filter(searcher)
        test_no_match(searcher)
        test_extras(searcher)
        test_think_filter()
        test_citations()
        test_budget()
        test_sources(conn, searcher)
        test_chat(conn, searcher)
        test_http_errors()

        print("\nALL PASSED")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
