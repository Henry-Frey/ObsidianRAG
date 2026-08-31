"""The index: one SQLite file holding notes, chunks, vectors and the BM25 index.

Layout
------
notes         one row per markdown file, keyed by vault-relative path
chunks        one row per retrieval unit, with its vector as a float32 BLOB
chunks_fts    FTS5 mirror of chunks, giving us real Okapi BM25 for free
embed_cache   sha256(embed_text) -> vector, so unchanged *sections* of a
              changed note are never re-embedded

Why the embed_cache matters: hashing whole notes gives you note-level
incrementality, so fixing one typo re-embeds the entire note. Hashing each
chunk's embed_text gives you section-level incrementality -- edit one heading
and you pay for one embedding. At 120 notes that is the difference between a
30-second re-index and an instant one, and it costs one extra table.

Vectors live in the same file rather than a sidecar .npy: 2,500 chunks at 1024
float32 dims is ~10 MB, SQLite reads that in milliseconds, and there is exactly
one file to back up, delete, or inspect.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path, PurePosixPath

import numpy as np

from . import config, dates

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS notes (
    path         TEXT PRIMARY KEY,      -- vault-relative, forward slashes
    content_hash TEXT NOT NULL,         -- sha256 of the raw file bytes
    mtime        REAL NOT NULL,
    size         INTEGER NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    aliases      TEXT NOT NULL DEFAULT '[]',
    tags         TEXT NOT NULL DEFAULT '[]',
    links        TEXT NOT NULL DEFAULT '[]',
    embeds       TEXT NOT NULL DEFAULT '[]',
    date         TEXT,                  -- ISO YYYY-MM-DD, NULL if undated
    date_source  TEXT NOT NULL DEFAULT '',   -- 'filename' | 'frontmatter:<key>'
    indexed_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_date ON notes(date);

-- Resolved wikilink graph. dst is NULL for a link naming a note that does not
-- exist -- which in Obsidian is most of them, and is meaningful rather than
-- broken: it is a concept you referenced but have not written up.
CREATE TABLE IF NOT EXISTS links (
    src  TEXT NOT NULL,
    dst  TEXT,
    raw  TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- 'link' | 'embed'
    PRIMARY KEY (src, raw, kind)
);
CREATE INDEX IF NOT EXISTS links_dst ON links(dst);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    ord         INTEGER NOT NULL,
    breadcrumb  TEXT NOT NULL DEFAULT '',
    heading     TEXT NOT NULL DEFAULT '',
    start_line  INTEGER NOT NULL DEFAULT 1,
    body        TEXT NOT NULL,      -- what the model is shown
    embed_text  TEXT NOT NULL,      -- what was actually embedded
    meta_text   TEXT NOT NULL DEFAULT '',
    text_hash   TEXT NOT NULL,      -- sha256(embed_text)
    UNIQUE(path, ord)
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(text_hash);

-- Contentless=0 (a normal FTS5 table) on purpose: it costs a copy of the text
-- but keeps the SQL you have to read down to one INSERT and one SELECT.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    breadcrumb,
    body,
    meta,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS embed_cache (
    text_hash TEXT NOT NULL,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,
    PRIMARY KEY (text_hash, model)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    val TEXT NOT NULL
);
"""


SCHEMA_VERSION = "2"


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _check_schema(conn, path)           # must precede SCHEMA: see below
    conn.executescript(SCHEMA)          # CREATE IF NOT EXISTS: no-op if present
    set_meta(conn, "schema_version", SCHEMA_VERSION)
    conn.commit()
    return conn


class SchemaError(RuntimeError):
    pass


def _check_schema(conn: sqlite3.Connection, path: Path) -> None:
    """Detect an index built by an older version of this code.

    CREATE TABLE IF NOT EXISTS will not add a column to a table that already
    exists, so an old index survives table creation and then fails on an INSERT
    halfway through the run. This has to run *before* the schema script,
    because CREATE INDEX on a column the old table lacks raises first, and a
    raw OperationalError tells you nothing about what to do next.
    """
    tables = {r["name"] for r in
              conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "notes" not in tables:
        return                                       # fresh database
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(notes)")}
    missing = {"date", "date_source"} - cols
    version = get_meta(conn, "schema_version", "") if "meta" in tables else ""
    if missing or (version and version != SCHEMA_VERSION):
        detail = ", ".join(sorted(missing)) if missing else f"schema v{version}"
        raise SchemaError(
            f"Index at {path} predates the current schema (missing: {detail}).\n"
            f"Re-run with --rebuild -- indexing from scratch takes seconds."
        )


# ------------------------------------------------------------------- vectors
def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cache_get(conn: sqlite3.Connection, hashes: list[str], model: str) -> dict[str, np.ndarray]:
    """Look up already-computed embeddings. Chunked to stay under SQLite's
    999-variable limit on older builds."""
    found: dict[str, np.ndarray] = {}
    for start in range(0, len(hashes), 400):
        batch = hashes[start:start + 400]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT text_hash, vec FROM embed_cache WHERE model = ? AND text_hash IN ({placeholders})",
            [model, *batch],
        )
        for row in rows:
            found[row["text_hash"]] = unpack(row["vec"])
    return found


def cache_put(conn: sqlite3.Connection, items: list[tuple[str, np.ndarray]], model: str) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embed_cache (text_hash, model, dim, vec) VALUES (?, ?, ?, ?)",
        [(h, model, len(v), pack(v)) for h, v in items],
    )


# --------------------------------------------------------------------- notes
def existing_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["path"]: r["content_hash"] for r in conn.execute("SELECT path, content_hash FROM notes")}


def delete_note(conn: sqlite3.Connection, path: str) -> None:
    """Remove a note and everything derived from it.

    The FTS table has no foreign keys, so its rows must be deleted explicitly.
    Forgetting this is how you end up with a search index full of notes you
    deleted months ago -- results that cite files which no longer exist.
    """
    ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE path = ?", (path,))]
    if ids:
        conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
    conn.execute("DELETE FROM notes WHERE path = ?", (path,))


def upsert_note(conn: sqlite3.Connection, note, meta, chunks) -> None:
    """Replace a note's row and all of its chunks. Caller wraps this in a txn."""
    delete_note(conn, note.rel)
    note_date, date_source = dates.note_date(note.rel, meta.frontmatter)
    conn.execute(
        """INSERT INTO notes (path, content_hash, mtime, size, title, aliases,
                              tags, links, embeds, date, date_source, indexed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (note.rel, note.content_hash, note.mtime, note.size, meta.title,
         json.dumps(meta.aliases, ensure_ascii=False),
         json.dumps(meta.tags, ensure_ascii=False),
         json.dumps(meta.links, ensure_ascii=False),
         json.dumps(meta.embeds, ensure_ascii=False),
         note_date, date_source,
         time.time()),
    )
    for chunk in chunks:
        cur = conn.execute(
            """INSERT INTO chunks (path, ord, breadcrumb, heading, start_line,
                                   body, embed_text, meta_text, text_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (note.rel, chunk.ord, chunk.breadcrumb, chunk.heading,
             chunk.start_line, chunk.body, chunk.embed_text,
             chunk.meta_text, chunk.text_hash),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, breadcrumb, body, meta) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, chunk.breadcrumb, chunk.body, chunk.meta_text),
        )


# --------------------------------------------------------------------- links
def _resolution_table(rows) -> dict[str, set[str]]:
    """Every name a note can be linked by -> the notes answering to that name.

    Deliberately matches Obsidian's own rule: full path, and filename stem, and
    explicit frontmatter aliases. NOT the H1 title -- I measured that on a real
    vault and it recovered one extra link while creating two new ambiguities.
    More importantly, resolving by title would make this tool's graph disagree
    with the graph Obsidian shows you, which is worse than missing an edge.
    """
    table: dict[str, set[str]] = {}

    def add(name: str, path: str) -> None:
        key = name.strip().lower()
        if key:
            table.setdefault(key, set()).add(path)

    for row in rows:
        path = row["path"]
        bare = path[:-3] if path.lower().endswith(".md") else path
        add(bare, path)                                  # Countries/USA/WV
        add(PurePosixPath(bare).name, path)              # WV
        for alias in json.loads(row["aliases"]):
            add(alias, path)
    return table


def _resolve_one(raw: str, table: dict[str, set[str]], src: str) -> str | None:
    key = raw.strip()
    if key.lower().endswith(".md"):
        key = key[:-3]
    found = table.get(key.lower()) or table.get(PurePosixPath(key).name.lower()) or set()
    found = {p for p in found if p != src}               # a self-link is not an edge
    if not found:
        return None
    # Deterministic tiebreak on ambiguity: shallowest path, then alphabetical.
    # A slightly-wrong edge beats a dropped one when links are sparse and
    # each one was placed deliberately.
    return min(found, key=lambda p: (p.count("/"), p))


def resolve_links(conn: sqlite3.Connection) -> dict:
    """Rebuild the link graph. Must run after every note is indexed, because
    resolution needs the complete set of filenames to resolve against."""
    rows = conn.execute("SELECT path, aliases, links, embeds FROM notes").fetchall()
    table = _resolution_table(rows)

    edges: list[tuple] = []
    for row in rows:
        src = row["path"]
        for kind, column in (("link", "links"), ("embed", "embeds")):
            for raw in json.loads(row[column]):
                edges.append((src, _resolve_one(raw, table, src), raw, kind))

    conn.execute("DELETE FROM links")
    conn.executemany(
        "INSERT OR REPLACE INTO links (src, dst, raw, kind) VALUES (?, ?, ?, ?)", edges
    )
    return {
        "edges": len(edges),
        "resolved": sum(1 for e in edges if e[1]),
        "unresolved": sum(1 for e in edges if not e[1]),
    }


def backlinks(conn: sqlite3.Connection, path: str) -> list[str]:
    return [r["src"] for r in conn.execute(
        "SELECT DISTINCT src FROM links WHERE dst = ? ORDER BY src", (path,))]


def outlinks(conn: sqlite3.Connection, path: str) -> list[str]:
    return [r["dst"] for r in conn.execute(
        "SELECT DISTINCT dst FROM links WHERE src = ? AND dst IS NOT NULL ORDER BY dst",
        (path,))]


def set_meta(conn: sqlite3.Connection, key: str, val: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, val) VALUES (?, ?)", (key, val))


def get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT val FROM meta WHERE key = ?", (key,)).fetchone()
    return row["val"] if row else default


def unembedded(conn: sqlite3.Connection, model: str) -> list[tuple[str, str]]:
    """(text_hash, embed_text) for chunks with no vector for this model.

    Storing embed_text on the chunk row (rather than only in memory during the
    run that created it) is what makes this query possible, and it is what lets
    an interrupted index run resume instead of silently leaving chunks that are
    in the BM25 index but invisible to vector search.
    """
    rows = conn.execute(
        """SELECT DISTINCT c.text_hash, c.embed_text
             FROM chunks c
        LEFT JOIN embed_cache e
               ON e.text_hash = c.text_hash AND e.model = ?
            WHERE e.text_hash IS NULL""",
        (model,),
    )
    return [(r["text_hash"], r["embed_text"]) for r in rows]


def stats(conn: sqlite3.Connection) -> dict:
    one = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "notes": one("SELECT COUNT(*) FROM notes"),
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "cached_vectors": one("SELECT COUNT(*) FROM embed_cache"),
        "fts_rows": one("SELECT COUNT(*) FROM chunks_fts"),
    }
