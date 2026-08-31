"""Stage 1: build and incrementally update the index.

    python index.py --vault "D:\\Notes" --dry-run     # chunk plan, no servers needed
    python index.py --vault "D:\\Notes"               # index for real
    python index.py --show "projects/acme.md"         # inspect one note
    python index.py --stats

The run is idempotent: index twice in a row and the second run does nothing.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from obsidian_rag import chunker, config, embed, generate, store, vault


@dataclass
class IndexedNote:
    rel: str
    content_hash: str
    mtime: float
    size: int


def collect(vault_path: Path, verbose: bool) -> tuple[dict[str, tuple], list[str]]:
    """Read every eligible note. Returns {rel: (NoteFile, text, hash)} and skips."""
    found: dict[str, tuple] = {}
    skipped: list[str] = []
    for item in vault.iter_notes(vault_path):
        if isinstance(item, tuple):
            skipped.append(item[1])
            continue
        try:
            text, digest = vault.read_note(item.path)
        except OSError as exc:
            skipped.append(f"{item.rel}  [read failed: {exc}]")
            continue
        found[item.rel] = (item, text, digest)
        if verbose:
            print(f"  read {item.rel}")
    return found, skipped


def do_index(args) -> int:
    vault_path = Path(args.vault or config.VAULT_PATH)
    if not vault_path.is_dir():
        print(f"Vault not found: {vault_path}\n"
              f"Pass --vault, or set OBSIDIAN_VAULT, or edit obsidian_rag/config.py.",
              file=sys.stderr)
        return 2

    index_path = Path(args.index or config.INDEX_PATH)
    # Hard stop before anything else touches disk.
    vault.assert_outside_vault(index_path, vault_path)

    if args.rebuild and index_path.exists():
        print(f"Rebuilding: removing {index_path}")
        index_path.unlink()
        for suffix in ("-wal", "-shm"):
            extra = index_path.with_name(index_path.name + suffix)
            if extra.exists():
                extra.unlink()

    started = time.time()
    conn = store.connect(index_path)

    print(f"Scanning {vault_path} ...")
    found, skipped = collect(vault_path, args.verbose)
    known = store.existing_hashes(conn)

    changed = [rel for rel, (_, _, h) in found.items() if known.get(rel) != h]
    deleted = [rel for rel in known if rel not in found]
    print(f"  {len(found)} notes  |  {len(changed)} new/changed  |  "
          f"{len(deleted)} deleted  |  {len(skipped)} skipped")

    if skipped:
        print("\nSkipped:")
        for line in skipped[:20]:
            print(f"  - {line}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    # ------------------------------------------------------------- chunking
    total_chunks = 0
    sizes: list[int] = []
    with conn:
        for rel in deleted:
            store.delete_note(conn, rel)

        for rel in changed:
            note, text, digest = found[rel]
            meta, chunks = chunker.chunk_note(rel, text)
            sizes.extend(len(c.body) for c in chunks)
            total_chunks += len(chunks)
            if args.dry_run:
                print(f"  {rel}: {len(chunks)} chunks "
                      f"({', '.join(str(len(c.body)) for c in chunks[:8])}"
                      f"{' ...' if len(chunks) > 8 else ''})")
                continue
            store.upsert_note(
                conn,
                IndexedNote(rel, digest, note.mtime, note.size),
                meta, chunks,
            )

    if sizes:
        print(f"\nChunk sizes (chars): min={min(sizes)} median={int(statistics.median(sizes))} "
              f"mean={int(statistics.mean(sizes))} max={max(sizes)} n={len(sizes)}")

    if args.dry_run:
        print("\nDry run: nothing written, no server contacted.")
        return 0

    # --------------------------------------------------------------- links
    # Runs after every note is written, because resolving [[Foo]] needs the
    # complete set of filenames to resolve against.
    with conn:
        graph = store.resolve_links(conn)
    print(f"\nLinks: {graph['edges']} edges, {graph['resolved']} resolved, "
          f"{graph['unresolved']} naming notes that do not exist")
    with conn:
        store.set_meta(conn, "vault_path", str(vault_path))

    # ------------------------------------------------------------ embedding
    if args.no_embed:
        # Lets you build a searchable index with no llama-server running at
        # all. BM25 works immediately; the vectors can be filled in later
        # without re-chunking, because embed_cache is keyed by text hash.
        print("\nSkipping embeddings (--no-embed). Keyword search works now; run\n"
              "again without the flag once the embedding server is up.")
        s = store.stats(conn)
        print(f"\nIndex: {s['notes']} notes, {s['chunks']} chunks, "
              f"{s['fts_rows']} FTS rows, no vectors")
        return 0

    # The model identity comes from the running server, not from a constant,
    # so swapping the GGUF invalidates cached vectors on its own.
    model = embed.model_id()
    previous = store.get_meta(conn, "embed_model")
    if previous and previous != model:
        print(f"\n! Embedding model changed: {previous} -> {model}")
        print("  Vectors are keyed by model, so everything will be re-embedded.")

    if model == "unknown":
        print("\nThe embedding server did not report which model it loaded,\n"
              "so vectors would be cached under the key 'unknown' and then\n"
              "re-embedded from scratch the moment it does report a name.\n"
              "Nothing was embedded. Start serve-embed.ps1, or set\n"
              "config.EMBED_MODEL_ID if you are running without a server.\n"
              "The text index is already usable: search.py --bm25-only works.")
        return 2

    pending = store.unembedded(conn, model)
    if pending:
        print(f"\nEmbedding {len(pending)} chunks with {model} ...")

        def progress(done: int, total: int) -> None:
            print(f"\r  {done}/{total}", end="", flush=True)

        hashes = [h for h, _ in pending]
        texts = [t for _, t in pending]
        vectors = embed.embed_batched(texts, progress=progress)
        print()
        if vectors.shape[1] != config.EMBED_DIM:
            print(f"! config.EMBED_DIM is {config.EMBED_DIM} but the server "
                  f"returned {vectors.shape[1]}. Update config.EMBED_DIM.")
        with conn:
            store.cache_put(conn, list(zip(hashes, vectors)), model)
    else:
        print(f"\nNo new chunks to embed ({model}).")

    with conn:
        store.set_meta(conn, "embed_model", model)
        store.set_meta(conn, "last_index", str(time.time()))

    s = store.stats(conn)
    print(f"\nIndex: {s['notes']} notes, {s['chunks']} chunks, "
          f"{s['cached_vectors']} vectors, {s['fts_rows']} FTS rows")
    # The write-ahead log holds most of a fresh index until SQLite checkpoints
    # it, so reporting only the main file says 0.0 MB after a rebuild.
    on_disk = sum(p.stat().st_size for p in
                  (index_path, index_path.with_name(index_path.name + "-wal"))
                  if p.exists())
    print(f"Wrote {index_path}  ({on_disk / 1e6:.1f} MB) "
          f"in {time.time() - started:.1f}s")
    return 0


def do_show(args) -> int:
    """Print the chunks for one note so you can eyeball the chunking."""
    vault_path = Path(args.vault or config.VAULT_PATH)
    target = vault_path / args.show
    if not target.is_file():
        print(f"Not found: {target}", file=sys.stderr)
        return 2
    text, _ = vault.read_note(target)
    meta, chunks = chunker.chunk_note(args.show.replace("\\", "/"), text)
    print(f"title:   {meta.title}")
    print(f"aliases: {meta.aliases}")
    print(f"tags:    {meta.tags}")
    print(f"links:   {meta.links}")
    print(f"embeds:  {meta.embeds}")
    for c in chunks:
        print("\n" + "=" * 72)
        print(f"[{c.ord}] {c.breadcrumb}   (line {c.start_line}, {len(c.body)} chars)")
        print("-" * 72)
        print(c.body)
    return 0


def do_stats(args) -> int:
    conn = store.connect(Path(args.index or config.INDEX_PATH))
    s = store.stats(conn)
    print(f"index:  {args.index or config.INDEX_PATH}")
    print(f"vault:  {store.get_meta(conn, 'vault_path', '(unset)')}")
    print(f"model:  {store.get_meta(conn, 'embed_model', '(unset)')}")
    for key, value in s.items():
        print(f"{key:>16}: {value}")
    return 0


def do_links(args) -> int:
    """Report the link graph: what is most referenced, and what is missing."""
    conn = store.connect(Path(args.index or config.INDEX_PATH))
    total, resolved = conn.execute(
        "SELECT COUNT(*), COUNT(dst) FROM links").fetchone()
    print(f"{total} edges, {resolved} resolved, {total - resolved} unresolved\n")

    print("most linked-to notes:")
    for r in conn.execute("""SELECT dst, COUNT(DISTINCT src) n FROM links
                              WHERE dst IS NOT NULL GROUP BY dst
                           ORDER BY n DESC, dst LIMIT 15"""):
        print(f"  {r['n']:3}  {r['dst']}")

    # Unresolved links are not errors. They are concepts you have referenced
    # but never written up -- arguably the most useful thing in this report.
    print("\nreferenced but never written:")
    rows = conn.execute("""SELECT raw, COUNT(*) n FROM links WHERE dst IS NULL
                        GROUP BY LOWER(raw) ORDER BY n DESC, raw LIMIT 20""").fetchall()
    for r in rows:
        print(f"  {r['n']:3}  {r['raw']}")
    if not rows:
        print("  (none)")
    return 0


def do_dates(args) -> int:
    """Show what date each note was assigned and where it came from.

    Worth eyeballing once on a real vault: DD-MM and MM-DD are indistinguishable
    whenever both numbers are <= 12, so a systematic misparse looks like
    perfectly ordinary dates until you notice none of them fall after the 12th.
    """
    conn = store.connect(Path(args.index or config.INDEX_PATH))
    total = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    dated = conn.execute("SELECT COUNT(*) FROM notes WHERE date IS NOT NULL").fetchone()[0]
    print(f"{dated}/{total} notes carry a date\n")
    for r in conn.execute("""SELECT date, date_source, COUNT(*) n FROM notes
                              WHERE date IS NOT NULL GROUP BY date_source"""):
        print(f"  {r['n']:4} from {r['date_source']}")
    print()
    for r in conn.execute("""SELECT path, date, date_source FROM notes
                              WHERE date IS NOT NULL ORDER BY date DESC LIMIT 25"""):
        print(f"  {r['date']}  {r['date_source']:22} {r['path']}")
    return 0


def _await_server(label: str, host: str, seconds: int) -> None:
    """Block while a server finishes loading its weights, if asked to.

    A 22 GB GGUF off a cold page cache takes minutes, and llama-server answers
    every request with 503 until it is done. Without this you just run
    --doctor over and over until one of them happens to succeed.
    """
    if seconds <= 0:
        return
    started = [False]

    def tick(remaining: int) -> None:
        if not started[0]:
            print(f"  waiting up to {seconds}s for {label} to load ",
                  end="", flush=True)
            started[0] = True
        print(".", end="", flush=True)

    try:
        embed.wait_until_ready(host, seconds, on_wait=tick)
    except embed.ServerError:
        pass                      # the check below reports it properly
    if started[0]:
        print()


def do_doctor(args) -> int:
    """Verify both llama-server instances before trusting an index run.

    Worth its 30 lines: a misconfigured embedding server does not crash, it
    returns plausible vectors of the wrong shape or from the wrong pooling
    mode, and you find out weeks later when retrieval is mediocre.
    """
    ok = True
    print(f"models dir : {config.MODELS_DIR}  "
          f"({'found' if config.MODELS_DIR.is_dir() else 'MISSING'})")
    # A configured GGUF that is not on disk is only a problem if the matching
    # server is not already serving something else. Running a model other than
    # the one config names is a perfectly reasonable thing to do, and calling
    # it MISSING sends you looking for a file you never wanted.
    configured = {}
    for key, label, name in (("embed", "embed gguf", config.EMBED_GGUF),
                             ("chat", "chat gguf ", config.CHAT_GGUF)):
        path = config.gguf(name)
        size = f"{path.stat().st_size / 1e9:.1f} GB" if path.is_file() else "not on disk"
        print(f"{label} : {name}  ({size})")
        configured[key] = (name, path.is_file())
    loaded = {"embed": None, "chat": None}

    print(f"\nembed host : {config.EMBED_HOST}")
    _await_server("the embedding server", config.EMBED_HOST, args.wait)
    try:
        info = embed.check(config.EMBED_HOST)
        loaded["embed"] = info["model"]
        print(f"  model    : {info['model']}")
        print(f"  n_ctx    : {info['n_ctx']}")
        print(f"  dim      : {info['dim']}"
              + ("" if info["dim"] == config.EMBED_DIM
                 else f"   ! config.EMBED_DIM says {config.EMBED_DIM}"))
        ok &= info["dim"] == config.EMBED_DIM
    except embed.ModelLoading as exc:
        print(f"  LOADING  : {exc}")
        ok = False
    except embed.ServerError as exc:
        print(f"  FAILED   : {exc}")
        ok = False

    print(f"\nchat host  : {config.CHAT_HOST}")
    _await_server("the chat server", config.CHAT_HOST, args.wait)
    try:
        # props() rather than model_id(): model_id swallows connection errors
        # and returns "unknown", which would let a dead server pass this check.
        embed.props(config.CHAT_HOST)
        loaded["chat"] = embed.model_id(config.CHAT_HOST)
        print(f"  model    : {loaded['chat']}")
        # Actually generate. Reaching /props only proves a process is
        # listening; it says nothing about whether the chat endpoint works or
        # whether the template honours the thinking switch.
        probe = generate.probe_chat()
        print(f"  n_ctx    : {probe['n_ctx']}")
        print(f"  generates: {probe['raw'][:60]!r}")
        if probe["thinking_leaked"]:
            print("  ! <think> came back in the reply, so enable_thinking was"
                  "\n    ignored. Restart llama-server with --jinja. Answers are"
                  "\n    filtered anyway, but the model is wasting your context"
                  "\n    and your tokens reasoning out loud.")
            ok = False
    except embed.ModelLoading as exc:
        print(f"  LOADING  : {exc}")
        ok = False
    except embed.ServerError as exc:
        print(f"  FAILED   : {exc}")
        ok = False

    # Reconcile what config expects against what is actually running.
    for key, var in (("embed", "EMBED_GGUF"), ("chat", "CHAT_GGUF")):
        name, on_disk = configured[key]
        running = loaded[key]
        if running and running not in ("unknown", name):
            print(f"\n! config.{var} is {name}, but that server has {running} "
                  f"loaded.\n  Not an error -- but set  $env:{var} = \"{running}\"  "
                  f"so the two halves\n  of your setup agree, and so --doctor stops "
                  f"looking for the wrong file.")
        elif not on_disk and not running:
            ok = False

    print("\nGPU offload is not reported over HTTP. Check each server's startup"
          "\nlog for a Vulkan device line and 'offloaded N/N layers to GPU'.")
    print("\nOK" if ok else "\nProblems found (see above).")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Index an Obsidian vault (read-only).")
    p.add_argument("--vault", help="vault root (default: config.VAULT_PATH)")
    p.add_argument("--index", help="index file (default: config.INDEX_PATH)")
    p.add_argument("--dry-run", action="store_true",
                   help="chunk everything and report, but write nothing and contact no server")
    p.add_argument("--rebuild", action="store_true", help="delete the index and start fresh")
    p.add_argument("--no-embed", action="store_true",
                   help="index text only, contact no server (BM25 search still works)")
    p.add_argument("--show", metavar="REL_PATH", help="print the chunks for one note")
    p.add_argument("--stats", action="store_true", help="show index statistics")
    p.add_argument("--doctor", action="store_true",
                   help="check models and both llama-server instances, then exit")
    p.add_argument("--links", action="store_true",
                   help="report the wikilink graph and unwritten concepts")
    p.add_argument("--dates", action="store_true",
                   help="report the date assigned to each note and its source")
    p.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                   help="with --doctor, poll until the servers finish loading")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    try:
        if args.doctor:
            return do_doctor(args)
        if args.links:
            return do_links(args)
        if args.dates:
            return do_dates(args)
        if args.show:
            return do_show(args)
        if args.stats:
            return do_stats(args)
        return do_index(args)
    except (store.SchemaError, vault.VaultError, embed.ServerError) as exc:
        # These carry an actionable message; a traceback would only bury it.
        print(f"\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
