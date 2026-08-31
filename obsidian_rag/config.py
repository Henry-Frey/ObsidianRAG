"""Every tunable knob in the system. Nothing else hard-codes a path or a model.

Read this file first: it is the whole configuration surface.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --------------------------------------------------------------------- vault
# Your vault root. EVERYTHING under this path is treated as strictly read-only.
# Override without editing this file:  set OBSIDIAN_VAULT=D:\Notes
VAULT_PATH = Path(os.environ.get("OBSIDIAN_VAULT", r"C:\path\to\your\vault"))

# Directory names pruned during the walk (matched on the name, at any depth).
SKIP_DIRS = {
    ".obsidian",      # plugin + config data, never content
    ".trash",         # Obsidian's soft-delete folder
    ".smart-env",     # left behind by some community plugins
    ".git",
    "node_modules",
    "__pycache__",
}

# Prune any directory whose name starts with a dot. Obsidian never stores
# user notes in one, and this catches plugin folders we haven't named above.
SKIP_HIDDEN_DIRS = True

# Only these extensions are treated as notes. Everything else -- images, PDFs,
# canvases, Excalidraw files -- is an attachment and is ignored entirely.
NOTE_SUFFIXES = {".md"}

# Sync-conflict filenames. Left in, these produce near-duplicate chunks that
# crowd genuine hits out of the top-k with stale copies of the same text.
CONFLICT_PATTERNS = (
    re.compile(r"\.sync-conflict-\d", re.I),          # Syncthing
    re.compile(r"\(conflicted copy", re.I),           # Dropbox
    re.compile(r"\(\w+'s conflicted copy", re.I),     # Dropbox, named
    re.compile(r"-\w+-PC\.md$", re.I),                # OneDrive, hostname suffix
    re.compile(r"^~\$"),                              # Office lock files
)

# --------------------------------------------------------------------- dates
# Two-digit years: YY < pivot -> 20YY, otherwise 19YY. 40 puts 00-39 in this
# century, which covers any plausible note date, while still reading 98 as
# 1998 rather than 2098.
TWO_DIGIT_YEAR_PIVOT = 40

# Look for a date in the filename. Frontmatter keys are tried after it, in
# order. Body prose is never scanned -- see the note in dates.py.
DATE_FROM_FILENAME = True
DATE_FRONTMATTER_KEYS = ("date", "created")

# --------------------------------------------------------------------- index
# The index lives OUTSIDE the vault. Two reasons: (1) writing into the vault
# would violate the read-only guarantee, (2) an 8 MB file rewritten on every
# run, inside a synced folder, is a sync-churn machine.
INDEX_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "obsidian-rag"
INDEX_PATH = INDEX_DIR / "index.sqlite3"

# --------------------------------------------------------------------- models
# Local GGUF weights and the llama.cpp binaries that serve them.
#
# Home-relative on purpose. Path.home() resolves to C:\Users\Who on this
# machine, so MODELS_DIR lands on C:\Users\Who\models without baking a
# username into the repo -- the same checkout works on another machine, and
# nothing here leaks a personal path into version control.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / "models"))
LLAMA_BIN_DIR = MODELS_DIR / "llama-server"


def gguf(name: str) -> Path:
    """Resolve a GGUF by filename inside MODELS_DIR."""
    return MODELS_DIR / name


# ----------------------------------------------------------------- llama.cpp
# Two llama-server instances, because llama-server serves exactly one model per
# process. That is not just a constraint to work around -- it means the
# embedding model is never evicted to make room for the chat model partway
# through an index run, which is what would happen with a single-slot server.
#
# 127.0.0.1 rather than "localhost": on Windows localhost can resolve to ::1
# first and fail if the server is bound to IPv4 only.
EMBED_HOST = os.environ.get("EMBED_HOST", "http://127.0.0.1:8081")
CHAT_HOST = os.environ.get("CHAT_HOST", "http://127.0.0.1:8080")

# Filenames inside MODELS_DIR. These are what you pass to llama-server -m;
# nothing in this codebase opens them directly.
# Overridable so one checkout works on more than one machine -- the same
# variable names the serve-*.ps1 scripts read.
EMBED_GGUF = os.environ.get("EMBED_GGUF", "bge-m3-Q8_0.gguf")
CHAT_GGUF = os.environ.get("CHAT_GGUF", "Qwen3-8B-Q4_K_M.gguf")

EMBED_DIM = 1024            # bge-m3; index.py warns if the server disagrees
EMBED_BATCH = 16            # texts per HTTP request
REQUEST_TIMEOUT = 300       # seconds

# Cache key for stored vectors. Left as None, it is read from the running
# server's /props, so swapping the GGUF automatically invalidates the cache
# instead of silently mixing vectors from two different models -- a failure
# that produces quietly wrong retrieval rather than an error. Only set this to
# a literal string if you are running without a reachable server.
EMBED_MODEL_ID: str | None = None

# ------------------------------------------------------------------- chunking
# Sizes are in characters, not tokens: characters are what we can measure
# without a tokenizer, and the ratio is stable enough (~3.5 chars/token for
# English, denser for CJK) that precision here buys nothing.
TARGET_CHARS = 1200         # preferred chunk size (~300 tokens)
MAX_CHARS = 2000            # hard ceiling; larger sections get split
MIN_CHARS = 250             # below this, merge into a neighbour
OVERLAP_PARAS = 1           # paragraphs repeated across a split boundary

# Prepend note title / breadcrumb / tags / aliases to the embedded text.
# Costs a little dilution, buys a lot of recall on proper nouns. See README
# notes in chunker.py for the reasoning.
INCLUDE_META_IN_EMBEDDING = True

# Include the containing folders in each chunk's breadcrumb, so
# Countries/USA/West Virginia.md embeds as "Countries > USA > West Virginia".
# Most vaults encode real taxonomy in folders (Projects/, People/, Areas/Work)
# and this is free context. Set False if your folders are an unsorted dumping
# ground, where the folder names would only add noise to every chunk.
INCLUDE_FOLDER_IN_BREADCRUMB = True

# Fenced blocks with these info strings are plugin *queries*, not content:
# they render into something else at view time, so indexing them retrieves
# code you never wrote and cannot read in the source note.
DROP_FENCE_LANGS = {"dataview", "dataviewjs", "query", "tasks", "kanban-plugin"}

# Extensions that mark an ![[embed]] as an attachment rather than a note.
ATTACHMENT_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif",
    ".pdf", ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".mov", ".webm", ".mkv",
    ".canvas", ".excalidraw", ".base",
}

# ------------------------------------------------------------------ retrieval
# How many chunks an answer is built from, and how deep each retriever digs
# before fusion. CANDIDATES is deliberately much larger than TOP_K: fusion can
# only rerank what it was given, and a chunk ranked 30th by BM25 and 25th by
# vector is often better than either list's 5th.
TOP_K = 8
CANDIDATES = 50

# Reciprocal Rank Fusion. score = sum over retrievers of weight / (RRF_K + rank).
# RRF_K = 60 is the value from the original TREC paper and is not sensitive:
# it flattens the difference between rank 1 and rank 2 so that a chunk both
# retrievers rank highly beats a chunk one retriever loves and the other has
# never heard of. Lower it to make first place count for more.
RRF_K = 60
RRF_WEIGHTS = {"vector": 1.0, "bm25": 1.0}

# BM25 column weights, in the column order of chunks_fts:
#   (chunk_id, breadcrumb, body, meta)
# chunk_id is UNINDEXED and can never match, so its weight is 0. The breadcrumb
# carries folders + note title + heading trail, so weighting it above the body
# makes a query that names a note surface that note.
FTS_WEIGHTS = (0.0, 2.0, 1.0, 1.5)

# At most this many chunks from any one note in a result set. Without a cap, a
# single long note that happens to match well fills every slot and the answer
# is written from one source.
MAX_CHUNKS_PER_NOTE = 3

# Drop a candidate whose vector is at least this similar to one already
# selected. This is what catches the same topic written up twice in different
# words -- content hashing cannot, because the text is not identical. Set to
# 1.0 to disable.
DEDUPE_COSINE = 0.95

# Link-graph boost, in units of "fraction of a first-place vote". 0.0 = off.
# See the note in search.py: this is off by default because its value depends
# on how you actually use links, and that is measurable with --link-boost.
LINK_BOOST = 0.0
LINK_SEEDS = 5              # top results whose neighbourhood gets the boost

# Chunks either side of a hit to attach as context. 0 = just the hit.
NEIGHBOUR_CONTEXT = 0

# Some embedding models want queries prefixed with an instruction ("Represent
# this sentence for searching relevant passages:"). BGE-M3 does NOT -- it was
# trained without one, and adding it measurably hurts. Left empty on purpose;
# change it only if you swap to a model whose card asks for one.
QUERY_PREFIX = ""

# Minimum cosine similarity for a vector hit. 0.0 = keep everything, which is
# the honest default: vector search cannot say "no match", so without a floor
# a question your vault has never covered still returns its least-bad guesses.
# BGE-M3 puts genuinely related text around 0.6+; measure on your own notes
# with search.py --explain before raising this above 0.
MIN_VECTOR_SCORE = 0.0

# ----------------------------------------------------------------- generation
# Not 0.0. Qwen3's model card explicitly warns that greedy decoding drives it
# into repetition loops. Low but non-zero is right for grounded extraction;
# the card's recommended 0.7 is for open-ended chat, which this is not.
CHAT_TEMPERATURE = 0.3
CHAT_TOP_P = 0.9
CHAT_MAX_TOKENS = 800       # room for a thorough answer, not an essay
CHAT_STREAM = True

# Fallback only -- the real n_ctx is read from the chat server when it is
# reachable. This matters because being wrong in the optimistic direction
# does not raise: llama.cpp drops tokens off the FRONT of the prompt, which
# is where the system message and its grounding rules live.
CHAT_N_CTX = 8192
CONTEXT_RESERVE_TOKENS = 320    # chat template, framing text, slack

# Qwen3 emits <think>...</think> unless told not to. Disabled two ways; see
# ThinkFilter in generate.py for why one is not enough.
CHAT_ENABLE_THINKING = False

# Follow ![[transclusions]] out of a retrieved chunk and add the target as its
# own numbered source. The indexer never inlines them, so this is the only
# point at which they are resolved.
EXPAND_TRANSCLUSIONS = True
TRANSCLUSION_CHARS = 1200
