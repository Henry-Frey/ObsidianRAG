# ObsidianRAG

Local retrieval-augmented question answering over an Obsidian vault. Runs
outside Obsidian, reads the markdown folder, and answers questions with
citations back to `path:line`.

Built as a standalone tool rather than a community plugin because Obsidian
plugins are unsandboxed.

## Guarantees

- **Fully local.** The only network calls are HTTP to `127.0.0.1`, and they all
  go through [`obsidian_rag/embed.py`](obsidian_rag/embed.py) and
  [`obsidian_rag/generate.py`](obsidian_rag/generate.py). No API keys.
- **Read-only.** [`obsidian_rag/vault.py`](obsidian_rag/vault.py) is the only
  module that opens a path inside the vault, and it opens every file `"rb"`.
  `assert_outside_vault()` refuses to run if the index would land inside it.
  Searching and answering never touch the vault at all.
- **One dependency.** numpy. Everything else is the standard library.

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) — pick the build
  matching your GPU (`cuda` for NVIDIA, `vulkan` for Intel/AMD)
- An embedding GGUF, default `bge-m3-Q8_0.gguf` (1024 dims, multilingual)
- A chat GGUF, default `Qwen3-8B-Q4_K_M.gguf`

Put the binaries in `%USERPROFILE%\models\llama-server\` and the GGUFs in
`%USERPROFILE%\models\`, or point `MODELS_DIR` elsewhere.

## Setup

```powershell
pip install -r requirements.txt
. .\orag.ps1                       # gives you the `orag` command

$env:OBSIDIAN_VAULT = "D:\Notes"
orag serve both                    # two llama-servers, each in its own window
orag doctor --wait 600             # verify before trusting anything
orag index
```

`orag env` shows which paths and models are in effect. Everything is settable
by environment variable so one checkout works on more than one machine:

| Variable | Default |
|---|---|
| `OBSIDIAN_VAULT` | see `config.py` |
| `MODELS_DIR` | `%USERPROFILE%\models` |
| `EMBED_GGUF` | `bge-m3-Q8_0.gguf` |
| `CHAT_GGUF` | `Qwen3-8B-Q4_K_M.gguf` |
| `EMBED_HOST` / `CHAT_HOST` | `http://127.0.0.1:8081` / `:8080` |
| `LLAMA_EXTRA` | passed through to llama-server |

## Use

```powershell
orag ask    "what did we decide about the Atlas rollback"
orag search "atlas rollback"          # the retrieved chunks, no generation
orag ask                              # interactive
```

Useful flags, on both `ask` and `search`:

```
-k N              chunks to retrieve            --since / --until DATE
--bm25-only       no embedding server needed    --per-note N
--vector-only     semantic only                 --neighbours N
--explain         both ranked lists, unfused    --link-boost 0.5
```

`--since 01.01.26`, `--since 01-01-2026` and `--since 2026-01-01` all work.

Reports: `orag stats`, `orag links`, `orag dates`, `orag show "note.md"`.

## How it works

**Stage 1 — index** ([`index.py`](index.py)). Walks the vault, skips
`.obsidian`, attachments, cloud placeholders and sync-conflict copies. Splits
each note on markdown headings rather than fixed token windows, because
headings are boundaries a human already placed. Every chunk carries a
breadcrumb (`Projects > Atlas Migration > Decisions`) into both its embedding
and its BM25 row, so a passage reading "they came down to 40k" is still
findable. Incremental at two levels: a note's content hash decides what gets
re-chunked, and each chunk's text hash decides what gets re-embedded.

**Stage 2 — retrieve** ([`obsidian_rag/search.py`](obsidian_rag/search.py)).
BM25 via SQLite FTS5 and brute-force cosine over numpy, fused with Reciprocal
Rank Fusion. The two retrievers fail in opposite directions: BM25 finds
`Kubernetes 1.29`, vectors find "reverting a release" when you asked about
"rolling back a deploy". Results are capped per note and de-duplicated by
cosine, so one topic written up twice cannot fill the result list.

**Stage 3 — answer** ([`obsidian_rag/generate.py`](obsidian_rag/generate.py)).
Sources are numbered and the model cites `[n]`, never a file path — a model
asked to write `Countries/USA/West Virginia.md` will eventually write something
close but wrong, and a citation you cannot click is worse than none. The prompt
is budgeted before it is sent, because an overrun drops tokens off the *front*,
where the grounding rules live. Citations are verified afterwards.

Everything tunable lives in [`obsidian_rag/config.py`](obsidian_rag/config.py),
which is the file to read first.

## Tests

```powershell
python tests/test_all.py
```

No pytest. Stubs the embedder with a hashing vectoriser and runs a real HTTP
server on a loopback port for the chat model, so streaming, SSE parsing and
`<think>` filtering are exercised over an actual socket. It does not prove that
a real llama-server returns the shapes expected — that is what `orag doctor`
is for — and it does not measure retrieval quality.

## Known gaps

- **No retrieval evaluation.** Nothing here measures whether the answers are
  good. A question/expected-source set is the obvious next thing to build.
- `--doctor` cannot detect a wrong `--pooling`: you get correctly shaped
  vectors that are quietly worse, with no error.
- Anchor text from `[[links]]` is not folded into the BM25 row yet.
- The frontmatter parser handles a flat YAML subset, not the whole spec.
- Merged chunks carry the breadcrumb of their first section.
