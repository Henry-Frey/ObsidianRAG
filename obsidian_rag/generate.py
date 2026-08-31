"""Stage 3: turning retrieved chunks into a cited answer.

The retrieval in stage 2 is the hard part; this file is mostly about not
squandering it. Four decisions carry the weight:

1. THE MODEL CITES BY NUMBER, NOT BY PATH.
   Sources are presented as [1], [2], [3] and the model is asked to write
   those markers. It never sees a file path in a form it is invited to repeat.
   Ask a model to write "Countries/USA/West Virginia.md" and it will
   eventually write "countries/west virginia.md", or drop a folder, or add an
   extension -- and a citation you cannot click is worse than no citation,
   because it looks checkable. A single digit is hard to corrupt, and we map
   it back to the exact path ourselves. Numbers also let us verify afterwards
   that every citation refers to a source that actually exists.

2. THE PROMPT IS BUDGETED IN ADVANCE.
   Eight chunks at 2,000 characters is ~4,600 tokens before the question and
   the answer. Overrun an 8k context and llama.cpp either errors or silently
   drops tokens from the *front* of the prompt -- which is where the system
   message and its grounding rules live. So the sources are trimmed here, and
   the CLI reports how many were dropped rather than letting it happen quietly.

3. THINKING MODE IS DISABLED TWICE.
   Once by asking the server to disable it, once by filtering the stream. See
   ThinkFilter for why one is not enough.

4. TRANSCLUSIONS ARE FOLLOWED AT ANSWER TIME, NOT INDEX TIME.
   The indexer deliberately leaves "![[Runbook]]" as a marker instead of
   inlining the target, so the same text is never embedded twice under two
   different citations. Here we can afford to resolve it: if a retrieved chunk
   transcludes another note, that note is added as its own numbered source,
   cited in its own right.
"""
from __future__ import annotations

import json
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import config, embed

# The llama-server that refuses a chat request and the one that refuses an
# embedding request fail the same way, so they share an exception.
ServerError = embed.ServerError
ModelLoading = embed.ModelLoading


# ----------------------------------------------------------------- budgeting
def estimate_tokens(text: str) -> int:
    """Rough token count, deliberately pessimistic.

    We have no tokenizer and do not want the dependency. English runs about
    3.5 characters per token; scripts outside ASCII (Cyrillic, Greek, CJK,
    heavily accented text) run far denser -- often under 2. Counting the two
    separately keeps the estimate honest for a mixed-language vault, where a
    flat 3.5 would under-count exactly the notes most likely to overflow.
    """
    ascii_chars = sum(1 for ch in text if ch < "\x80")
    return int(ascii_chars / 3.5 + (len(text) - ascii_chars) / 1.6) + 1


# ------------------------------------------------------------------- sources
_EMBED_MARK = re.compile(r"\[embeds: ([^\]]+)\]")


@dataclass
class Source:
    n: int
    path: str
    line: int
    breadcrumb: str
    date: str | None
    body: str
    kind: str = "hit"          # 'hit' | 'transclusion'
    label: str = ""            # only set when the breadcrumb is ambiguous

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line}"

    def render(self) -> str:
        """What the model sees. Note the absence of the file path.

        The breadcrumb ("Projects > Atlas Migration > Decisions") tells the
        model what the passage is about, which is what it needs to judge
        relevance. The path is ours to print at the end.
        """
        head = f"[{self.n}] {self.breadcrumb}"
        if self.label:
            head += f" -- {self.label}"
        if self.date:
            head += f"  ({self.date})"
        if self.kind == "transclusion":
            head += "  (transcluded into the note above)"
        return f"{head}\n{self.body.strip()}"


def _transclusion_targets(conn: sqlite3.Connection, path: str, body: str) -> list[str]:
    """Notes transcluded by this chunk, resolved through the link graph.

    The marker records only the filename ("[embeds: Runbook]") while the edge
    records the target as written ("Ops/Runbook"), so they are matched on the
    filename. Unresolved embeds are skipped: they name a note that does not
    exist, and there is nothing to add.
    """
    names = {n.strip().lower() for n in _EMBED_MARK.findall(body)}
    if not names:
        return []
    out = []
    for row in conn.execute(
            "SELECT raw, dst FROM links WHERE src = ? AND kind = 'embed' AND dst IS NOT NULL",
            (path,)):
        if PurePosixPath(row["raw"]).name.strip().lower() in names and row["dst"] not in out:
            out.append(row["dst"])
    return out


def _note_excerpt(conn: sqlite3.Connection, path: str, limit: int):
    """The opening chunks of a note, up to `limit` characters."""
    rows = conn.execute(
        """SELECT c.ord, c.breadcrumb, c.start_line, c.body, n.date
             FROM chunks c JOIN notes n ON n.path = c.path
            WHERE c.path = ? ORDER BY c.ord""", (path,)).fetchall()
    if not rows:
        return None
    parts, total = [], 0
    for row in rows:
        if total and total + len(row["body"]) > limit:
            break
        parts.append(row["body"])
        total += len(row["body"])
    return rows[0], "\n\n".join(parts)[:limit]


def gather_sources(conn: sqlite3.Connection, hits, budget_tokens: int,
                   expand_transclusions: bool | None = None):
    """Numbered sources that fit the budget. Returns (sources, dropped_count).

    Greedy in rank order, and it skips rather than stops: one oversized chunk
    at rank 3 should not cost you ranks 4 through 8.
    """
    if expand_transclusions is None:
        expand_transclusions = config.EXPAND_TRANSCLUSIONS

    sources: list[Source] = []
    seen: set[str] = set()
    used, dropped = 0, 0

    def fits(text: str) -> bool:
        nonlocal used
        cost = estimate_tokens(text) + 8      # header and blank lines
        if used + cost > budget_tokens:
            return False
        used += cost
        return True

    for hit in hits:
        body = hit.body
        # Neighbour chunks, if stage 2 attached any, ride along inside the
        # same source rather than becoming separate citations -- they are
        # context for a hit, not independently retrieved evidence.
        for extra in getattr(hit, "context", []):
            body += "\n\n" + extra["body"]
        if not fits(body):
            dropped += 1
            continue
        sources.append(Source(len(sources) + 1, hit.path, hit.start_line,
                              hit.breadcrumb, hit.date, body))
        seen.add(hit.path)

        if not expand_transclusions:
            continue
        for target in _transclusion_targets(conn, hit.path, hit.body):
            if target in seen:
                continue
            found = _note_excerpt(conn, target, config.TRANSCLUSION_CHARS)
            if not found:
                continue
            row, text = found
            if not fits(text):
                dropped += 1
                continue
            seen.add(target)
            sources.append(Source(len(sources) + 1, target, row["start_line"],
                                  row["breadcrumb"], row["date"], text,
                                  kind="transclusion"))
    _disambiguate(sources)
    return sources, dropped


def _disambiguate(sources: list[Source]) -> None:
    """Make sure no two sources look the same to the model.

    Breadcrumbs collide constantly in a real vault: "meetings/2025-05-20
    Kestrel Cache.md" and "meetings/2025-08-22 Kestrel Cache.md" both render
    as "meetings > Kestrel Cache sync > Actions". Two indistinguishable
    sources is worse than a slightly longer header -- the model cites [1]
    when it means [2], the citation resolves to a real note, and nothing
    about the answer looks wrong. Where breadcrumbs repeat, add the filename
    stem, which is the part that actually differs.
    """
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.breadcrumb] = counts.get(source.breadcrumb, 0) + 1
    for source in sources:
        if counts[source.breadcrumb] > 1:
            source.label = PurePosixPath(source.path).stem


# -------------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You answer questions about the user's personal notes.

Rules:
1. Use only the numbered sources given to you. No outside knowledge, no
   guessing, no filling in what a note "probably" said.
2. Cite the source number in brackets, like [2], immediately after each claim
   it supports. Cite several where several apply.
3. If the sources do not answer the question, say so in one sentence and stop.
   Name what is missing if you can. Do not pad the answer to look useful.
4. Where the sources disagree, say so and cite both rather than picking one.
5. Use the user's own names and terminology from the notes, not synonyms.
6. Answer in the language of the question, even when the notes are in another
   language.
7. Be brief. No preamble, no restating the question, no summary of which
   sources you read."""


def build_messages(question: str, sources: list[Source]) -> list[dict]:
    blocks = "\n\n".join(source.render() for source in sources)
    user = (f"SOURCES\n=======\n\n{blocks}\n\n"
            f"QUESTION\n========\n\n{question.strip()}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ------------------------------------------------------------ thinking mode
class ThinkFilter:
    """Remove <think>...</think> from a token stream.

    Belt and braces on purpose. The request asks the server to disable
    thinking, but that only takes effect if llama-server was started with
    --jinja *and* the GGUF carries Qwen3's own chat template. If either is
    missing, the flag is ignored without error and the model's entire
    reasoning monologue lands in your answer -- often contradicting the
    conclusion it eventually reaches.

    Written as a state machine rather than a regex because it runs on a
    stream: a tag can arrive split across two network chunks, so a partial
    match at the end of the buffer has to be held back rather than emitted.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    @staticmethod
    def _partial_tail(buf: str, tag: str) -> str:
        """The longest suffix of `buf` that could be the start of `tag`."""
        for size in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:size]):
                return buf[-size:]
        return ""

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []
        while self.buf:
            if self.inside:
                at = self.buf.find(self.CLOSE)
                if at < 0:
                    self.buf = self._partial_tail(self.buf, self.CLOSE)
                    break
                self.buf = self.buf[at + len(self.CLOSE):]
                self.inside = False
            else:
                at = self.buf.find(self.OPEN)
                if at < 0:
                    keep = self._partial_tail(self.buf, self.OPEN)
                    out.append(self.buf[:len(self.buf) - len(keep)])
                    self.buf = keep
                    break
                out.append(self.buf[:at])
                self.buf = self.buf[at + len(self.OPEN):]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        """Anything held back at end of stream. An unclosed <think> is dropped."""
        rest = "" if self.inside else self.buf
        self.buf = ""
        return rest


# ---------------------------------------------------------------- chat client
def _payload(messages: list[dict], stream: bool, temperature: float | None,
             max_tokens: int | None) -> dict:
    body = {
        "model": "chat",
        "messages": messages,
        # Not 0.0: Qwen3's own model card warns that greedy decoding sends it
        # into repetition loops. Low but non-zero is the grounded-extraction
        # setting; the card's 0.7 is for open-ended chat, which this is not.
        "temperature": config.CHAT_TEMPERATURE if temperature is None else temperature,
        "top_p": config.CHAT_TOP_P,
        "max_tokens": config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens,
        "stream": stream,
    }
    if not config.CHAT_ENABLE_THINKING:
        # Honoured only with --jinja and a template that defines it. Harmless
        # otherwise, which is exactly why ThinkFilter also exists.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _open(url: str, body: dict):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        return urllib.request.urlopen(request, timeout=config.REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise embed.http_error(url, exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise ServerError(
            f"Cannot reach the chat server at {url} ({exc.reason}).\n"
            f"Start it with serve-chat.ps1, or check CHAT_HOST in config.py."
        ) from exc


def chat(messages: list[dict], stream: bool = True, on_token=None,
         temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Send a chat request. Returns the full answer with thinking removed."""
    url = f"{config.CHAT_HOST}/v1/chat/completions"
    body = _payload(messages, stream, temperature, max_tokens)
    think = ThinkFilter()

    if not stream:
        with _open(url, body) as response:
            data = json.loads(response.read().decode("utf-8"))
        try:
            raw = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ServerError(f"Unexpected chat response: {str(data)[:300]}") from exc
        text = think.feed(raw) + think.flush()
        if on_token:
            on_token(text)
        return text

    parts: list[str] = []
    with _open(url, body) as response:
        for line in response:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content") or ""
            if not piece:
                continue
            visible = think.feed(piece)
            if visible:
                parts.append(visible)
                if on_token:
                    on_token(visible)
    tail = think.flush()
    if tail:
        parts.append(tail)
        if on_token:
            on_token(tail)
    return "".join(parts)


def probe_chat() -> dict:
    """One tiny generation, to prove the chat server actually answers.

    This is also the only way to detect a missing --jinja. Without it,
    chat_template_kwargs is accepted and ignored, no error is raised, and the
    first symptom is <think> blocks turning up in your answers weeks later.
    Here we look at the raw text before ThinkFilter touches it, so the leak is
    visible rather than quietly cleaned up.
    """
    messages = [{"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": "Reply with the single word OK."}]
    body = _payload(messages, False, None, 24)
    with _open(f"{config.CHAT_HOST}/v1/chat/completions", body) as response:
        data = json.loads(response.read().decode("utf-8"))
    try:
        raw = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise ServerError(f"Unexpected chat response: {str(data)[:300]}") from exc
    return {"raw": raw.strip(),
            "thinking_leaked": "<think>" in raw,
            "n_ctx": server_context_size()}


# ------------------------------------------------------------------ citations
_CITE = re.compile(r"\[(\d{1,3})\]")


def check_citations(text: str, count: int) -> tuple[list[int], list[int]]:
    """Which sources the answer cited, and which numbers it invented.

    A model that cites [9] when you gave it six sources has stopped reading
    and started pattern-matching, and that is worth seeing rather than
    rendering as a dead link.
    """
    used = sorted({int(m) for m in _CITE.findall(text)})
    return ([n for n in used if 1 <= n <= count],
            [n for n in used if not 1 <= n <= count])


# -------------------------------------------------------------------- answer
@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    bogus: list[int] = field(default_factory=list)
    dropped: int = 0
    prompt_tokens: int = 0
    messages: list[dict] = field(default_factory=list)


def context_budget(n_ctx: int | None = None, max_tokens: int | None = None) -> int:
    """Tokens available for source text, after the answer and the framing."""
    n_ctx = n_ctx or config.CHAT_N_CTX
    max_tokens = config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens
    fixed = estimate_tokens(SYSTEM_PROMPT) + config.CONTEXT_RESERVE_TOKENS
    return max(256, n_ctx - max_tokens - fixed)


def server_context_size() -> int | None:
    """n_ctx as the chat server actually reports it, or None if unreachable.

    Worth the one extra GET: config.CHAT_N_CTX is a guess about how you
    launched the server, and being wrong in the optimistic direction is how
    you lose the system prompt off the front of the context.
    """
    try:
        info = embed.props(config.CHAT_HOST)
    except ServerError:
        return None
    value = info.get("n_ctx") or (info.get("default_generation_settings") or {}).get("n_ctx")
    return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else None


def answer(conn: sqlite3.Connection, hits, question: str, stream: bool = True,
           on_token=None, temperature: float | None = None,
           max_tokens: int | None = None, n_ctx: int | None = None,
           expand_transclusions: bool | None = None,
           send: bool = True) -> Answer:
    """Assemble the prompt, ask the model, and check what came back."""
    budget = context_budget(n_ctx, max_tokens)
    sources, dropped = gather_sources(conn, hits, budget, expand_transclusions)
    messages = build_messages(question, sources)
    prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)

    if not sources:
        return Answer(question, "", [], [], [], dropped, prompt_tokens, messages)
    if not send:                                   # --show-prompt
        return Answer(question, "", sources, [], [], dropped, prompt_tokens, messages)

    text = chat(messages, stream=stream, on_token=on_token,
                temperature=temperature, max_tokens=max_tokens)
    cited, bogus = check_citations(text, len(sources))
    return Answer(question, text, sources, cited, bogus, dropped, prompt_tokens, messages)
