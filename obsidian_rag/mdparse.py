"""Turning one Obsidian markdown file into clean, structured sections.

The hard parts, in order of how likely they are to bite:

1. Code fences. A shell or Python block containing hash-prefixed comment lines
   will be read as headings by any naive parser, shattering the note into
   nonsense sections. Every heading decision here is made against a fence mask.
2. Frontmatter. Raw YAML embedded as prose poisons an embedding. We parse it
   out and reuse only the parts that help retrieval (title/aliases/tags).
3. Wikilinks. The link text is usually the proper noun you will later search
   for, so we keep the words and throw away only the brackets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import config

# ------------------------------------------------------------------ patterns
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*(\S*)")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

# (!)[[target#heading|alias]] -- one regex covers links and transclusions.
WIKILINK_RE = re.compile(r"(!?)\[\[([^\[\]|#]*?)(#[^\[\]|]*?)?(?:\|([^\[\]]*?))?\]\]")
MDLINK_RE = re.compile(r"(!?)\[([^\[\]]*?)\]\(([^()\s]+)(?:\s+\"[^\"]*\")?\)")

CALLOUT_HEAD_RE = re.compile(r"^\s*\[!(\w+)\]([+-])?\s*(.*)$")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>+\s?")

# ==highlight== -> highlight. The markers carry emphasis a reader can see but
# an embedding cannot use, and they corrupt the word when left attached.
HIGHLIGHT_RE = re.compile(r"==(.+?)==", re.DOTALL)
# [^1] inline markers and the "[^1]:" prefix of a definition. The definition's
# text is real content and is kept; only the marker goes.
FOOTNOTE_RE = re.compile(r"\[\^[^\]\s]+\]:?")

OBSIDIAN_COMMENT_RE = re.compile(r"%%.*?%%", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BLOCK_REF_RE = re.compile(r"\s\^[A-Za-z0-9-]{1,32}\s*$", re.MULTILINE)

# A tag only at line start or after whitespace: this deliberately misses URL
# fragments (http://x#y) at the cost of also matching #hexcolors.
INLINE_TAG_RE = re.compile(r"(?:(?<=^)|(?<=\s))#([A-Za-z][\w/-]*)", re.MULTILINE)

_SENTINEL = "\x00F{}\x00"


@dataclass
class Section:
    """One heading and the body text directly under it (not its subsections)."""
    level: int                       # 0 = preamble above the first heading
    heading: str
    trail: list[str] = field(default_factory=list)   # ancestor headings
    start_line: int = 1              # 1-based line number in the source file
    lines: list[str] = field(default_factory=list)


@dataclass
class NoteMeta:
    title: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)     # wikilink targets
    embeds: list[str] = field(default_factory=list)    # transclusion targets
    frontmatter: dict = field(default_factory=dict)


# ---------------------------------------------------------------- frontmatter
def split_frontmatter(text: str) -> tuple[str, str, int]:
    """Return (frontmatter_text, body, body_start_line_1based)."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return "", text, 1
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:]), i + 2
    return "", text, 1          # unterminated -- treat the whole file as body


def parse_frontmatter(fm: str) -> dict:
    """A deliberately small YAML subset: the flat shapes Obsidian actually writes.

    Handles three forms only:
        key: value
        key: [a, b]
        key:
          - a
          - b

    Anything nested deeper is ignored rather than guessed at. This is ~35 lines
    you can read instead of a PyYAML dependency, whose safe_load also raises on
    unquoted wikilinks in frontmatter values -- a common Obsidian pattern.
    """
    out: dict = {}
    key: str | None = None
    for raw in fm.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s*-\s+(.*)$", raw)
        if item and key:                                  # block list entry
            out.setdefault(key, [])
            if isinstance(out[key], list):
                out[key].append(_unquote(item.group(1)))
            continue
        kv = re.match(r"^(\w[\w .-]*?)\s*:\s*(.*)$", raw)
        if not kv:
            continue
        key, value = kv.group(1).strip(), kv.group(2).strip()
        if not value:
            out[key] = []                                 # list follows below
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            out[key] = [_unquote(v) for v in inner.split(",") if v.strip()] if inner else []
        else:
            out[key] = _unquote(value)
    return out


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _as_list(value) -> list[str]:
    """Normalise a frontmatter value into a list of bare words."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lstrip("#") for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[,\s]+", text) if ("," in text or " " in text) else [text]
    return [p.strip().lstrip("#") for p in parts if p.strip()]


# ----------------------------------------------------------------- fence mask
def fence_mask(lines: list[str]) -> tuple[list[bool], list[str]]:
    """Mark which lines sit inside a fenced code block, plus each line's language.

    Correctness detail: a closing fence must use the same character and be at
    least as long as the opener, so a 4-backtick run inside a 3-backtick block
    does not close it.
    """
    inside = [False] * len(lines)
    langs = [""] * len(lines)
    opener: str | None = None
    lang = ""
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if opener is None:
            if m:
                opener, lang = m.group(1), m.group(2).lower()
                inside[i], langs[i] = True, lang
            continue
        inside[i], langs[i] = True, lang
        if m and m.group(1)[0] == opener[0] and len(m.group(1)) >= len(opener) and not m.group(2):
            opener, lang = None, ""
    return inside, langs


# -------------------------------------------------------------------- cleaning
def _protect_fences(text: str) -> tuple[str, list[str]]:
    """Swap fenced blocks for sentinels so regex cleaning cannot mangle code."""
    lines = text.split("\n")
    inside, langs = fence_mask(lines)
    out: list[str] = []
    blocks: list[str] = []
    buf: list[str] = []
    for i, line in enumerate(lines):
        if inside[i]:
            buf.append(line)
            is_last = i == len(lines) - 1 or not inside[i + 1]
            if is_last:
                block = "\n".join(buf)
                lang = next((l for l in langs[i - len(buf) + 1:i + 1] if l), "")
                if lang in config.DROP_FENCE_LANGS:
                    out.append("")            # plugin query, not authored content
                else:
                    out.append(_SENTINEL.format(len(blocks)))
                    blocks.append(block)
                buf = []
        else:
            out.append(line)
    return "\n".join(out), blocks


def _restore_fences(text: str, blocks: list[str]) -> str:
    for i, block in enumerate(blocks):
        text = text.replace(_SENTINEL.format(i), block)
    return text


def _strip_callouts(text: str) -> str:
    """Unwrap blockquotes and callouts into plain prose.

    A warning callout header becomes "Warning: <title>". We keep the callout
    type as a word because it is genuinely searchable ("what did I flag as a
    warning about X"), and we drop the quote markers because they add nothing
    to an embedding and interfere with paragraph-boundary splitting.

    Tradeoff: quoted text is no longer visually distinct from your own prose in
    a retrieved chunk. Citations still point at the note, so it stays checkable.
    """
    out = []
    in_quote = False
    for line in text.split("\n"):
        if BLOCKQUOTE_RE.match(line):
            body = BLOCKQUOTE_RE.sub("", line)
            head = CALLOUT_HEAD_RE.match(body)
            if head and not in_quote:
                kind, _fold, title = head.group(1), head.group(2), head.group(3)
                body = (f"{kind.capitalize()}: {title}".strip().rstrip(":")).strip()
            out.append(body)
            in_quote = True
        else:
            out.append(line)
            if not line.strip():
                in_quote = False
    return "\n".join(out)


def _resolve_wikilinks(text: str, meta: NoteMeta) -> str:
    """Replace links with their human-readable words; record the graph edges.

    Transclusions are deliberately NOT inlined. Inlining copies the target's
    text into this note, so the same passage gets embedded twice and both copies
    compete for the same top-k slots -- and worse, a citation would then
    attribute that text to the wrong note. We leave a marker and record the
    edge; stage 2 can follow it explicitly if you decide you want that.
    """
    def sub(m: re.Match) -> str:
        bang = m.group(1)
        target = (m.group(2) or "").strip()
        heading = (m.group(3) or "").lstrip("#").strip()
        alias = m.group(4)
        name = PurePosixPath(target).name if target else ""

        if bang:
            if name and PurePosixPath(name).suffix.lower() in config.ATTACHMENT_SUFFIXES:
                return ""                                    # image/PDF embed
            if name:
                meta.embeds.append(target)
                return f"[embeds: {name}]"
            return heading                                   # same-note embed
        if target:
            meta.links.append(target)
        if alias:
            return alias.strip()
        if not target:
            return heading                                   # same-note link
        return f"{name} {heading}".strip() if heading else name

    text = WIKILINK_RE.sub(sub, text)

    def sub_md(m: re.Match) -> str:
        bang, label, href = m.group(1), m.group(2), m.group(3)
        if bang:
            return ""                                        # inline image
        if href.lower().endswith(".md"):
            meta.links.append(href)
        return label or href

    return MDLINK_RE.sub(sub_md, text)


def clean_body(text: str, meta: NoteMeta) -> str:
    """Full cleaning pass for one section body, fence-safe."""
    text, blocks = _protect_fences(text)
    text = OBSIDIAN_COMMENT_RE.sub("", text)
    text = HTML_COMMENT_RE.sub("", text)
    text = _strip_callouts(text)
    text = HIGHLIGHT_RE.sub(r"\1", text)
    text = FOOTNOTE_RE.sub("", text)
    text = _resolve_wikilinks(text, meta)
    meta.tags.extend(INLINE_TAG_RE.findall(text))
    text = INLINE_TAG_RE.sub(r"\1", text)        # keep the word, drop the hash
    text = BLOCK_REF_RE.sub("", text)            # trailing block ids
    text = _restore_fences(text, blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------------- sections
def split_sections(body: str, first_line: int, note_title: str) -> list[Section]:
    """Split a note body into heading-scoped sections, ignoring fenced code."""
    lines = body.split("\n")
    inside, _ = fence_mask(lines)

    sections: list[Section] = []
    current = Section(level=0, heading=note_title, trail=[], start_line=first_line)
    stack: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        m = None if inside[i] else HEADING_RE.match(line)
        if not m:
            current.lines.append(line)
            continue
        sections.append(current)
        level, heading = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        trail = [h for _, h in stack]
        stack.append((level, heading))
        current = Section(level=level, heading=heading, trail=trail,
                          start_line=first_line + i)
    sections.append(current)
    return sections


def first_h1(body: str) -> str:
    """The first level-1 heading outside any code fence, if there is one."""
    lines = body.split("\n")
    inside, _ = fence_mask(lines)
    for i, line in enumerate(lines):
        if inside[i]:
            continue
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return ""


def parse_note(text: str, rel_path: str) -> tuple[NoteMeta, list[Section]]:
    """Parse one note into (metadata, cleaned sections)."""
    fm_text, body, body_line = split_frontmatter(text)
    fm = parse_frontmatter(fm_text) if fm_text else {}

    # Title precedence: frontmatter, then the first H1, then the filename.
    # The filename is last because Obsidian filenames are often abbreviations
    # ("VW.md" for West Virginia) -- fine as a file label, but noise at the
    # front of every embedded chunk in the note. The filename still reaches
    # BM25 through meta_text, so nothing becomes unsearchable by dropping it.
    meta = NoteMeta(
        title=(str(fm.get("title") or "").strip()
               or first_h1(body)
               or PurePosixPath(rel_path).stem),
        aliases=_as_list(fm.get("aliases") or fm.get("alias")),
        tags=_as_list(fm.get("tags") or fm.get("tag")),
        frontmatter=fm,
    )
    sections = split_sections(body, body_line, meta.title)
    for sec in sections:
        sec.lines = clean_body("\n".join(sec.lines), meta).split("\n")

    meta.tags = _dedupe(meta.tags)
    meta.aliases = _dedupe(meta.aliases)
    meta.links = _dedupe(meta.links)
    meta.embeds = _dedupe(meta.embeds)
    return meta, sections


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
