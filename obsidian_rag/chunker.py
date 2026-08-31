"""Turning parsed sections into retrieval units.

Why chunk by heading rather than fixed token windows
----------------------------------------------------
A fixed 512-token window cuts wherever it lands: mid-argument, mid-table, two
sentences into a section whose topic sentence is now in the previous chunk.
The retrieved text then reads as a fragment, and the model either hedges or
fills the gap with invention.

Headings are semantic boundaries a human already placed. Chunking on them means
every chunk is a unit somebody deliberately wrote as a unit.

The two failure modes of naive heading chunking, and the fixes:

  * Sections are wildly uneven. A note has an H2 with one sentence and another
    with 3,000 words. Fix: merge anything under MIN_CHARS into its neighbour,
    split anything over MAX_CHARS at paragraph boundaries.

  * A section body is often meaningless alone. Under "## Pricing" you wrote
    "they came down to 40k after the second call" -- no subject, no company.
    Retrieved bare, that chunk is unusable and unciteable. Fix: prepend the
    breadcrumb (note title > H1 > H2) to the embedded text, so both the
    embedding and the model see what the passage is *about*.

That second fix is the single highest-leverage decision in this file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from . import config
from .mdparse import NoteMeta, Section, fence_mask, parse_note


@dataclass
class Chunk:
    ord: int                # position within the note, 0-based
    breadcrumb: str         # "Note title > H1 > H2"
    heading: str            # the nearest heading
    start_line: int         # 1-based line in the source file, for jumping back
    body: str               # the text shown to the model
    embed_text: str         # the text actually sent to the embedding model
    meta_text: str          # extra lexical field for BM25 (path/tags/aliases)
    text_hash: str          # sha256(embed_text) -- the incremental-index key


# ---------------------------------------------------------------- paragraphs
def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, but never inside a fenced code block.

    Splitting inside a fence would produce a chunk with an unclosed fence and a
    chunk of orphaned code, both of which read as corrupt.
    """
    lines = text.split("\n")
    inside, _ = fence_mask(lines)
    paras: list[str] = []
    buf: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip() and not inside[i]:
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]


def _hard_split(para: str, limit: int) -> list[str]:
    """Last resort for a single oversized paragraph (a pasted file, a huge table).

    Splits on line boundaries so we never cut mid-line.
    """
    out, buf = [], []
    size = 0
    for line in para.split("\n"):
        if size + len(line) > limit and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        out.append("\n".join(buf))
    return out


# --------------------------------------------------------------------- units
@dataclass
class _Unit:
    """A candidate chunk before merging: one section, or one slice of one."""
    trail: list[str]
    heading: str
    level: int
    start_line: int
    text: str


def _section_units(sec: Section) -> list[_Unit]:
    """One section becomes one unit, or several if it is over MAX_CHARS."""
    body = "\n".join(sec.lines).strip()
    if not body:
        return []

    # The heading is written into the body so that merging is plain
    # concatenation and the model still sees the document structure.
    prefix = f"{'#' * max(sec.level, 1)} {sec.heading}\n\n" if sec.level else ""
    full = prefix + body
    if len(full) <= config.MAX_CHARS:
        return [_Unit(sec.trail, sec.heading, sec.level, sec.start_line, full)]

    paras: list[str] = []
    for para in _paragraphs(body):
        paras.extend(_hard_split(para, config.MAX_CHARS) if len(para) > config.MAX_CHARS else [para])

    units: list[_Unit] = []
    buf: list[str] = []
    size = 0
    for para in paras:
        if buf and size + len(para) > config.TARGET_CHARS:
            units.append(_Unit(sec.trail, sec.heading, sec.level, sec.start_line,
                               prefix + "\n\n".join(buf)))
            # Carry the tail paragraph forward so a claim split across the
            # boundary still appears whole in at least one chunk.
            carry = buf[-config.OVERLAP_PARAS:] if config.OVERLAP_PARAS else []
            # ...but only when the carried text is small relative to the
            # target. With long paragraphs, an unconditional carry makes every
            # chunk "overlap + one full paragraph" and blows past MAX_CHARS.
            if sum(len(p) for p in carry) > config.TARGET_CHARS // 2:
                carry = []
            buf = carry
            size = sum(len(p) + 2 for p in buf)
        buf.append(para)
        size += len(para) + 2
    if buf:
        units.append(_Unit(sec.trail, sec.heading, sec.level, sec.start_line,
                           prefix + "\n\n".join(buf)))
    return units


def _same_branch(anchor: _Unit, other: _Unit) -> bool:
    """True if `other` is a sibling or a descendant of `anchor` in the heading tree.

    This is the guard that stops merging from crossing unrelated topics. Without
    it, a short "### Tourism To-Do List" happily absorbs the following
    "## Historical Context" -- two different subjects in one chunk, labelled
    with the first one's breadcrumb. The chunk then embeds as a blend of both
    and cites the wrong section.
    """
    if other.trail == anchor.trail:                    # siblings
        return True
    extended = list(anchor.trail) + [anchor.heading]   # descendants
    return other.trail[:len(extended)] == extended


def _merge_small(units: list[_Unit]) -> list[list[_Unit]]:
    """Group consecutive units so no group is pointlessly small.

    A chunk of "## Status\\n\\nDone." embeds to noise: too little signal to
    match anything, but it still occupies a top-k slot. Merging it with the
    sections around it produces something with actual content -- but only with
    sections it is actually related to, hence _same_branch.
    """
    groups: list[list[_Unit]] = []
    buf: list[_Unit] = []
    size = 0
    for unit in units:
        if buf and (size >= config.MIN_CHARS
                    or size + len(unit.text) > config.MAX_CHARS
                    or not _same_branch(buf[0], unit)):
            groups.append(buf)
            buf, size = [], 0
        buf.append(unit)
        size += len(unit.text) + 2
    if buf:
        groups.append(buf)
    return groups


# -------------------------------------------------------------------- chunks
def _breadcrumb(rel_path: str, meta: NoteMeta, unit: _Unit) -> str:
    """Folder path + note title + heading trail, deduplicated.

    The folders come first because that is the reading order a person would
    use, and because it puts the broadest context where an embedding model
    weighs it as scene-setting rather than as the subject.
    """
    parts: list[str] = []
    if config.INCLUDE_FOLDER_IN_BREADCRUMB:
        parts.extend(PurePosixPath(rel_path).parent.parts)
    parts.append(meta.title)
    parts.extend(unit.trail)
    if unit.level and unit.heading and unit.heading not in parts:
        parts.append(unit.heading)
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return " > ".join(out)


def _meta_text(rel_path: str, meta: NoteMeta) -> str:
    """The lexical-only field: things worth matching on but not worth embedding.

    Folder names and file paths are strong signals for a keyword search
    ("that thing in projects/acme") and pure noise in a semantic one.
    """
    bits = [rel_path.replace("/", " ").replace("-", " ").replace("_", " ")]
    if meta.aliases:
        bits.append(" ".join(meta.aliases))
    if meta.tags:
        bits.append(" ".join(meta.tags))
    return "  ".join(bits)


def chunk_note(rel_path: str, text: str) -> tuple[NoteMeta, list[Chunk]]:
    """Parse and chunk one note. The only entry point this module needs to expose."""
    meta, sections = parse_note(text, rel_path)

    units: list[_Unit] = []
    for sec in sections:
        units.extend(_section_units(sec))

    meta_text = _meta_text(rel_path, meta)
    chunks: list[Chunk] = []

    for i, group in enumerate(_merge_small(units)):
        head = group[0]
        body = "\n\n".join(u.text for u in group).strip()
        breadcrumb = _breadcrumb(rel_path, meta, head)

        # What we embed is not what we display. The header lines give the
        # embedding model the context the body assumes; aliases and tags are
        # included because they are usually the proper nouns you search by,
        # and bge-m3 handles them fine as a short prefix.
        if config.INCLUDE_META_IN_EMBEDDING:
            header = [breadcrumb]
            if meta.aliases:
                header.append("aliases: " + ", ".join(meta.aliases))
            if meta.tags:
                header.append("tags: " + ", ".join(meta.tags))
            embed_text = "\n".join(header) + "\n\n" + body
        else:
            embed_text = f"{breadcrumb}\n\n{body}"

        chunks.append(Chunk(
            ord=i,
            breadcrumb=breadcrumb,
            heading=head.heading,
            start_line=head.start_line,
            body=body,
            embed_text=embed_text,
            meta_text=meta_text,
            text_hash=hashlib.sha256(embed_text.encode("utf-8")).hexdigest(),
        ))
    return meta, chunks
