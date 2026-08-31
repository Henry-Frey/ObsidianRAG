"""Extracting a note's date from its filename or frontmatter.

Supported, all normalised to ISO YYYY-MM-DD for storage:

    2026-08-31      ISO / Obsidian's default daily-note format
    31-08-2026      DD-MM-YYYY
    31.08.2026      DD.MM.YYYY
    31-08-26        DD-MM-YY
    31.08.26        DD.MM.YY

Four decisions worth understanding, because each one rules out a class of
silent misparse:

1. DAY FIRST, NEVER MONTH FIRST. "03-04-2026" is 3 April, not 4 March. There is
   no way to support DD-MM and MM-DD together -- they are indistinguishable
   whenever both numbers are <= 12, and guessing would corrupt roughly a third
   of dates with no error to show for it. US-style MM-DD-YYYY is deliberately
   not supported.

2. WHICH END HAS FOUR DIGITS DECIDES. 2026-08-31 is unambiguously ISO;
   31-08-2026 is unambiguously day-first. No heuristics needed.

3. THE SEPARATOR MUST MATCH ITSELF. "31-08.2026" is not a date, it is two
   numbers that happen to be adjacent. Enforced with a backreference.

4. FILENAME AND FRONTMATTER ONLY -- never body prose. In running text,
   "1.2.26" is far more often a version number than a date, and there is no
   reliable way to tell. Scanning bodies would quietly date hundreds of notes
   from changelogs and numbered lists.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import PurePosixPath

from . import config

# A date must start at the beginning of the string or after whitespace or an
# underscore. That accepts "31-08-2026 Standup" and "Meeting_31.08.26" while
# rejecting "v1.2.26" and "Note-1.2.26", where the digits belong to a version
# number. The trailing boundary is looser (a hyphen is allowed) so that
# "2026-08-31-Retro" still parses.
_LEAD = r"(?:(?<=^)|(?<=[\s_]))"
_TAIL = r"(?=$|[\s_\-])"

_ISO = re.compile(_LEAD + r"(\d{4})-(\d{1,2})-(\d{1,2})" + _TAIL)
_DMY4 = re.compile(_LEAD + r"(\d{1,2})([-.])(\d{1,2})\2(\d{4})" + _TAIL)
_DMY2 = re.compile(_LEAD + r"(\d{1,2})([-.])(\d{1,2})\2(\d{2})" + _TAIL)


def _expand_year(raw: str) -> int:
    """Two-digit year to four, pivoting on config.TWO_DIGIT_YEAR_PIVOT."""
    value = int(raw)
    if len(raw) == 4:
        return value
    return 2000 + value if value < config.TWO_DIGIT_YEAR_PIVOT else 1900 + value


def _iso(day: int, month: int, year: int) -> str | None:
    """Validate and format. datetime does the calendar work, including leap years.

    This is what rejects 32-13-2026 and 29-02-2025 without a line of
    month-length logic of our own.
    """
    try:
        return _dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def _from_match(pattern: re.Pattern, m: re.Match) -> str | None:
    if pattern is _ISO:
        return _iso(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    # Both day-first patterns: group(1)=day, group(3)=month, group(4)=year.
    return _iso(int(m.group(1)), int(m.group(3)), _expand_year(m.group(4)))


def find_date(text: str) -> str | None:
    """First valid date in `text`, as ISO, or None.

    ISO is tried first, then four-digit-year day-first, then two-digit. The
    ordering matters: without it, "31.08.2026" could match the two-digit
    pattern against "20" and silently produce the year 2020.
    """
    if not text:
        return None
    for pattern in (_ISO, _DMY4, _DMY2):
        for m in pattern.finditer(text):
            iso = _from_match(pattern, m)
            if iso:
                return iso
    return None


def note_date(rel_path: str, frontmatter: dict) -> tuple[str | None, str]:
    """Resolve a note's date. Returns (iso_date_or_None, source_label).

    Filename wins over frontmatter. In Obsidian the filename of a dated note is
    its *subject* date -- what the note is about -- while `created:` is file
    metadata that a template or a sync client may have written on a different
    day. When they disagree, the filename is the one you meant.

    The source label is stored alongside the date so a misparse is auditable
    later ("why does this note think it is from 1998?") instead of being an
    unexplained value in a column.
    """
    if config.DATE_FROM_FILENAME:
        iso = find_date(PurePosixPath(rel_path).stem)
        if iso:
            return iso, "filename"

    for key in config.DATE_FRONTMATTER_KEYS:
        value = frontmatter.get(key)
        if value is None:
            continue
        text = str(value[0] if isinstance(value, list) and value else value).strip()
        if not text:
            continue
        # Frontmatter values are usually the bare date, so try the whole field
        # before falling back to a bounded search inside it.
        iso = find_date(text) or find_date(f" {text} ")
        if iso:
            return iso, f"frontmatter:{key}"

    return None, ""
