"""Read-only discovery and reading of vault files.

This is the ONLY module in the project that opens a path inside the vault,
and it opens them exclusively in binary read mode ("rb"). There is no code
path here that creates, writes, moves, or deletes anything. If you audit one
file to satisfy yourself about the read-only guarantee, audit this one.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import config

# Windows file attributes marking a cloud placeholder (OneDrive Files
# On-Demand, Dropbox smart sync). stat.FILE_ATTRIBUTE_OFFLINE exists in the
# stdlib; RECALL_ON_DATA_ACCESS does not, so we spell it out.
FILE_ATTRIBUTE_OFFLINE = 0x00001000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
_PLACEHOLDER_BITS = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
)


class VaultError(RuntimeError):
    pass


@dataclass(frozen=True)
class NoteFile:
    """A note we intend to read. `rel` is the vault-relative key used everywhere
    else in the system -- always forward slashes, so the index is portable and
    stable regardless of which machine indexed it."""
    path: Path
    rel: str
    size: int
    mtime: float


def assert_outside_vault(target: Path, vault: Path) -> None:
    """Refuse to let the index (or anything else we write) live in the vault.

    Called at startup. This is belt-and-braces on top of the fact that nothing
    here opens a vault path for writing -- but it is cheap, and it also stops
    you sync-churning an 8 MB database through OneDrive by accident.
    """
    target = target.resolve()
    vault = vault.resolve()
    if target == vault or vault in target.parents:
        raise VaultError(
            f"Refusing to write inside the vault.\n"
            f"  vault:  {vault}\n"
            f"  target: {target}\n"
            f"Point INDEX_DIR somewhere outside the vault."
        )


def is_conflict_name(name: str) -> bool:
    """True for sync-conflict copies (Syncthing / Dropbox / OneDrive)."""
    return any(p.search(name) for p in config.CONFLICT_PATTERNS)


def placeholder_reason(st: os.stat_result) -> str | None:
    """Return a reason string if this file is a cloud placeholder, else None.

    Reading a placeholder forces the sync client to download it. Doing that
    across a whole vault turns 'index my notes' into 'download my notes',
    and fails outright when offline. We skip and report instead.
    """
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs & _PLACEHOLDER_BITS:
        return "cloud placeholder (not downloaded locally)"
    return None


def iter_notes(vault: Path) -> Iterator[NoteFile | tuple[str, str]]:
    """Walk the vault yielding NoteFile for readable notes.

    Yields ("skip", message) tuples for anything deliberately passed over, so
    the caller can report it rather than have files vanish silently.
    """
    vault = vault.resolve()
    if not vault.is_dir():
        raise VaultError(f"Vault path is not a directory: {vault}")

    for dirpath, dirnames, filenames in os.walk(vault):
        # Prune in place -- os.walk honours mutation of `dirnames`, so this
        # avoids descending into .obsidian at all rather than filtering later.
        dirnames[:] = [
            d for d in dirnames
            if d not in config.SKIP_DIRS
            and not (config.SKIP_HIDDEN_DIRS and d.startswith("."))
        ]
        dirnames.sort()

        for name in sorted(filenames):
            if Path(name).suffix.lower() not in config.NOTE_SUFFIXES:
                continue
            if name.startswith("."):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(vault).as_posix()

            if is_conflict_name(name):
                yield ("skip", f"{rel}  [sync conflict copy]")
                continue
            try:
                st = full.stat()
            except OSError as exc:
                yield ("skip", f"{rel}  [stat failed: {exc}]")
                continue

            reason = placeholder_reason(st)
            if reason:
                yield ("skip", f"{rel}  [{reason}]")
                continue
            if not stat.S_ISREG(st.st_mode):
                continue

            yield NoteFile(path=full, rel=rel, size=st.st_size, mtime=st.st_mtime)


def read_note(path: Path) -> tuple[str, str]:
    """Read a note read-only. Returns (text, sha256-of-raw-bytes).

    We hash the raw bytes rather than the decoded text so that an encoding
    change alone still counts as a change. We hash content rather than trusting
    mtime because (a) sync clients rewrite mtimes without changing content, and
    (b) a file caught mid-sync reads as garbage once, then self-heals on the
    next run when its hash differs again.
    """
    raw = path.read_bytes()                     # 'rb' -- never opened for write
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    return text.replace("\r\n", "\n").replace("\r", "\n"), digest
