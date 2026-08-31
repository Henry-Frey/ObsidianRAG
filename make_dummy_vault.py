"""Generate a dummy vault for development, shaped like a real one.

    python make_dummy_vault.py                 # 120 notes into ./dummy-vault
    python make_dummy_vault.py --notes 600     # test a larger vault

This is a dev tool, not part of the RAG system. It exists so we can tune
chunking and retrieval without touching your real notes, and so we can measure
timings at a realistic size before you scale up.

It deliberately reproduces the things that make real vaults awkward:

  * heavy template boilerplate in daily and meeting notes, which produces
    byte-identical chunks that all compete for the same top-k slots
  * wildly uneven note lengths, from one-line stubs to long reference pages
  * headings inside code fences
  * mixed languages
  * a .obsidian folder, attachments, and a sync-conflict copy

Output is deterministic (seeded), so two runs give the same vault.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

SEED = 20250831

PROJECTS = [
    "Atlas Migration", "Beacon Rollout", "Cobalt Pricing", "Delta Ingest",
    "Ember Search", "Foxglove API", "Granite Billing", "Harbour Sync",
    "Iris Dashboard", "Juniper Auth", "Kestrel Cache", "Lantern Reports",
]
PEOPLE = [
    "Ana Ruiz", "Ben Okafor", "Clara Lindqvist", "Dmitri Volkov", "Elena Sato",
    "Farid Haddad", "Grace Mwangi", "Hugo Bernard", "Ines Ferreira", "Jonas Weber",
    "Karin Falk", "Liam O'Donnell", "Mei Tanaka", "Nadia Petrova", "Omar Aziz",
]
TOPICS = [
    "Retrieval evaluation", "Vector index tradeoffs", "BM25 tuning",
    "Chunking strategies", "Embedding drift", "Query rewriting",
    "Reranking models", "Latency budgets", "Sharding notes",
    "Backup and restore", "Access control", "Schema migrations",
]
TAGS = ["project", "active", "archived", "meeting", "reference", "idea",
        "infra", "research", "todo", "decision"]

# ~15% of notes are in a second language. Swap this block for whichever
# language your vault actually uses -- it is here to exercise the multilingual
# embedding path and FTS5's diacritic folding, not to be realistic prose.
SPANISH = [
    "El equipo confirmo el presupuesto para el proximo trimestre.",
    "Revisar la documentacion de la migracion antes del viernes.",
    "Muller propuso reducir la latencia de busqueda a menos de 200 ms.",
    "Quedan pendientes las pruebas de integracion con el servicio de facturacion.",
    "Se acordo posponer el lanzamiento hasta resolver el problema de indexacion.",
]
ENGLISH = [
    "The retrieval quality dropped noticeably after we switched embedding models.",
    "We agreed to keep the hybrid scoring weights fixed until the eval set grows.",
    "Latency is dominated by the embedding call, not the vector search itself.",
    "Chunk boundaries matter more than the embedding model for short notes.",
    "Proper nouns are consistently missed by pure semantic search on this corpus.",
    "The index rebuild takes under a minute, so incremental updates are a nicety.",
    "Nobody has looked at the archived notes in months, but they still rank.",
    "Citations need note paths, not just titles, because titles collide.",
]

# Every daily note opens with this. Real vaults do exactly this, and it is why
# retrieval returns eight identical chunks unless you dedupe.
DAILY_TEMPLATE = """## Focus

## Notes

## Follow-ups

"""


def sentences(rng: random.Random, n: int, spanish: bool) -> str:
    pool = SPANISH if spanish else ENGLISH
    return " ".join(rng.choice(pool) for _ in range(n))


def frontmatter(rng: random.Random, title: str, tags: list[str],
                aliases: list[str] | None = None) -> str:
    lines = ["---", f"title: {title}"]
    if aliases:
        lines.append("aliases: [" + ", ".join(aliases) + "]")
    lines.append("tags:")
    lines.extend(f"  - {t}" for t in tags)
    lines.append(f"created: 2025-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}")
    lines.append("---\n")
    return "\n".join(lines)


def project_note(rng: random.Random, name: str, others: list[str]) -> str:
    spanish = rng.random() < 0.15
    body = [frontmatter(rng, name, rng.sample(TAGS, 2), aliases=[name.split()[0]])]
    body.append(f"Owner: [[people/{rng.choice(PEOPLE)}]]\n")
    body.append(sentences(rng, 3, spanish) + "\n")

    body.append("## Status\n")
    body.append(sentences(rng, rng.randint(2, 5), spanish) + "\n")

    if rng.random() < 0.5:
        body.append("> [!warning] Blocked\n> " + sentences(rng, 1, spanish) + "\n")

    body.append("## Decisions\n")
    for _ in range(rng.randint(1, 4)):
        body.append(f"- {sentences(rng, 1, spanish)}")
    body.append("")

    if rng.random() < 0.4:
        body.append("## Runbook\n")
        body.append("```bash")
        body.append("# Deploy step one -- this heading must not split the note")
        body.append("## And neither must this")
        body.append("set -euo pipefail")
        body.append(f"deploy --target {name.lower().replace(' ', '-')}")
        body.append("```\n")

    body.append("## Related\n")
    for other in rng.sample(others, min(3, len(others))):
        body.append(f"- [[projects/{other}|{other}]]")
    body.append("")

    if rng.random() < 0.3:
        body.append(f"![[reference/{rng.choice(TOPICS)}]]\n")
    return "\n".join(body)


def person_note(rng: random.Random, name: str) -> str:
    spanish = rng.random() < 0.15
    first = name.split()[0]
    body = [frontmatter(rng, name, ["person"], aliases=[first])]
    body.append(sentences(rng, 2, spanish) + "\n")
    body.append("## Context\n")
    body.append(sentences(rng, rng.randint(1, 3), spanish) + "\n")
    body.append("## Threads\n")
    for _ in range(rng.randint(1, 3)):
        body.append(f"- [[projects/{rng.choice(PROJECTS)}]] -- {sentences(rng, 1, spanish)}")
    return "\n".join(body) + "\n"


def meeting_note(rng: random.Random, idx: int) -> tuple[str, str]:
    spanish = rng.random() < 0.15
    month, day = rng.randint(1, 8), rng.randint(1, 28)
    topic = rng.choice(PROJECTS)
    name = f"meetings/2025-{month:02d}-{day:02d} {topic}.md"
    body = [frontmatter(rng, f"{topic} sync", ["meeting"])]
    body.append("## Attendees\n")
    for p in rng.sample(PEOPLE, rng.randint(2, 4)):
        body.append(f"- [[people/{p}]]")
    body.append("\n## Discussion\n")
    body.append(sentences(rng, rng.randint(3, 8), spanish) + "\n")
    body.append("## Actions\n")
    for _ in range(rng.randint(1, 4)):
        body.append(f"- [ ] {sentences(rng, 1, spanish)} #todo")
    return name, "\n".join(body) + "\n"


def daily_note(rng: random.Random, month: int, day: int) -> str:
    spanish = rng.random() < 0.15
    body = [frontmatter(rng, f"2025-{month:02d}-{day:02d}", ["daily"])]
    body.append(DAILY_TEMPLATE)
    # Roughly a third of daily notes are the bare template and nothing else --
    # exactly like a real vault, and exactly what pollutes retrieval.
    if rng.random() > 0.35:
        body.append(sentences(rng, rng.randint(1, 4), spanish) + "\n")
        if rng.random() < 0.4:
            body.append(f"Spoke to [[people/{rng.choice(PEOPLE)}]] about "
                        f"[[projects/{rng.choice(PROJECTS)}]].\n")
    return "\n".join(body)


def reference_note(rng: random.Random, topic: str) -> str:
    spanish = rng.random() < 0.1
    body = [frontmatter(rng, topic, ["reference", "research"])]
    body.append(sentences(rng, 2, spanish) + "\n")
    for section in ("Background", "Approach", "Tradeoffs", "Open questions"):
        body.append(f"## {section}\n")
        for _ in range(rng.randint(1, 4)):
            body.append(sentences(rng, rng.randint(3, 9), spanish) + "\n")
        if section == "Approach" and rng.random() < 0.5:
            body.append("```python")
            body.append("# scoring helper -- fence must survive chunking")
            body.append("def rrf(rank: int, k: int = 60) -> float:")
            body.append("    return 1.0 / (k + rank)")
            body.append("```\n")
    body.append("%% private: revisit this before sharing %%\n")
    return "\n".join(body)


def build(root: Path, target: int) -> dict:
    rng = random.Random(SEED)
    for sub in ("projects", "people", "meetings", "daily", "reference",
                "attachments", ".obsidian/plugins"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    written = 0

    for name in PROJECTS:
        (root / "projects" / f"{name}.md").write_text(
            project_note(rng, name, [p for p in PROJECTS if p != name]), encoding="utf-8")
        written += 1

    for name in PEOPLE:
        (root / "people" / f"{name}.md").write_text(
            person_note(rng, name), encoding="utf-8")
        written += 1

    for topic in TOPICS:
        (root / "reference" / f"{topic}.md").write_text(
            reference_note(rng, topic), encoding="utf-8")
        written += 1

    idx = 0
    while written < target:
        if idx % 2 == 0:
            name, text = meeting_note(rng, idx)
            path = root / name
            if not path.exists():
                path.write_text(text, encoding="utf-8")
                written += 1
        else:
            month, day = rng.randint(1, 8), rng.randint(1, 28)
            path = root / "daily" / f"2025-{month:02d}-{day:02d}.md"
            if not path.exists():
                path.write_text(daily_note(rng, month, day), encoding="utf-8")
                written += 1
        idx += 1
        if idx > target * 20:
            break

    # Things that must be ignored by the indexer.
    (root / ".obsidian" / "plugins" / "readme.md").write_text(
        "# plugin\nshould never be indexed\n", encoding="utf-8")
    (root / "attachments" / "diagram.png").write_bytes(b"\x89PNG\r\n")
    (root / "projects" / "Atlas Migration.sync-conflict-20250830-113355-ABCDEF.md"
     ).write_text("# stale\nold content\n", encoding="utf-8")

    return {"notes": written}


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a dummy Obsidian vault.")
    p.add_argument("--out", default="dummy-vault", help="output directory")
    p.add_argument("--notes", type=int, default=120, help="approximate note count")
    args = p.parse_args()

    root = Path(args.out).resolve()
    result = build(root, args.notes)
    print(f"wrote {result['notes']} notes to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
