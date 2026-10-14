"""Curated primary-source research overlays, independent of XHS frequency counts."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "data" / "research"
MANIFEST = json.loads((RESEARCH / "topic_updates.json").read_text(encoding="utf-8"))
UPDATES = {item["key"]: item for item in MANIFEST["topics"]}
SOURCES = json.loads((RESEARCH / "sources.json").read_text(encoding="utf-8"))
SOURCE_MAP = {item["source_id"]: item for item in SOURCES}


def split_answers(text: str) -> tuple[str, dict[str, str]]:
    parts = re.split(r"(?m)^## ", text)
    sections = {}
    for part in parts[1:]:
        key, _, body = part.partition("\n")
        key = key.strip()
        if key in sections:
            raise ValueError(f"duplicate answer key: {key}")
        sections[key] = body.strip()
    return parts[0].rstrip(), sections


def curated_answers() -> dict[str, str]:
    _, answers = split_answers((RESEARCH / "answers.md").read_text(encoding="utf-8"))
    if set(answers) != set(UPDATES):
        raise ValueError("research answer keys disagree with topic_updates.json")
    result = {}
    for key, body in answers.items():
        refs = []
        for sid in UPDATES[key]["sources"]:
            source = SOURCE_MAP[sid]
            refs.append(f"- [{sid}]({source['url']}): {source['title']} — "
                        f"{source['locator']}（核对：{source['accessed']}）。")
        result[key] = body + "\n\n### 研究来源\n\n" + "\n".join(refs)
    return result


def sync_curated_answers() -> None:
    """Update protected sections; repeated runs are byte-identical."""
    path = ROOT / "data" / "processed" / "detailed_answers.md"
    header, sections = split_answers(path.read_text(encoding="utf-8"))
    sections.update(curated_answers())
    row = ("| v1.2 | 2026-10-13 | Primary-source research; no new XHS rows | 0 | 3 | "
           "1199 | 45 | Deepened 6 clusters; added SAO, AttnRes and CED/CSA2. |")
    if "| v1.2 |" not in header:
        header = re.sub(r"(?m)^(\| v1\.1 \|.*)$", lambda m: m[0] + "\n" + row, header)
    header = re.sub(r"(?m)^- Normalized topic sections:.*$",
                    f"- Normalized topic sections: {len(sections)} (42 XHS baseline + 3 research topics)",
                    header)
    note = ("Research refresh: 2026-10-13. Curated bodies in data/research/answers.md "
            "override nine sections; XHS corpus counts and last corpus refresh are unchanged.")
    if note not in header:
        header += "\n\n" + note
    text = header + "\n\n" + "\n\n".join(f"## {k}\n\n{v}" for k, v in sections.items()) + "\n"
    path.write_text(text, encoding="utf-8")
