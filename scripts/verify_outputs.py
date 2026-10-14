#!/usr/bin/env python3
"""Verify generated notebook artifacts before committing."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import argparse
import ast
from pathlib import Path

from generate_notebook import TOPICS, read_detailed_answers
from research_updates import MANIFEST, UPDATES, SOURCES, SOURCE_MAP, curated_answers, split_answers


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-24"
EXPECTED = {
    "favorites_extracted_min": 315,
    "favorites_relevant_min": 204,
    "topics_min": 36,
}
TOKEN_RE = re.compile("|".join([re.escape("xsec_" + "token"), r"AB[A-Za-z0-9_-]{10,}"]))
PRIVATE_MARKERS = [
    "/Users/" + "didi",
    "Data Analysis/" + "xhs-llm-interview-notes",
    "5c8455f3000000001102" + "efd5",
    "uohzey" + "0519",
    "ssh.github.com/" + "uohzey" + "0519",
]
PRIVATE_RE = re.compile(
    "|".join([re.escape(marker) for marker in PRIVATE_MARKERS] + [r"https://www\.xiaohongshu\.com/user/profile/[0-9a-f]{24}"])
)
SCAN_PATHS = [
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "notes",
    ROOT / "data" / "processed",
    ROOT / "data" / "research",
    ROOT / "scripts",
    ROOT / "skills",
    ROOT / "tests",
]


def fail(message: str) -> None:
    raise SystemExit(f"verification failed: {message}")


def read_sources() -> list[dict[str, str]]:
    path = ROOT / "data" / "processed" / "sources.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scan_files() -> list[Path]:
    """Scan output files while excluding gitignored Python bytecode caches."""
    files: list[Path] = []
    for path in SCAN_PATHS:
        candidates = [path] if path.is_file() else path.rglob("*")
        files.extend(
            p for p in candidates
            if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
        )
    return files


def scan_for_tokens() -> list[str]:
    hits: list[str] = []
    for path in scan_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if TOKEN_RE.search(text):
            hits.append(str(path.relative_to(ROOT)))
    return hits


def scan_for_private_markers() -> list[str]:
    hits: list[str] = []
    for path in scan_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PRIVATE_RE.search(text):
            hits.append(str(path.relative_to(ROOT)))
    return hits


def raw_ignore_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--ignored", "--short", "data/raw"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated notebook artifacts.")
    parser.add_argument("--target-discovery-related", type=int, default=0)
    parser.add_argument("--allow-private-output", action="store_true", help="Allow local-only favorite order/profile URLs in generated output.")
    args = parser.parse_args()

    summary_path = ROOT / "data" / "processed" / f"run_summary_{DATE}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED.items():
        summary_key = key.removesuffix("_min")
        actual = int(summary.get(summary_key, 0))
        if actual < expected:
            fail(f"{summary_key}: expected at least {expected}, got {actual}")

    sources = read_sources()
    urls = [row["url"] for row in sources]
    if len(sources) != int(summary.get("extracted", len(sources))):
        fail(f"sources rows disagree with summary: summary={summary.get('extracted')}, csv={len(sources)}")
    if len(set(urls)) != len(urls):
        fail("sources.csv contains duplicate URLs")

    relevant = sum(row["is_llm_related"] == "yes" for row in sources)
    if relevant != int(summary.get("relevant", relevant)):
        fail(f"relevant rows disagree with summary: summary={summary.get('relevant')}, csv={relevant}")
    favorites_relevant = sum(
        row.get("source_origin", "favorites") == "favorites" and row["is_llm_related"] == "yes"
        for row in sources
    )
    if favorites_relevant < EXPECTED["favorites_relevant_min"]:
        fail(f"favorites relevant rows: expected at least {EXPECTED['favorites_relevant_min']}, got {favorites_relevant}")
    discover_relevant = sum(
        row.get("source_origin") == "discover" and row["is_llm_related"] == "yes"
        for row in sources
    )
    if args.target_discovery_related and discover_relevant < args.target_discovery_related:
        fail(f"discover relevant rows: expected at least {args.target_discovery_related}, got {discover_relevant}")

    token_hits = scan_for_tokens()
    if token_hits:
        fail("possible tokenized URL leak in tracked outputs: " + ", ".join(token_hits))
    private_hits = scan_for_private_markers()
    if private_hits:
        fail("possible private marker leak in tracked outputs: " + ", ".join(private_hits))
    if not args.allow_private_output:
        source_private_rows = [
            row["source_id"]
            for row in sources
            if row.get("favorite_order") or row.get("collect_url")
        ]
        if source_private_rows:
            preview = ", ".join(source_private_rows[:8])
            fail(f"sources.csv contains private favorite_order/collect_url values: {preview}")

    detailed_answers = read_detailed_answers()
    # Detect duplicate sections before the legacy dict parser can hide them.
    split_answers((ROOT / "data/processed/detailed_answers.md").read_text(encoding="utf-8"))
    expected_answer_keys = {topic.key for topic in TOPICS}
    actual_answer_keys = set(detailed_answers)
    if actual_answer_keys != expected_answer_keys:
        missing = sorted(expected_answer_keys - actual_answer_keys)
        extra = sorted(actual_answer_keys - expected_answer_keys)
        fail(f"detailed answer key mismatch; missing={missing}, extra={extra}")

    short_answers = [key for key, value in detailed_answers.items() if len(value) < 600]
    if short_answers:
        fail("detailed answers too short: " + ", ".join(sorted(short_answers)))

    master_text = (ROOT / "notes" / "llm-interview-master.md").read_text(encoding="utf-8")
    code_block_count = master_text.count("```")
    if code_block_count < 40:
        fail(f"expected detailed code/formula blocks in master note, got fence count {code_block_count}")
    if master_text.count("**面试官常问：**") < EXPECTED["topics_min"]:
        fail("master note must include interviewer question lists for every topic")
    if master_text.count("$$") < 80:
        fail("master note must include display-math $$ blocks for formulas")
    if master_text.count("```text") > 24:
        fail("too many text fences remain; formulas should use $$ display math")

    questions = [re.sub(r"[\W_]+", "", t.question).casefold() for t in TOPICS]
    if len(questions) != len(set(questions)):
        fail("duplicate normalized topic questions")
    if len(SOURCE_MAP) != len(SOURCES) or len({s["url"] for s in SOURCES}) != len(SOURCES):
        fail("duplicate research source IDs or URLs")
    if len(UPDATES) != len(MANIFEST["topics"]):
        fail("duplicate research topic keys")
    bank = (ROOT / "data/processed/question_bank.md").read_text(encoding="utf-8")
    all_keys = {topic.key for topic in TOPICS}
    curated = curated_answers()
    for key, item in UPDATES.items():
        if not set(item["sources"]) <= set(SOURCE_MAP):
            fail(f"unknown research source for {key}")
        if not set(item["related"]) <= all_keys:
            fail(f"unknown related topic for {key}")
        body = detailed_answers[key]
        if body != curated[key] or body not in master_text:
            fail(f"stale curated answer in generated output: {key}")
        for marker in ["**面试官常问：**", "### 原理与关键公式", "### 代码或伪代码",
                       "### 技术取舍", "### 深入追问", "$$"]:
            if marker not in body:
                fail(f"missing {marker} in {key}")
        cited = set(re.findall(r"\bR\d{2}\b", body))
        if cited != set(item["sources"]):
            fail(f"research citations disagree with manifest for {key}: {sorted(cited)}")
    for topic in TOPICS:
        if master_text.count(f'<a id="{topic.key}"></a>') != 1:
            fail(f"missing/duplicate master anchor: {topic.key}")
        if bank.count(f"| {topic.question} |") != 1:
            fail(f"missing/duplicate question bank row: {topic.key}")
    python_blocks = 0
    for key, body in detailed_answers.items():
        if body.count("```") % 2 or body.count("$$") % 2:
            fail(f"unbalanced code/math fences: {key}")
        for index, code in enumerate(re.findall(r"```python\n(.*?)```", body, flags=re.S)):
            ast.parse(code, filename=f"{key}:python:{index}")
            python_blocks += 1

    raw_status = raw_ignore_status()
    if raw_status and not all(line.startswith("!! ") for line in raw_status):
        fail("data/raw has non-ignored files: " + "; ".join(raw_status))

    print(json.dumps({
        "status": "ok",
        "extracted": len(sources),
        "relevant": relevant,
        "discover_relevant": discover_relevant,
        "xhs_topics": summary["topics"],
        "topics": len(TOPICS),
        "detailed_answers": len(detailed_answers),
        "research_topics_added": sum(t["kind"] == "new" for t in UPDATES.values()),
        "research_topics_expanded": sum(t["kind"] == "expand" for t in UPDATES.values()),
        "research_sources": len(SOURCES),
        "python_blocks_syntax_checked": python_blocks,
        "raw_ignored_entries": len(raw_status),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        fail(str(exc))
