#!/usr/bin/env python3
"""Build inventory JSON and sources.csv from xhs_inventory_steps JSONL."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TODAY = "2026-05-24"


def classify_title(title: str) -> tuple[bool, str]:
    patterns = {
        "rag_agent_tool": r"(?i)(agent|rag|claude|code|工具|检索|function call|mcp)",
        "post_training_rl": r"(?i)(rlhf|dpo|grpo|ppo|强化学习|后训练|reward|奖励|kl|sft|lampo)",
        "attention_kv_cache": r"(?i)(attention|gqa|mha|mqa|mla|kv|cache|显存|softmax)",
        "moe_sparse": r"(?i)(moe|稀疏|dense|专家)",
        "training_optimization": r"(?i)(训练|优化器|optimizer|loss|crossentropy|mlp|反向|梯度|muon)",
        "serving_performance": r"(?i)(推理|部署|量化|算子|显存|吞吐|latency|flash)",
        "company_interviews": r"(?i)(面试|一面|二面|三面|oc|字节|阿里|腾讯|百度|快手|滴滴|算法岗|面经|手撕)",
        "architecture": r"(?i)(transformer|deepseek|qwen|llama|架构|模型|大模型|llm|v4)",
    }
    hits = [name for name, pat in patterns.items() if re.search(pat, title or "")]
    return bool(hits), ";".join(hits)


def main() -> None:
    steps_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data/raw/xhs_inventory_steps_2026-05-24.jsonl"
    raw_path = REPO_ROOT / f"data/raw/xhs_inventory_{TODAY}.json"
    sources_path = REPO_ROOT / "data/processed/sources.csv"

    seen: dict[str, dict] = {}
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        for item in payload.get("items", []):
            note_id = item.get("note_id")
            if not note_id:
                continue
            current = seen.get(note_id, {})
            seen[note_id] = {**current, **item}

    items = sorted(seen.values(), key=lambda x: (x.get("y", 0), x.get("note_id", "")))
    for idx, item in enumerate(items, start=1):
        is_related, categories = classify_title(item.get("title", ""))
        item["favorite_order"] = idx
        item["is_llm_related_by_title"] = is_related
        item["categories_by_title"] = categories
        item["processed_at"] = TODAY

    raw_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    include_personal_order = os.environ.get("XHS_INCLUDE_PERSONAL_ORDER") == "1"

    with sources_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "title",
                "author",
                "url",
                "status",
                "categories",
                "notes",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "source_id": f"xhs_{item['note_id'][:8]}",
                    "title": item.get("title", ""),
                    "author": item.get("author", ""),
                    "url": item.get("url", ""),
                    "status": "pending-title-related" if item["is_llm_related_by_title"] else "pending-review",
                    "categories": item.get("categories_by_title", ""),
                    "notes": (
                        f"order={item['favorite_order']}; " if include_personal_order else ""
                    ) + f"likes={item.get('like_hint', '')}; collected={TODAY}",
                }
            )

    print(json.dumps({
        "total": len(items),
        "title_related": sum(1 for item in items if item["is_llm_related_by_title"]),
        "raw": str(raw_path),
        "sources": str(sources_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
