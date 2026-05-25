#!/usr/bin/env python3
"""Generate comparison reports for processed Xiaohongshu LLM interview sources."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import generate_notebook


ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "processed" / "sources.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "run_summary_2026-05-24.json"
REPORT_PATH = ROOT / "docs" / "discovery_delta_report_2026-05-25.md"


def pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def split_cell(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def read_sources() -> list[dict[str, str]]:
    with SOURCES_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def origin_label(origin: str) -> str:
    return "收藏夹原始基线" if origin == "favorites" else "发现/搜索新增"


def count_by_origin(rows: list[dict[str, str]], field: str) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        origin = row.get("source_origin") or "favorites"
        for item in split_cell(row.get(field, "")):
            result[origin][item] += 1
    return result


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def build_report() -> str:
    rows = read_sources()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    relevant = [row for row in rows if row.get("is_llm_related") == "yes"]
    by_origin = {
        origin: [row for row in rows if row.get("source_origin") == origin]
        for origin in ("favorites", "discover")
    }
    relevant_by_origin = {
        origin: [row for row in relevant if row.get("source_origin") == origin]
        for origin in ("favorites", "discover")
    }

    status_by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        status_by_origin[row.get("source_origin") or "favorites"][row.get("extraction_status") or ""] += 1

    category_by_origin = count_by_origin(relevant, "categories")
    topic_by_origin = count_by_origin(relevant, "matched_topics")
    all_topic_keys = sorted(set(topic_by_origin["favorites"]) | set(topic_by_origin["discover"]))
    topic_defs = {topic.key: topic for topic in generate_notebook.TOPICS}

    topic_rows = []
    for key in all_topic_keys:
        topic = topic_defs.get(key)
        fav_count = topic_by_origin["favorites"][key]
        discover_count = topic_by_origin["discover"][key]
        total = fav_count + discover_count
        topic_rows.append(
            {
                "key": key,
                "question": topic.question if topic else key,
                "category": topic.category if topic else "",
                "favorites": fav_count,
                "discover": discover_count,
                "total": total,
                "delta": discover_count - fav_count,
            }
        )
    topic_rows.sort(key=lambda item: (-item["total"], item["question"]))

    category_rows = []
    for category in sorted(set(category_by_origin["favorites"]) | set(category_by_origin["discover"])):
        fav_count = category_by_origin["favorites"][category]
        discover_count = category_by_origin["discover"][category]
        category_rows.append([category, fav_count, discover_count, fav_count + discover_count, f"+{discover_count}"])
    category_rows.sort(key=lambda row: -int(row[3]))

    total_questions_by_origin = {
        origin: sum(int(row.get("question_count") or 0) for row in origin_rows)
        for origin, origin_rows in relevant_by_origin.items()
    }

    lines: list[str] = [
        "# 小红书大模型面经 Discovery 增量分析报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 数据源：`{SOURCES_PATH.relative_to(ROOT)}`",
        "- 说明：分类和主题是多标签统计，同一篇笔记可以命中多个主题，因此分类/主题频次不是互斥计数。",
        "",
        "## 一句话结论",
        "",
        (
            f"本次 discovery/search 新增了 {len(relevant_by_origin['discover'])} 条严格相关笔记，"
            f"把可用相关来源从收藏夹基线的 {len(relevant_by_origin['favorites'])} 条扩展到 {len(relevant)} 条。"
            "新增部分最明显地放大了公司面经、后训练/RL、Agent/RAG、Transformer、推理性能和 LoRA 等方向。"
        ),
        "",
        "## 规模对比",
        "",
    ]
    lines.extend(
        table(
            ["来源", "抽取详情", "严格相关", "相关率", "相关占总体", "题目命中次数"],
            [
                [
                    origin_label("favorites"),
                    len(by_origin["favorites"]),
                    len(relevant_by_origin["favorites"]),
                    pct(len(relevant_by_origin["favorites"]), len(by_origin["favorites"])),
                    pct(len(relevant_by_origin["favorites"]), len(relevant)),
                    total_questions_by_origin["favorites"],
                ],
                [
                    origin_label("discover"),
                    len(by_origin["discover"]),
                    len(relevant_by_origin["discover"]),
                    pct(len(relevant_by_origin["discover"]), len(by_origin["discover"])),
                    pct(len(relevant_by_origin["discover"]), len(relevant)),
                    total_questions_by_origin["discover"],
                ],
                [
                    "合计",
                    len(rows),
                    len(relevant),
                    pct(len(relevant), len(rows)),
                    "100.0%",
                    sum(total_questions_by_origin.values()),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 抽取状态",
            "",
        ]
    )
    status_rows = []
    for origin in ("favorites", "discover"):
        counter = status_by_origin[origin]
        known = (
            counter.get("text-ok", 0)
            + counter.get("title-only", 0)
            + counter.get("unavailable", 0)
            + counter.get("mismatched-note", 0)
            + counter.get("security-limited", 0)
        )
        status_rows.append(
            [
                origin_label(origin),
                counter.get("text-ok", 0),
                counter.get("title-only", 0),
                counter.get("unavailable", 0),
                counter.get("mismatched-note", 0),
                counter.get("security-limited", 0),
                sum(counter.values()) - known,
                sum(counter.values()),
            ]
        )
    lines.extend(table(["来源", "text-ok", "title-only", "unavailable", "mismatched-note", "security-limited", "other", "合计"], status_rows))

    lines.extend(["", "## 分类增量", ""])
    lines.extend(table(["分类", "收藏夹", "discovery 新增", "合计", "新增量"], category_rows))

    top_topic_rows = topic_rows[:20]
    lines.extend(["", "## 高频主题 Top 20", ""])
    lines.extend(
        table(
            ["主题", "分类", "收藏夹", "discovery 新增", "合计", "新增占比"],
            [
                [
                    item["question"],
                    item["category"],
                    item["favorites"],
                    item["discover"],
                    item["total"],
                    pct(item["discover"], item["total"]),
                ]
                for item in top_topic_rows
            ],
        )
    )

    discovery_gain_rows = sorted(topic_rows, key=lambda item: (-item["discover"], item["question"]))[:15]
    lines.extend(["", "## 本次新增最集中的方向", ""])
    lines.extend(
        table(
            ["主题", "收藏夹", "discovery 新增", "合计"],
            [[item["question"], item["favorites"], item["discover"], item["total"]] for item in discovery_gain_rows],
        )
    )

    new_or_nearly_new = [item for item in topic_rows if item["favorites"] <= 3 and item["discover"] >= 10]
    new_or_nearly_new.sort(key=lambda item: (-item["discover"], item["question"]))
    if new_or_nearly_new:
        lines.extend(["", "## 收藏夹基线较弱、discovery 补强明显的主题", ""])
        lines.extend(
            table(
                ["主题", "分类", "收藏夹", "discovery 新增", "合计"],
                [
                    [item["question"], item["category"], item["favorites"], item["discover"], item["total"]]
                    for item in new_or_nearly_new
                ],
            )
        )

    lines.extend(["", "## 总体画像", ""])
    lines.extend(
        [
            f"- 当前 notebook 覆盖 {summary.get('topics', len(topic_rows))} 个归一化主题。",
            f"- 发现/搜索新增贡献了 {pct(len(relevant_by_origin['discover']), len(relevant))} 的相关来源，是当前题频统计的主要来源。",
            "- 高频问题呈现明显的岗位面经导向：公司面经大类会把多个技术方向聚合到同一篇笔记里，因此频率很高；复习时应把它当作“面试入口”，再下钻到具体技术主题。",
            "- 技术主题的第二梯队主要是后训练/RL、Agent/RAG、Transformer/Attention、推理性能、LoRA/微调和评估，对应大模型算法岗最常见的追问路径。",
            "- 长尾主题仍有价值：如 loss spike、GRPO 训练监控、两层 MLP 手撕等频率低，但面试一旦出现通常要求公式或代码推导。",
            "",
            "## 后续采集建议",
            "",
            "- 后续不要再先攒超大候选池；默认使用 `run` 的小队列策略，让 `candidate_pending` 维持在 20-60 左右。",
            "- 每条详情抽取都会立即写入 raw JSONL；即使中途触发风控，已完成笔记也不会丢。",
            "- 触发 `security-limited` 后应停止详情访问并等待冷却，继续搜索页 inventory 可以保守进行，但不要裸开 `/explore/{note_id}`。",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(REPORT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
