#!/usr/bin/env python3
"""Crawl Xiaohongshu discovery feed and extract LLM interview notes.

Raw discovery URLs can contain short-lived sensitive query params. This script
writes them only under data/raw, which is ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DISCOVER_URL = "https://www.xiaohongshu.com/explore?channel_id=homefeed_recommend"

LLM_RE = re.compile(
    r"(?i)(大模型|大语言|llm|transformer|attention|kv ?cache|gqa|mqa|mla|"
    r"flash ?attention|rope|rmsnorm|layernorm|deepseek|qwen|llama|kimi|claude|gpt|"
    r"agent|rag|mcp|function call|tool call|sft|rlhf|(?<![a-z])(dpo|ppo|grpo|dapo|gspo)(?![a-z])|"
    r"reward|prm|lora|qlora|moe|vllm|sglang|deepspeed|zero|微调|后训练|"
    r"强化学习|推理优化|蒸馏|量化|embedding|rerank|向量数据库|cot)"
)
INTERVIEW_RE = re.compile(r"(面试|面经|八股|手撕|校招|社招|一面|二面|三面|算法岗|面试题|常问|被问)")
STRONG_INTERVIEW_RE = re.compile(r"(面试|面经|八股|手撕|校招|社招|算法岗|面试题|常问|被问)")
TECH_RE = re.compile(
    r"(?i)(transformer|attention|kv ?cache|rope|norm|loss|训练|微调|推理|部署|框架|架构|"
    r"agent|rag|mcp|function call|tool|sft|rlhf|(?<![a-z])(dpo|ppo|grpo)(?![a-z])|reward|prm|lora|moe|"
    r"vllm|sglang|蒸馏|量化|评估|检索|召回|rerank|embedding|上下文|代码|(?<![a-z])code(?![a-z])|算法)"
)
NEGATIVE_RE = re.compile(
    r"(?i)(前端|vue|react|javascript|java面试|spring|redis|kafka|微服务|"
    r"考公|考研|留学|穿搭|减肥|美甲|装修|旅游|vlog|摄影|星穹铁道|崩坏)"
)
FINANCE_QUANT_RE = re.compile(
    r"(?i)(空仓|股票|基金|交易|投研|投资|炒股|行情|撮合|下单|成交|"
    r"量化研究员|量化交易|量化开发|量化私募|公募量化|小资金量化|量化操作|量化比赛|跑量化|做量化|因子|回测|alpha|cta)"
)
MARKET_RE = re.compile(r"(?i)(股票|股价|a股|港股|美股|英伟达|三星电子|海力士|美光|长鑫|摩根士丹利|高盛|基本面)")
AI_PRODUCTIVITY_RE = re.compile(r"(?i)(ppt|figma|答辩|论文排版|操作浏览器|创业手册|创业公司|提示词模板|个人开发者|aicoding)")
GENERIC_AI_TOOL_RE = re.compile(r"(?i)(claude|codex|gpt)")
VISION_GEN_RE = re.compile(r"(?i)(aigc|计算机视觉|视觉|视频生成|图像生成|diffusion|stable diffusion|controlnet|ddpm|ddim)")
STRONG_LLM_RE = re.compile(
    r"(?i)(大模型|大语言|llm|transformer|agent|rag|deepseek|qwen|llama|lora|moe|sft|vla|vlm|openvla|(?<![a-z])(grpo|dpo)(?![a-z])|vllm|flash ?attention)"
)


COLLECT_JS = r"""
(() => {
  const normalize = (s) => (s || "").replace(/\s+/g, " ").trim();
  const pickTitle = (a) => {
    const chunks = [];
    const push = (value) => {
      const text = normalize(value);
      if (text) chunks.push(text);
    };
    push(a.getAttribute("title"));
    push(a.getAttribute("aria-label"));
    push(a.innerText);
    const card = a.closest("section, article, li, div.note-item, div.feeds-page, div") || a;
    push(card.innerText);
    const blacklist = /^(赞|点赞|收藏|评论|分享|作者|小红书|搜索|登录|打开看看|展开|更多|\d+)$/;
    const lines = chunks.flatMap((text) => text.split(/[\n\r]+| {2,}/))
      .map(normalize)
      .filter(Boolean)
      .filter((line) => line.length >= 2 && line.length <= 140)
      .filter((line) => !blacklist.test(line));
    lines.sort((a, b) => {
      const score = (s) => (/[大模面经面试八股手撕]|llm|agent|rag|dpo|grpo|lora|moe|vllm/i.test(s) ? 1000 : 0) + s.length;
      return score(b) - score(a);
    });
    return lines[0] || normalize(a.innerText).slice(0, 120);
  };
  const pickAuthor = (a) => {
    const card = a.closest("section, article, li, div.note-item, div") || a;
    const node = card.querySelector("[class*='author'], [class*='user'], [class*='name']");
    return normalize(node && node.innerText).slice(0, 80);
  };
  const seen = new Map();
  for (const a of Array.from(document.querySelectorAll("a[href*='/explore/'], a[href*='/search_result/']"))) {
    const href = a.href || a.getAttribute("href") || "";
    const m = href.match(/(?:\/explore\/|\/search_result\/)([^/?#]+)/);
    if (!m) continue;
    const noteId = m[1];
    if (!noteId || noteId.length < 12) continue;
    const title = pickTitle(a);
    const item = { note_id: noteId, raw_url: href, title, author: pickAuthor(a) };
    const old = seen.get(noteId);
    const betterTitle = !old || (title || "").length > (old.title || "").length;
    const betterRoute = old && item.raw_url.includes("?") && !(old.raw_url || "").includes("?");
    if (!old || betterTitle || betterRoute) {
      seen.set(noteId, {
        note_id: noteId,
        raw_url: betterRoute && !betterTitle ? item.raw_url : item.raw_url,
        title: betterTitle ? title : old.title,
        author: item.author || (old && old.author) || ""
      });
    }
  }
  return JSON.stringify({
    url: location.href,
    page_title: document.title,
    ready_state: document.readyState,
    scroll_y: Math.round(window.scrollY),
    viewport_height: Math.round(window.innerHeight),
    body_height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
    item_count: seen.size,
    items: Array.from(seen.values())
  });
})()
"""


SCROLL_JS = r"""
(() => {
  window.scrollBy(0, Math.max(520, Math.floor(window.innerHeight * 0.72)));
  return JSON.stringify({
    scroll_y: Math.round(window.scrollY),
    body_height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)
  });
})()
"""


BACK_JS = r"""
(() => {
  history.back();
  return JSON.stringify({url: location.href});
})()
"""


TOP_JS = r"""
(() => {
  window.scrollTo(0, 0);
  return JSON.stringify({scroll_y: Math.round(window.scrollY)});
})()
"""


EXTRACT_JS = r"""
(() => {
  const normalize = (s) => (s || "").replace(/\s+/g, " ").trim();
  const pick = (selectors) => {
    for (const sel of selectors) {
      const node = document.querySelector(sel);
      const text = normalize(node && node.innerText);
      if (text) return text;
    }
    return "";
  };
  const title = pick(["#detail-title", "[class*='note-content'] [class*='title']", "[class*='title']"]) ||
    normalize((document.title || "").replace(/ - 小红书$/, ""));
  const descNode = document.querySelector("#detail-desc") ||
    document.querySelector("[class*='note-content'] [class*='desc']") ||
    document.querySelector("[class*='desc']");
  let desc = normalize(descNode && descNode.innerText);
  if (!desc) {
    const container = document.querySelector("[class*='note-content']") ||
      document.querySelector("[class*='note-container']") ||
      document.querySelector("main") ||
      document.body;
    desc = normalize(container && container.innerText);
  }
  const author = pick([".author-wrapper .username", "[class*='author'] [class*='name']", "[class*='username']", "[class*='author']"]);
  return JSON.stringify({
    url: location.href,
    page_title: document.title,
    title,
    desc,
    author,
    image_count: document.querySelectorAll("img").length
  });
})()
"""


def today() -> str:
    return os.environ.get("XHS_DATE", datetime.now().strftime("%Y-%m-%d"))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_dump_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def run_osascript(lines: list[str], args: list[str] | None = None, timeout: int = 60) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".applescript", delete=False) as f:
        f.write("\n".join(lines))
        script_path = f.name
    try:
        cmd: list[str] = ["/usr/bin/osascript", script_path]
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "osascript failed")
        return result.stdout.strip()
    finally:
        try:
            Path(script_path).unlink()
        except FileNotFoundError:
            pass


def chrome_open(url: str) -> None:
    run_osascript(
        [
            "on run argv",
            "set discoverUrl to item 1 of argv",
            'tell application "Google Chrome"',
            "if not (exists window 1) then make new window",
            "set URL of active tab of front window to discoverUrl",
            "activate",
            "end tell",
            "end run",
        ],
        [url],
        timeout=20,
    )


def chrome_open_new_tab(url: str) -> None:
    run_osascript(
        [
            "on run argv",
            "set discoverUrl to item 1 of argv",
            'tell application "Google Chrome"',
            "if not (exists window 1) then make new window",
            "make new tab at end of tabs of front window with properties {URL:discoverUrl}",
            "set active tab index of front window to (count of tabs of front window)",
            "activate",
            "end tell",
            "end run",
        ],
        [url],
        timeout=20,
    )


def chrome_close_active_tab() -> None:
    run_osascript(
        [
            'tell application "Google Chrome"',
            "if (count of tabs of front window) > 1 then close active tab of front window",
            "end tell",
        ],
        timeout=20,
    )


def chrome_js(js: str, timeout: int = 60) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".js", delete=False) as f:
        f.write(js)
        js_path = f.name
    try:
        return run_osascript(
            [
                "on run argv",
                "set jsPath to item 1 of argv",
                "set jsSource to read POSIX file jsPath",
                'tell application "Google Chrome"',
                "if not (exists window 1) then make new window",
                "set jsResult to execute active tab of front window javascript (jsSource as text)",
                "end tell",
                "return jsResult",
                "end run",
            ],
            [js_path],
            timeout=timeout,
        )
    finally:
        try:
            Path(js_path).unlink()
        except FileNotFoundError:
            pass


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def public_url(note_id: str) -> str:
    return f"https://www.xiaohongshu.com/explore/{note_id}"


def score_title(title: str) -> tuple[int, bool]:
    text = clean(title)
    score = 0
    if LLM_RE.search(text):
        score += 3
    if STRONG_LLM_RE.search(text):
        score += 2
    if INTERVIEW_RE.search(text):
        score += 2
    if (NEGATIVE_RE.search(text) or FINANCE_QUANT_RE.search(text) or MARKET_RE.search(text)) and not STRONG_LLM_RE.search(text):
        score -= 4
    return score, score >= 3 and bool(INTERVIEW_RE.search(text) or TECH_RE.search(text))


def is_detail_related(row: dict[str, Any]) -> bool:
    if row.get("extract_status") == "text-ok":
        text = clean(" ".join(str(row.get(k, "")) for k in ("title", "desc")))
    else:
        text = clean(" ".join(str(row.get(k, "")) for k in ("source_title", "title", "desc")))
    if not LLM_RE.search(text):
        return False
    if not (INTERVIEW_RE.search(text) or TECH_RE.search(text)):
        return False
    if FINANCE_QUANT_RE.search(text) and not STRONG_LLM_RE.search(text):
        return False
    if MARKET_RE.search(text) and not INTERVIEW_RE.search(text):
        return False
    if GENERIC_AI_TOOL_RE.search(text) and AI_PRODUCTIVITY_RE.search(text) and not (STRONG_LLM_RE.search(text) or STRONG_INTERVIEW_RE.search(text)):
        return False
    if VISION_GEN_RE.search(text) and not STRONG_LLM_RE.search(text):
        return False
    if NEGATIVE_RE.search(text) and not STRONG_LLM_RE.search(text):
        return False
    return True


def existing_favorite_ids() -> set[str]:
    ids: set[str] = set()
    raw = RAW_DIR / "xhs_note_extracts_2026-05-24.jsonl"
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                note_id = json.loads(line).get("note_id")
            except json.JSONDecodeError:
                continue
            if note_id:
                ids.add(note_id)
    sources = ROOT / "data" / "processed" / "sources.csv"
    if sources.exists():
        with sources.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("source_origin", "favorites") == "favorites" and row.get("note_id"):
                    ids.add(row["note_id"])
    return ids


def inventory_path(date: str) -> Path:
    return RAW_DIR / f"xhs_discover_inventory_{date}.json"


def steps_path(date: str) -> Path:
    return RAW_DIR / f"xhs_discover_steps_{date}.jsonl"


def extracts_path(date: str) -> Path:
    return RAW_DIR / f"xhs_discover_note_extracts_{date}.jsonl"


def load_inventory(date: str) -> dict[str, Any]:
    path = inventory_path(date)
    if not path.exists():
        return {"items": [], "meta": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"items": payload, "meta": {}}
    return payload


def inventory_items_by_id(date: str) -> dict[str, dict[str, Any]]:
    payload = load_inventory(date)
    return {item.get("note_id"): item for item in payload.get("items", []) if item.get("note_id")}


def write_inventory(date: str, items: dict[str, dict[str, Any]], step: int, errors: int = 0) -> None:
    favorite_ids = existing_favorite_ids()
    ordered = sorted(items.values(), key=lambda x: int(x.get("discover_order") or 10**12))
    related = [x for x in ordered if x.get("related_by_title")]
    related_new = [x for x in related if x.get("note_id") not in favorite_ids]
    route_ready_new = [x for x in related_new if "?" in (x.get("raw_url") or "")]
    payload = {
        "meta": {
            "generated_at": now_iso(),
            "source": "xiaohongshu_discovery",
            "steps": step,
            "errors": errors,
            "unique_notes": len(ordered),
            "related_by_title": len(related),
            "related_new_by_title": len(related_new),
            "route_ready_new_by_title": len(route_ready_new),
        },
        "items": ordered,
    }
    json_dump_atomic(inventory_path(date), payload)


def collect_once() -> dict[str, Any]:
    return json.loads(chrome_js(COLLECT_JS, timeout=45))


def scroll_once() -> None:
    chrome_js(SCROLL_JS, timeout=20)


def top_once() -> None:
    chrome_js(TOP_JS, timeout=20)


def click_visible_note(note_id: str) -> bool:
    note_json = json.dumps(note_id)
    js = f"""
(() => {{
  const noteId = {note_json};
  const links = Array.from(document.querySelectorAll("a[href*='/explore/'], a[href*='/search_result/']"));
  const a = links.find((node) => (node.href || node.getAttribute("href") || "").includes(`/explore/${{noteId}}`));
  if (!a) return JSON.stringify({{clicked: false, reason: "missing"}});
  a.scrollIntoView({{block: "center", inline: "center"}});
  a.click();
  return JSON.stringify({{clicked: true, href: a.href || a.getAttribute("href") || ""}});
}})()
"""
    payload = json.loads(chrome_js(js, timeout=20))
    return bool(payload.get("clicked"))


def run_inventory(args: argparse.Namespace, deadline: float | None = None) -> dict[str, Any]:
    date = args.date
    items = inventory_items_by_id(date)
    next_order = max([int(x.get("discover_order") or 0) for x in items.values()] + [0]) + 1
    step = int(load_inventory(date).get("meta", {}).get("steps", 0))
    errors = int(load_inventory(date).get("meta", {}).get("errors", 0))
    plateau = 0

    chrome_open(args.entry_url)
    time.sleep(args.initial_wait)
    try:
        top_once()
    except Exception:
        pass
    time.sleep(1.0)

    started = time.time()
    while True:
        if deadline and time.time() >= deadline:
            break
        if args.max_seconds and time.time() - started >= args.max_seconds:
            break
        if args.max_steps and step >= args.max_steps:
            break
        meta = load_inventory(date).get("meta", {})
        if args.target_related and int(meta.get("route_ready_new_by_title", 0)) >= args.target_related:
            break

        step += 1
        snapshot: dict[str, Any]
        try:
            snapshot = collect_once()
            snapshot["step"] = step
            snapshot["collected_at"] = now_iso()
            append_jsonl(steps_path(date), snapshot)
            new_count = 0
            for raw in snapshot.get("items", []):
                note_id = raw.get("note_id")
                if not note_id:
                    continue
                title = clean(raw.get("title", ""))
                score, related = score_title(title)
                old = items.get(note_id)
                if old is None:
                    items[note_id] = {
                        "note_id": note_id,
                        "title": title,
                        "author": clean(raw.get("author", "")),
                        "url": public_url(note_id),
                        "raw_url": raw.get("raw_url", ""),
                        "discover_order": next_order,
                        "first_seen_step": step,
                        "last_seen_step": step,
                        "title_score": score,
                        "related_by_title": related,
                    }
                    next_order += 1
                    new_count += 1
                else:
                    old["last_seen_step"] = step
                    if title and len(title) > len(old.get("title", "")):
                        old["title"] = title
                        old["title_score"], old["related_by_title"] = score_title(title)
                    if raw.get("raw_url") and ("?" in raw["raw_url"] or "?" not in old.get("raw_url", "")):
                        old["raw_url"] = raw["raw_url"]
                    if raw.get("author") and not old.get("author"):
                        old["author"] = clean(raw.get("author", ""))
            plateau = 0 if new_count else plateau + 1
            write_inventory(date, items, step, errors)
            meta = load_inventory(date).get("meta", {})
            print(
                json.dumps(
                    {
                        "mode": "inventory",
                        "step": step,
                        "new": new_count,
                        "unique": meta.get("unique_notes"),
                        "related_new_by_title": meta.get("related_new_by_title"),
                        "scroll_y": snapshot.get("scroll_y"),
                        "body_height": snapshot.get("body_height"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if plateau >= args.plateau_reload_steps:
                chrome_open(args.entry_url)
                time.sleep(args.reload_wait)
                plateau = 0
            else:
                scroll_once()
                time.sleep(args.sleep)
        except Exception as exc:
            errors += 1
            print(json.dumps({"mode": "inventory", "step": step, "error": str(exc)}, ensure_ascii=False), flush=True)
            write_inventory(date, items, step, errors)
            time.sleep(min(args.sleep * 3, 8))
            if errors % 5 == 0:
                chrome_open(args.entry_url)
                time.sleep(args.reload_wait)

    return status_payload(date, args.target_final or args.target_related or 0)


def upsert_snapshot_items(
    snapshot: dict[str, Any],
    items: dict[str, dict[str, Any]],
    step: int,
    next_order: int,
) -> tuple[int, int, list[dict[str, Any]]]:
    new_count = 0
    fresh_related: list[dict[str, Any]] = []
    for raw in snapshot.get("items", []):
        note_id = raw.get("note_id")
        if not note_id:
            continue
        title = clean(raw.get("title", ""))
        score, related = score_title(title)
        old = items.get(note_id)
        if old is None:
            item = {
                "note_id": note_id,
                "title": title,
                "author": clean(raw.get("author", "")),
                "url": public_url(note_id),
                "raw_url": raw.get("raw_url", ""),
                "discover_order": next_order,
                "first_seen_step": step,
                "last_seen_step": step,
                "title_score": score,
                "related_by_title": related,
            }
            items[note_id] = item
            next_order += 1
            new_count += 1
        else:
            item = old
            old["last_seen_step"] = step
            if title and len(title) > len(old.get("title", "")):
                old["title"] = title
                old["title_score"], old["related_by_title"] = score_title(title)
            if raw.get("raw_url") and ("?" in raw["raw_url"] or "?" not in old.get("raw_url", "")):
                old["raw_url"] = raw["raw_url"]
            if raw.get("author") and not old.get("author"):
                old["author"] = clean(raw.get("author", ""))
        if item.get("related_by_title"):
            fresh_related.append(item)
    return new_count, next_order, fresh_related


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    date = args.date
    deadline = time.time() + args.max_hours * 3600
    favorite_ids = existing_favorite_ids()
    extracted = load_discover_extracts(date)
    completed_ids = {note_id for note_id, row in extracted.items() if is_completed_extract(row)}
    items = inventory_items_by_id(date)
    next_order = max([int(x.get("discover_order") or 0) for x in items.values()] + [0]) + 1
    step = int(load_inventory(date).get("meta", {}).get("steps", 0))
    errors = int(load_inventory(date).get("meta", {}).get("errors", 0))

    if args.security_cooldown > 0:
        latest_limited, limited_age = latest_security_limited(date)
        if latest_limited and limited_age is not None and limited_age < args.security_cooldown:
            payload = status_payload(date, args.target_final or 0)
            payload.update(
                {
                    "mode": "live",
                    "blocked": "cooldown-active",
                    "last_security_note_id": latest_limited.get("note_id"),
                    "cooldown_seconds": args.security_cooldown,
                    "cooldown_remaining_seconds": round(args.security_cooldown - limited_age, 1),
                }
            )
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            return payload

    chrome_open(args.entry_url)
    time.sleep(args.initial_wait)
    try:
        top_once()
    except Exception:
        pass
    time.sleep(1.0)

    while time.time() < deadline:
        current = status_payload(date, args.target_final)
        if args.target_final and current["discover_relevant"] >= args.target_final:
            break
        if args.security_cooldown > 0:
            latest_limited, limited_age = latest_security_limited(date)
            if latest_limited and limited_age is not None and limited_age < args.security_cooldown:
                print(
                    json.dumps(
                        {
                            "mode": "live",
                            "blocked": "cooldown-active",
                            "last_security_note_id": latest_limited.get("note_id"),
                            "cooldown_seconds": args.security_cooldown,
                            "cooldown_remaining_seconds": round(args.security_cooldown - limited_age, 1),
                            "confirmed": current["discover_relevant"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break
        if args.max_steps and step >= args.max_steps:
            break
        step += 1
        try:
            snapshot = collect_once()
            snapshot["step"] = step
            snapshot["collected_at"] = now_iso()
            append_jsonl(steps_path(date), snapshot)
            new_count, next_order, fresh_related = upsert_snapshot_items(snapshot, items, step, next_order)
            write_inventory(date, items, step, errors)
            fresh_candidates = [
                item for item in fresh_related
                if item.get("note_id") not in favorite_ids
                and item.get("note_id") not in completed_ids
                and (args.allow_naked_candidates or "?" in (item.get("raw_url") or ""))
            ]
            fresh_candidates.sort(key=lambda x: int(x.get("discover_order") or 10**12))
            extracted_this_step = 0
            security_hits = 0
            for item in fresh_candidates[: args.live_extract_per_step]:
                use_click = args.allow_naked_candidates and "?" not in (item.get("raw_url") or "")
                row = extract_one(item, args, new_tab=not use_click, click_visible=use_click)
                append_jsonl(extracts_path(date), row)
                extracted[item["note_id"]] = row
                if is_completed_extract(row):
                    completed_ids.add(item["note_id"])
                extracted_this_step += 1
                if row.get("extract_status") == "security-limited":
                    security_hits += 1
                print(
                    json.dumps(
                        {
                            "mode": "live-extract",
                            "step": step,
                            "note_id": item["note_id"],
                            "related": row.get("is_llm_related"),
                            "status": row.get("extract_status"),
                            "title": row.get("title") or row.get("source_title", ""),
                            "confirmed": status_payload(date, args.target_final)["discover_relevant"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                time.sleep(args.extract_sleep)
            status = status_payload(date, args.target_final)
            print(
                json.dumps(
                    {
                        "mode": "live",
                        "step": step,
                        "new": new_count,
                        "fresh_candidates": len(fresh_candidates),
                        "extracted": extracted_this_step,
                        "unique": status["unique_notes"],
                        "route_ready": status["route_ready_new_by_title"],
                        "confirmed": status["discover_relevant"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if security_hits >= args.security_break_hits:
                break
            scroll_once()
            time.sleep(args.sleep)
        except Exception as exc:
            errors += 1
            print(json.dumps({"mode": "live", "step": step, "error": str(exc)}, ensure_ascii=False), flush=True)
            write_inventory(date, items, step, errors)
            time.sleep(min(args.sleep * 3, 8))
            if errors % 5 == 0:
                chrome_open(args.entry_url)
                time.sleep(args.reload_wait)
    return status_payload(date, args.target_final)


FINAL_EXTRACT_STATUSES = {"text-ok", "title-only", "unavailable", "mismatched-note"}


def is_completed_extract(row: dict[str, Any]) -> bool:
    return row.get("extract_status") in FINAL_EXTRACT_STATUSES


def extract_rank(row: dict[str, Any]) -> tuple[int, int]:
    status = row.get("extract_status")
    priority = {
        "text-ok": 5,
        "title-only": 4,
        "unavailable": 3,
        "mismatched-note": 2,
        "security-limited": 1,
    }.get(status, 0)
    return priority, len(row.get("desc") or "")


def load_discover_extracts(date: str) -> dict[str, dict[str, Any]]:
    path = extracts_path(date)
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        note_id = row.get("note_id")
        if not note_id:
            continue
        old = rows.get(note_id)
        if old is None or extract_rank(row) > extract_rank(old):
            rows[note_id] = row
    return rows


def latest_security_limited(date: str) -> tuple[dict[str, Any] | None, float | None]:
    path = extracts_path(date)
    if not path.exists():
        return None, None
    latest: dict[str, Any] | None = None
    latest_dt: datetime | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("extract_status") != "security-limited":
            continue
        try:
            dt = datetime.fromisoformat(row.get("extracted_at") or "")
        except ValueError:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest = row
    if latest is None or latest_dt is None:
        return None, None
    age = (datetime.now().astimezone() - latest_dt).total_seconds()
    return latest, max(age, 0.0)


def extract_one(
    item: dict[str, Any],
    args: argparse.Namespace,
    *,
    new_tab: bool = False,
    click_visible: bool = False,
) -> dict[str, Any]:
    raw_url = item.get("raw_url") or item.get("url") or public_url(item["note_id"])
    clicked_from_listing = False
    if click_visible:
        clicked_from_listing = click_visible_note(item["note_id"])
        if not clicked_from_listing:
            return {
                "note_id": item["note_id"],
                "source_origin": "discover",
                "source_title": item.get("title", ""),
                "source_author": item.get("author", ""),
                "source_url": raw_url,
                "url": raw_url,
                "title": item.get("title", ""),
                "desc": "",
                "author": item.get("author", ""),
                "image_count": "",
                "extract_status": "click-missing",
                "extracted_at": now_iso(),
                "is_llm_related": False,
            }
    elif new_tab:
        chrome_open_new_tab(raw_url)
    else:
        chrome_open(raw_url)
    time.sleep(args.detail_wait)
    try:
        detail = json.loads(chrome_js(EXTRACT_JS, timeout=45))
        desc = clean(detail.get("desc", ""))
        title = clean(detail.get("title", "")) or item.get("title", "")
        unavailable = (
            "当前笔记暂时无法浏览" in title
            or "当前笔记暂时无法浏览" in (detail.get("page_title") or "")
            or "/404" in (detail.get("url") or "")
            or "你访问的页面不见了" in (detail.get("page_title") or "")
        )
        security_limited = "安全限制" in title or "安全限制" in (detail.get("page_title") or "") or "安全限制" in desc
        footer_only = "沪ICP备" in desc and "违法不良信息举报" in desc and len(desc) < 1200
        if security_limited:
            desc = ""
            status = "security-limited"
        elif unavailable or footer_only:
            desc = ""
            status = "unavailable"
        elif item["note_id"] not in (detail.get("url") or ""):
            desc = ""
            status = "mismatched-note"
        else:
            status = "text-ok" if len(desc) >= 40 else "title-only"
        row = {
            "note_id": item["note_id"],
            "source_origin": "discover",
            "source_title": item.get("title", ""),
            "source_author": item.get("author", ""),
            "source_url": raw_url,
            "url": detail.get("url") or raw_url,
            "title": title,
            "desc": desc,
            "author": clean(detail.get("author", "")) or item.get("author", ""),
            "image_count": detail.get("image_count", ""),
            "extract_status": status,
            "extracted_at": now_iso(),
        }
    except Exception as exc:
        row = {
            "note_id": item["note_id"],
            "source_origin": "discover",
            "source_title": item.get("title", ""),
            "source_author": item.get("author", ""),
            "source_url": raw_url,
            "url": raw_url,
            "title": item.get("title", ""),
            "desc": "",
            "author": item.get("author", ""),
            "image_count": "",
            "extract_status": f"error:{str(exc)[:80]}",
            "extracted_at": now_iso(),
        }
    finally:
        if clicked_from_listing:
            try:
                chrome_js(BACK_JS, timeout=20)
                time.sleep(1.0)
            except Exception:
                chrome_open(args.entry_url)
                time.sleep(args.reload_wait)
        elif new_tab:
            try:
                chrome_close_active_tab()
            except Exception:
                pass
    row["is_llm_related"] = row.get("extract_status") == "text-ok" and is_detail_related(row)
    return row


def run_extract(args: argparse.Namespace, deadline: float | None = None) -> dict[str, Any]:
    date = args.date
    if args.security_cooldown > 0:
        latest_limited, limited_age = latest_security_limited(date)
        if latest_limited and limited_age is not None and limited_age < args.security_cooldown:
            payload = status_payload(date, args.target_final or 0)
            payload.update(
                {
                    "mode": "extract",
                    "blocked": "cooldown-active",
                    "last_security_note_id": latest_limited.get("note_id"),
                    "cooldown_seconds": args.security_cooldown,
                    "cooldown_remaining_seconds": round(args.security_cooldown - limited_age, 1),
                }
            )
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            return payload
    favorite_ids = existing_favorite_ids()
    extracted = load_discover_extracts(date)
    completed_ids = {note_id for note_id, row in extracted.items() if is_completed_extract(row)}
    items = list(inventory_items_by_id(date).values())
    candidates = [
        item
        for item in items
        if item.get("related_by_title")
        and (args.allow_naked_candidates or "?" in (item.get("raw_url") or ""))
        and item.get("note_id") not in favorite_ids
        and item.get("note_id") not in completed_ids
    ]
    candidates.sort(
        key=lambda x: (
            1 if extracted.get(x.get("note_id"), {}).get("extract_status") == "security-limited" else 0,
            int(x.get("discover_order") or 10**12),
        )
    )
    limit = args.limit or len(candidates)
    processed = 0
    security_hits = 0

    for item in candidates[:limit]:
        if deadline and time.time() >= deadline:
            break
        current = status_payload(date, args.target_final or 0)
        if args.target_final and current["discover_relevant"] >= args.target_final:
            break
        row = extract_one(item, args)
        if row.get("extract_status") == "security-limited":
            append_jsonl(extracts_path(date), row)
            extracted[item["note_id"]] = row
            processed += 1
            security_hits += 1
            print(
                json.dumps(
                    {
                        "mode": "extract",
                        "blocked": "security-limited",
                        "note_id": item["note_id"],
                        "processed": processed,
                        "cooldown_seconds": args.security_cooldown,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if security_hits >= args.security_break_hits:
                break
            time.sleep(args.extract_sleep)
            continue
        append_jsonl(extracts_path(date), row)
        extracted[item["note_id"]] = row
        processed += 1
        print(
            json.dumps(
                {
                    "mode": "extract",
                    "processed": processed,
                    "note_id": item["note_id"],
                    "related": row.get("is_llm_related"),
                    "status": row.get("extract_status"),
                    "title": row.get("title") or row.get("source_title", ""),
                    "confirmed": status_payload(date, args.target_final or 0)["discover_relevant"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(args.extract_sleep)
    return status_payload(date, args.target_final or 0)


def status_payload(date: str, target: int = 0) -> dict[str, Any]:
    inv = load_inventory(date)
    meta = inv.get("meta", {})
    favorite_ids = existing_favorite_ids()
    items = inv.get("items", [])
    related_new = [
        item for item in items if item.get("related_by_title") and item.get("note_id") not in favorite_ids
    ]
    route_ready = [item for item in related_new if "?" in (item.get("raw_url") or "")]
    extracted = load_discover_extracts(date)
    completed_ids = {note_id for note_id, row in extracted.items() if is_completed_extract(row)}
    text_related = [row for row in extracted.values() if is_detail_related(row)]
    confirmed = [
        row for row in extracted.values()
        if row.get("extract_status") == "text-ok" and is_detail_related(row)
    ]
    pending = [item for item in route_ready if item.get("note_id") not in completed_ids]
    pending_all = [item for item in related_new if item.get("note_id") not in completed_ids]
    return {
        "date": date,
        "inventory_path": str(inventory_path(date).relative_to(ROOT)),
        "extracts_path": str(extracts_path(date).relative_to(ROOT)),
        "inventory_steps": meta.get("steps", 0),
        "inventory_errors": meta.get("errors", 0),
        "unique_notes": meta.get("unique_notes", len(items)),
        "related_new_by_title": len(related_new),
        "route_ready_new_by_title": len(route_ready),
        "discover_extracted": len(extracted),
        "discover_relevant": len(confirmed),
        "discover_text_related": len(text_related),
        "candidate_pending": len(pending),
        "candidate_pending_all": len(pending_all),
        "target": target,
        "remaining": max(target - len(confirmed), 0) if target else None,
    }


def run_until_target(args: argparse.Namespace) -> dict[str, Any]:
    deadline = time.time() + args.max_hours * 3600
    checkpoint_at = status_payload(args.date, args.target_final).get("discover_extracted", 0)
    while time.time() < deadline:
        status = status_payload(args.date, args.target_final)
        pending_key = "candidate_pending_all" if args.allow_naked_candidates else "candidate_pending"
        pending_count = int(status.get(pending_key) or 0)
        print(json.dumps({"mode": "status", **status}, ensure_ascii=False), flush=True)
        if status["discover_relevant"] >= args.target_final:
            return status
        if pending_count < args.queue_low_watermark:
            inv_args = argparse.Namespace(**vars(args))
            inv_args.max_steps = int(status.get("inventory_steps") or 0) + args.inventory_chunk_steps
            inv_args.max_seconds = 0
            queue_gap = max(args.queue_high_watermark - pending_count, args.extract_chunk)
            if args.allow_naked_candidates:
                inv_args.target_related = status["related_new_by_title"] + queue_gap
            else:
                inv_args.target_related = status["route_ready_new_by_title"] + queue_gap
            run_inventory(inv_args, deadline=deadline)
        ext_args = argparse.Namespace(**vars(args))
        latest = status_payload(args.date, args.target_final)
        latest_pending = latest["candidate_pending_all"] if args.allow_naked_candidates else latest["candidate_pending"]
        ext_args.limit = min(args.extract_chunk, latest_pending)
        if ext_args.limit <= 0:
            time.sleep(5)
            continue
        after_extract = run_extract(ext_args, deadline=deadline)
        if args.checkpoint_generate_every > 0:
            extracted_now = int(after_extract.get("discover_extracted") or 0)
            if extracted_now - int(checkpoint_at or 0) >= args.checkpoint_generate_every:
                checkpoint_at = extracted_now
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "generate_notebook.py")],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                print(
                    json.dumps(
                        {
                            "mode": "checkpoint-generate",
                            "returncode": result.returncode,
                            "discover_extracted": extracted_now,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return status_payload(args.date, args.target_final)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inventory", "extract", "status", "run", "live"])
    parser.add_argument("--date", default=today())
    parser.add_argument("--target-related", type=int, default=0, help="Stop inventory after this many new title-related candidates.")
    parser.add_argument("--target-final", type=int, default=0, help="Stop extraction/run after this many confirmed discovery-related notes.")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--inventory-chunk-steps", type=int, default=45)
    parser.add_argument("--extract-chunk", type=int, default=20)
    parser.add_argument("--queue-low-watermark", type=int, default=20, help="Run inventory only when pending route-ready candidates drop below this count.")
    parser.add_argument("--queue-high-watermark", type=int, default=60, help="When topping up inventory, try to refill the pending candidate queue to about this count.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.25)
    parser.add_argument("--extract-sleep", type=float, default=8.0)
    parser.add_argument("--initial-wait", type=float, default=4.0)
    parser.add_argument("--reload-wait", type=float, default=6.0)
    parser.add_argument("--detail-wait", type=float, default=5.0)
    parser.add_argument("--plateau-reload-steps", type=int, default=32)
    parser.add_argument("--security-cooldown", type=float, default=3600.0, help="Skip detail extraction while a recent security-limited row is within this many seconds.")
    parser.add_argument("--security-break-hits", type=int, default=1)
    parser.add_argument("--checkpoint-generate-every", type=int, default=100, help="Regenerate processed notebook outputs after this many newly extracted discovery rows in run mode. Use 0 to disable.")
    parser.add_argument("--live-extract-per-step", type=int, default=2)
    parser.add_argument("--entry-url", default=DISCOVER_URL)
    parser.add_argument("--allow-naked-candidates", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "inventory":
        payload = run_inventory(args)
    elif args.mode == "extract":
        payload = run_extract(args)
    elif args.mode == "live":
        if not args.target_final:
            raise SystemExit("--target-final is required for live mode")
        payload = run_live(args)
    elif args.mode == "run":
        if not args.target_final:
            raise SystemExit("--target-final is required for run mode")
        payload = run_until_target(args)
    else:
        payload = status_payload(args.date, args.target_final or args.target_related)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.mode == "run" and args.target_final and payload["discover_relevant"] < args.target_final:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
