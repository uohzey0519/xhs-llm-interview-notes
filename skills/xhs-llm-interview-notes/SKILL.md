---
name: xhs-llm-interview-notes
description: Update a git-backed large-model interview question notebook from the user's Xiaohongshu favorites, discovery feed, or search pages. Use when the user asks to process 小红书收藏夹, 小红书发现页, 大模型面经, 八股整理, LLM interview notes, or incremental updates from new favorites/search results.
---

# XHS LLM Interview Notes

Use this skill to update the current repository from a logged-in Xiaohongshu favorites page, discovery feed, or search result pages.

## Workflow

1. Open Chrome and verify the Xiaohongshu page is logged in. For favorites collection, set either:
   - `XHS_FAVORITES_URL='https://www.xiaohongshu.com/user/profile/<profile-id>?tab=fav&subTab=note'`
   - `XHS_PROFILE_ID='<profile-id>'`
2. Prefer DOM extraction if Chrome has `View -> Developer -> Allow JavaScript from Apple Events` enabled.
3. If DOM extraction is blocked, use Computer Use accessibility output as the fallback:
   - Scroll the favorites page.
   - Collect note title, author, like count, URL, and visible order.
   - Open relevant notes one by one.
   - Read title and text from the accessibility tree.
4. For image-heavy notes:
   - Capture the note image region or page screenshot.
   - Prefer visual reading from the screenshot in the current environment.
   - Treat `scripts/ocr_image.swift` as experimental unless it has been re-verified.
   - Mark low-confidence or partial OCR as `待复核`.
5. Filter for large-model interview material:
   - 大模型, LLM, Agent, RAG, SFT, RLHF, DPO, PPO, GRPO, MoE, Attention, KV Cache, Transformer, DeepSeek, Claude Code, 面试, 一面, 二面, 算法岗.
6. Normalize each extracted item into atomic questions.
7. Deduplicate semantically equivalent questions and count source frequency.
8. Write outputs:
   - `notes/llm-interview-master.md`
   - `data/processed/detailed_answers.md`
   - `data/processed/question_bank.md`
   - `data/processed/sources.csv`
9. Keep source traceability through source IDs and URLs.
10. Commit updates to git after verification.

## Scripted Update Path

From the repository root:

1. Configure favorites locally if needed:
   `cp .env.example .env`
2. Inventory favorites:
   `zsh scripts/xhs_inventory_scroll.zsh`
3. Build the source index:
   `python3 scripts/xhs_build_sources.py data/raw/xhs_inventory_steps_2026-05-24.jsonl`
4. Extract all inventoried notes:
   `zsh scripts/xhs_extract_notes.zsh all`
   For a quick incremental pass, use a numeric limit or the default related mode, e.g. `zsh scripts/xhs_extract_notes.zsh 50`.
5. Enrich detailed answers and generate the notebook:
   `python3 scripts/enrich_detailed_answers.py`
   `python3 scripts/generate_notebook.py`
6. Verify outputs:
   `python3 scripts/verify_outputs.py`

## Discovery/Search Update Path

Use this path for 小红书发现页 or search-based expansion.

1. Check progress:
   `python3 scripts/xhs_discover_run.py status --date 2026-05-24 --target-final 1000`
2. Prefer the streaming small-queue runner for long jobs. It alternates between a small inventory queue and conservative detail extraction, checkpointing raw rows after every note and regenerating processed outputs every 100 newly extracted discovery rows:
   `python3 scripts/xhs_discover_run.py run --date 2026-05-24 --target-final 1000 --max-hours 8`
   Stable defaults are: route-ready queue low watermark 20, queue refill target 60, extract chunk 20, detail wait 5s, extract sleep 8s, security cooldown 3600s, and stop after 1 security-limited hit.
3. Use `live` mode only when you want to analyze newly discovered cards immediately from the current page rather than draining the existing queue:
   `python3 scripts/xhs_discover_run.py live --date 2026-05-24 --target-final 1000 --max-hours 8 --entry-url '<search-url>'`
4. Inventory a search page only when you explicitly need to expand the candidate pool. Good search URLs include:
   - `https://www.xiaohongshu.com/search_result?keyword=RAG%20%E9%9D%A2%E8%AF%95&source=web_explore_feed`
   - `https://www.xiaohongshu.com/search_result?keyword=Transformer%20%E9%9D%A2%E8%AF%95&source=web_explore_feed`
   - `https://www.xiaohongshu.com/search_result?keyword=LLM%20%E9%9D%A2%E8%AF%95%E9%A2%98&source=web_explore_feed`
   - `https://www.xiaohongshu.com/search_result?keyword=%E5%A4%A7%E6%A8%A1%E5%9E%8B%20%E9%9D%A2%E8%AF%95&source=web_explore_feed`
   - `https://www.xiaohongshu.com/search_result?keyword=Agent%20%E9%9D%A2%E7%BB%8F&source=web_explore_feed`
5. Run inventory with a bounded window, for example:
   `python3 scripts/xhs_discover_run.py inventory --date 2026-05-24 --target-related 1200 --max-seconds 180 --sleep 1.2 --entry-url '<search-url>'`
6. Extract route-ready notes in conservative batches only when you intentionally want to drain an existing queue without more inventory:
   `python3 scripts/xhs_discover_run.py extract --date 2026-05-24 --target-final 1000 --limit 60 --detail-wait 5 --extract-sleep 8 --security-cooldown 3600 --security-break-hits 1`
7. If extraction prints `security-limited` or `cooldown-active`, stop detail access and wait for a later run. Do not keep opening detail pages during frequency control.
8. When the target is reached, enrich detailed answers, regenerate, and verify:
   `python3 scripts/enrich_detailed_answers.py`
   `python3 scripts/generate_notebook.py`
   `python3 scripts/verify_outputs.py --target-discovery-related 1000`

Important: search result pages contain hidden signed detail links shaped like `/search_result/{note_id}?<signed-query>...`; the collector must capture those. Direct naked `/explore/{note_id}` links were tested and frequently produce `mismatched-note`, so they should not be the primary extraction path.

`candidate_pending` means title-related, route-ready candidates that have been found in discovery/search lists but have not yet been opened as detail pages. It is a queue, not a confirmed relevant count. The preferred `run` strategy keeps this queue small instead of building a large backlog: top up only when pending candidates drop below 20, extract at most 20 details per batch, and checkpoint each extracted row immediately.

Detailed answer text is stored in `data/processed/detailed_answers.md` and injected into `notes/llm-interview-master.md`. Keep this file topic-key based so future runs can update statistics without overwriting the long-form explanations. Run `scripts/enrich_detailed_answers.py` before `scripts/generate_notebook.py` so newly promoted or high-frequency follow-up questions get full per-question answers. Each topic should include Chinese interviewer-style questions, display formulas using `$$...$$`, and keep executable code in fenced `python` blocks. Put corpus/version summary only in the top `Version Updates` section; do not add discovery-batch prose inside topic bodies.

The `zsh` steps control Chrome via AppleScript and may require permission/escalation. Raw extraction files under `data/raw/xhs_*` are local intermediates and are gitignored because they can contain full note text and ephemeral Xiaohongshu tokenized URLs. Public tracked outputs should not include personal profile URLs, local absolute paths, signed query parameters, cookies, or favorite order unless the user explicitly opts into a private report with `XHS_INCLUDE_PERSONAL_ORDER=1` or `XHS_INCLUDE_COLLECT_URL=1`. Commit only the processed summaries, source index, scripts, and skill/docs updates.

## Answer Policy

- If the Xiaohongshu note provides a usable answer, summarize it; do not copy the full note.
- If the note only lists questions, write concise interview-ready answers.
- If extraction is incomplete, keep the question but mark the answer `待复核`.
- Do not commit cookies, raw browser profiles, or unrelated personal screenshots.

## Recommended Categories

- Transformer 基础
- Attention 与 KV Cache
- 位置编码
- 归一化与 Loss
- 训练与优化
- 后训练与强化学习
- MoE 与稀疏模型
- 推理服务与性能
- RAG、Agent 与工具调用
- 评估与对齐
- 多模态与具身智能
- 手撕代码
- 公司面经

## Verification

Before finalizing an update:

1. Run `python3 scripts/verify_outputs.py`.
2. Check `git diff --stat`.
3. Confirm uncertain OCR content is marked `待复核`.

## Current Known Limitation

As of 2026-05-25, AppleScript DOM extraction works when Chrome's JavaScript from Apple Events is enabled. Discovery/search detail extraction is viable when tokenized `raw_url` values are collected from the page DOM. Xiaohongshu may return `security-limited` after bursts; prefer small batches and cooldowns. Local OCR is still not verified, so image-only notes should be marked honestly as `title-only` or `待复核` unless their text is available in the page DOM.
