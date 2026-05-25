# Goal-Ready Plan

Objective: Build and incrementally update a git-backed LLM interview question notebook from the user's Xiaohongshu favorites, discovery feed, and search pages.

## Copy-Paste Goal Prompt

Use the following text as the goal in a fresh Codex window:

```text
Goal: Build or incrementally update a Xiaohongshu large-model interview notebook from the current repository.

Context:
- The repo already exists locally and is connected to GitHub.
- The reusable skill is tracked in the repo under skills/xhs-llm-interview-notes/SKILL.md.
- For favorites collection, set either XHS_FAVORITES_URL or XHS_PROFILE_ID in a local .env file copied from .env.example.
- The target is 收藏 -> 笔记, especially notes related to 大模型面经 / LLM interview 八股.

Primary objective:
Open my Xiaohongshu favorites/discovery/search pages, identify large-model/interview-related notes, open matching notes one by one, extract questions and available answers, write missing answers yourself, deduplicate questions, classify them, count frequency, and generate a git-tracked Markdown notebook.

For discovery/search expansion, use the small-queue streaming strategy instead of building a large candidate backlog:

`python3 scripts/xhs_discover_run.py run --date 2026-05-24 --target-final 1000 --max-hours 8`

The runner should keep `candidate_pending` as a bounded queue, not a success metric: top up route-ready candidates only when the queue drops below 20, refill toward about 60, extract at most 20 details per batch, write each extracted row immediately, regenerate processed notebook outputs every 100 newly extracted discovery rows, and stop detail access for 3600 seconds after the first `security-limited` response.

Required outputs:
1. notes/llm-interview-master.md
   - Category-first interview notes.
   - Each question should have a concise interview-ready answer.
   - Include source IDs for traceability.
   - Mark uncertain extraction/OCR as 待复核.

2. data/processed/question_bank.md
   - Normalized question table.
   - Columns should include: question, category, frequency, difficulty if inferable, source IDs, status.
   - Sort by category first, then frequency descending.

3. data/processed/detailed_answers.md
   - Long-form answer source keyed by normalized topic ID.
   - Every normalized question should have Chinese interviewer-style question lists, `$$` display formulas, derivations, code or pseudocode, and common pitfalls.
   - Run `scripts/enrich_detailed_answers.py` before `scripts/generate_notebook.py` so each high-frequency follow-up has its own detailed answer.
   - `scripts/generate_notebook.py` should inject these answers into `notes/llm-interview-master.md`.
   - Keep version/corpus update prose only at the top `Version Updates` section, not inside topic bodies.

4. data/processed/sources.csv
   - One row per visited Xiaohongshu note.
   - Include: source_id, title, public note URL, author if useful, extraction_status, is_llm_related, processed_at, notes.
   - Do not include personal profile URLs, signed query parameters, local absolute paths, or favorite order in public output.
   - Do not duplicate URLs.

5. data/raw/
   - Save only safe extraction artifacts that are useful for audit, such as text snippets or screenshot references.
   - Do not save cookies, browser profiles, tokens, or unrelated personal screenshots.

5. Git commit and push.
   - Commit generated notebook outputs and any skill/docs updates.
   - Push to origin/main.

Execution rules:
- First load and follow the xhs-llm-interview-notes skill.
- Use Chrome's logged-in Xiaohongshu session.
- If Chrome DOM extraction works, use it for speed.
- If DOM extraction is blocked, use Computer Use accessibility output and screenshots.
- If a note is image-heavy and OCR is unreliable, visually inspect the screenshot when possible; otherwise record the note as partial and mark affected questions/answers 待复核.
- Do not get stuck on a single note for too long. If extraction is blocked after reasonable attempts, record it in sources.csv and move on.
- Do not keep opening detail pages after `security-limited` or `cooldown-active`; report current counts and wait for a later run.
- Do not use naked `/explore/{note_id}` URLs as the primary discovery detail route. Prefer signed `/search_result/{note_id}?...` URLs collected from search/discovery DOM.
- Do not copy full Xiaohongshu notes verbatim. Summarize source answers and write original interview-ready answers.
- Preserve source traceability for every question.

Classification taxonomy:
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
- 其他/待分类

Frequency policy:
- Treat semantically equivalent questions as one normalized question.
- Count frequency by distinct source notes, not repeated mentions inside one note.
- If two questions are similar but test different depth, keep both and cross-reference them.

Answer policy:
- If the source provides a usable answer, summarize it accurately.
- If the source only lists questions, write a strong concise answer yourself.
- Prefer practical interview explanations: definition, intuition, key formula or mechanism, tradeoffs, common pitfalls, and one example when useful.
- Mark any uncertain answer as 待复核 instead of pretending certainty.

Checkpointing:
- After roughly every 20 relevant notes, save outputs and check git diff.
- Keep sources.csv updated continuously so the run can resume if interrupted.
- Before final commit, verify:
  1. sources.csv has unique URLs.
  2. question_bank.md frequency counts match source IDs.
  3. master notes include all normalized questions.
  4. uncertain OCR/extraction is clearly marked 待复核.
  5. git status only includes intended project files.

Final response:
- Report processed note count, relevant note count, normalized question count, top high-frequency questions, files changed, commit SHA, and push status.
- Explicitly list any blocked/partial notes and what user action would unblock them.
```

## Preconditions

1. For favorites collection, set `XHS_FAVORITES_URL` or `XHS_PROFILE_ID` in `.env`, then open Chrome to that favorites page.
2. Confirm the page is logged in and shows `收藏 -> 笔记`.
3. Strongly recommended: in Chrome, enable `View -> Developer -> Allow JavaScript from Apple Events`.
   This allows fast DOM extraction. Without it, use the accessibility-tree fallback, which is slower.
4. Run commands from the repository root.

## Current Baseline From 2026-05-24 / 2026-05-25

- The first full run accepted 315 currently accessible favorite notes as the complete working set.
- All 315 notes were opened and extracted.
- 204 notes were classified as large-model/interview related.
- Discovery/search expansion opened 1,193 detail pages and confirmed 995 strictly related large-model/interview notes.
- Combined processed corpus: 1,508 extracted details, 1,199 relevant notes, 42 normalized question clusters, 42 detailed answer sections.
- Current discovery inventory: 13,703 unique note IDs, 1,952 route-ready title-related candidates, and 784 pending route-ready candidates that were found but not opened as details.
- 42 detailed answer sections with formulas/code were generated and injected into the master note.
- Follow-up runs should treat `data/processed/sources.csv` as the baseline and focus on newly discovered URLs. See `docs/discovery_delta_report_2026-05-25.md` for the current comparison report.

## Execution Checklist

1. Inventory favorites.
   - Scroll the favorites page from top to bottom.
   - Capture title, author, like count, note URL, and approximate order.
   - Save to `data/processed/sources.csv`.

2. Filter LLM/interview-related notes.
   - Include titles or text containing terms such as 大模型, LLM, Agent, RAG, SFT, RLHF, DPO, GRPO, PPO, MoE, Attention, KV Cache, Transformer, DeepSeek, Claude Code, 面试, 一面, 二面, 算法岗.
   - Exclude travel, personal posts, finance journals, and unrelated life content unless they include model-interview material.

3. Extract each selected note.
   - Open the note.
   - Read accessible title and body text.
   - For image-heavy cards or long images, capture the note image region and read it visually.
   - Treat `scripts/ocr_image.swift` as experimental until the local Swift/Vision toolchain issue is fixed.
   - Record extraction status: `text-ok`, `ocr-ok`, `partial`, `blocked`, or `irrelevant`.

4. Normalize questions.
   - Split note content into atomic questions.
   - Deduplicate semantically similar questions.
   - Preserve source titles and URLs per question.
   - Count frequency by normalized question and category.

5. Write answers.
   - If the source note has a reliable answer, summarize it in original wording, not a verbatim dump.
   - If the source has only a question, write a concise interview-ready answer.
   - Mark uncertain OCR or incomplete extraction as `待复核`.

6. Generate outputs.
   - `notes/llm-interview-master.md`: category-first notes with answers.
   - `data/processed/question_bank.md`: table of normalized questions, categories, frequency, and source count.
   - `data/processed/sources.csv`: all visited notes and extraction status.
   - Commit the generated artifacts.

7. Future update loop.
   - Re-run inventory.
   - Compare note URLs against `sources.csv`.
   - Only process new or changed favorites.
   - Append/update the question bank and master notes.

8. Discovery/search streaming update loop.
   - Run `python3 scripts/xhs_discover_run.py status --date 2026-05-24 --target-final <target>` first.
   - If `candidate_pending` is above 20, extract a small batch before collecting more inventory.
   - If `candidate_pending` is below 20, inventory a search/discovery page until the queue reaches about 60 route-ready candidates.
   - After each detail extraction, append to `data/raw/xhs_discover_note_extracts_*.jsonl` immediately.
   - After each 100 newly extracted discovery rows, regenerate processed outputs.
   - If `security-limited` appears, stop detail access and wait for cooldown.

## Known Risks

- Xiaohongshu may lazy-load notes and recycle DOM nodes, so inventory must scroll slowly and deduplicate by URL.
- Xiaohongshu may throttle detail-page access; use conservative sleeps and cooldowns rather than forcing the last few notes.
- Some important content is inside images; OCR or visual reading is required for complete extraction.
- Chrome DOM extraction is blocked unless JavaScript from Apple Events is enabled.
- `scripts/ocr_image.swift` currently fails in this environment with Swift module import errors around `SwiftBridging`; do not rely on it as a verified OCR path yet.
- Full note copying can create copyright/privacy noise; the repo should store summaries, source metadata, and original answers written by the agent.
