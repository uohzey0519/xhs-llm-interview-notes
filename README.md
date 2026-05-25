# XHS LLM Interview Notes

This repo organizes Xiaohongshu notes related to large-model interview preparation, 面经, and 八股 questions. It can be used with a private favorites page, discovery feed, or search result pages, while keeping local account details out of tracked files.

## Main Outputs

- `notes/llm-interview-master.md`: consolidated category-first study notes with interview-ready answers.
- `data/processed/question_bank.md`: normalized question bank with category, difficulty, source IDs, and frequency.
- `data/processed/detailed_answers.md`: long-form answer source with Chinese interviewer-style questions, `$$` display formulas, code, and interview notes for every normalized question cluster.
- `data/processed/sources.csv`: one row per extracted note, with public note URL, extraction status, relevance, and source ID.
- `skills/xhs-llm-interview-notes/SKILL.md`: reusable Codex skill for future incremental updates.
- `docs/goal_plan.md`: copy-paste goal plan for rerunning the full workflow in another Codex thread.

## Corpus Snapshot

Run date: 2026-05-24

- A private favorites page was opened through the logged-in Chrome session.
- The current DOM inventory exposed 315 unique note URLs, which were used as the first private baseline.
- All 315 accessible baseline notes were opened and extracted.
- 204 baseline notes were classified as large-model/interview related.
- Discovery/search expansion was then run from Xiaohongshu discovery and search pages.
- Discovery/search collected 13,703 unique candidate note IDs, including 1,952 route-ready title-related candidates with signed detail links.
- 1,193 discovery/search details were opened and extracted; 995 were classified as strictly large-model/interview related.
- Combined processed data now contains 1,508 extracted details and 1,199 relevant notes.
- 42 normalized question clusters were generated after promoting recurring discovery/search themes.
- All 42 question clusters have Chinese interviewer-style question lists and detailed answers with `$$` formulas plus code/pseudocode in the generated master note.
- Extraction status across the 315-note baseline: 255 `text-ok`, 60 `title-only`.
- Extraction status across the 204 related baseline notes: 198 `text-ok`, 6 `title-only`.
- The discovery delta and full-corpus overview are summarized in `docs/discovery_delta_report_2026-05-25.md`.

The first complete notebook is in `notes/llm-interview-master.md`; the highest-frequency topics are also summarized in `data/processed/run_summary_2026-05-24.json`.

## Taxonomy

The current notebook groups questions into:

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

## Privacy And Source Policy

Do not commit cookies, raw browser profiles, private screenshots containing unrelated personal content, full Xiaohongshu note dumps, personal profile URLs, or local absolute paths.

Raw extraction artifacts under `data/raw/xhs_*` are local and gitignored because they can contain full note text and temporary Xiaohongshu URL parameters. Tracked files should contain only concise summaries, source IDs, public note URLs without signed query parameters, frequency counts, and original answers written for interview review.

For local favorites inventory, copy `.env.example` to `.env` and set either `XHS_FAVORITES_URL` or `XHS_PROFILE_ID`. Public output hides personal favorite order and profile-note URLs by default. Use `XHS_INCLUDE_PERSONAL_ORDER=1` or `XHS_INCLUDE_COLLECT_URL=1` only for private local reports.

## Update Flow

From the repo root:

1. Configure a private favorites page if needed:
   `cp .env.example .env`
2. Inventory favorites:
   `zsh scripts/xhs_inventory_scroll.zsh`
3. Build the source index:
   `python3 scripts/xhs_build_sources.py data/raw/xhs_inventory_steps_2026-05-24.jsonl`
4. Extract all inventoried notes:
   `zsh scripts/xhs_extract_notes.zsh all`
5. Enrich detailed answers and regenerate notebook outputs:
   `python3 scripts/enrich_detailed_answers.py`
   `python3 scripts/generate_notebook.py`
6. Verify `sources.csv` uniqueness, generated counts, detailed-answer coverage, and raw-file ignore status before committing:
   `python3 scripts/verify_outputs.py`

For a quick incremental pass, run `zsh scripts/xhs_extract_notes.zsh 50` to process the first 50 title-related targets, or compare new inventory URLs against `data/processed/sources.csv` before a narrower extraction.

For discovery/search expansion, prefer the small-queue streaming runner instead of building a large candidate backlog:

```bash
python3 scripts/xhs_discover_run.py run --date 2026-05-24 --target-final 1000 --max-hours 8
```

The runner keeps `candidate_pending` as a queue rather than a goal: it tops up route-ready candidates only when the queue drops below 20, refills toward about 60, extracts at most 20 details per batch, writes each detail row immediately, and stops detail access for 3600 seconds after the first `security-limited` response. Use `live` mode when you want to analyze newly visible cards immediately from the current page:

```bash
python3 scripts/xhs_discover_run.py live --date 2026-05-24 --target-final 1000 --max-hours 8 --entry-url '<search-url>'
```

## Known Limitations

- A Xiaohongshu page may display a larger count than the currently reachable DOM inventory; for this corpus snapshot, the accepted accessible baseline set is 315 notes.
- Some notes are image-heavy. Local OCR is not yet treated as a verified backend, so image-only notes may remain `title-only` or `待复核`.
- The scripts use Chrome AppleScript automation and require Chrome's `View -> Developer -> Allow JavaScript from Apple Events` permission for DOM extraction.
- Xiaohongshu may throttle detail-page access. If `security-limited` appears, stop opening details and wait for the configured cooldown; raw inventory can be retained locally, but raw files under `data/raw/xhs_*` should not be committed.
