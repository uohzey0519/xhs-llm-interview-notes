# Trial Report 2026-05-24

## What Worked

- Chrome already had the user's Xiaohongshu profile favorites page open.
- The page showed `收藏 -> 笔记` with 353 notes.
- First-screen note titles and authors were visible through Computer Use accessibility output.
- A relevant sample note opened successfully:
  `面试被问到claude code的检索是咋做的？`
- The modal exposed title and body text, enough to extract at least text-based Q&A.

## Sample Extraction

The sample note was normalized into:

- Source row: `data/processed/sources.csv`
- Sample note: `data/samples/sample_note_claude_code.md`
- Master note section: `notes/llm-interview-master.md`
- Question bank row: `data/processed/question_bank.md`

## Blockers

- Chrome DOM extraction through AppleScript failed:
  `Allow JavaScript from Apple Events` is disabled.
- OCR is not verified:
  - Swift/Vision script fails due local Command Line Tools module import errors.
  - Python does not have PyObjC Vision installed.
  - `shortcuts list` could not communicate with its helper app.

## Practical Next Path

1. Enable Chrome JavaScript from Apple Events for fast extraction.
2. Use accessibility fallback for visible page inventory and text notes.
3. For image-heavy notes, use Codex visual reading from screenshots until OCR is fixed.
4. Keep all outputs as summarized notes with source links, not raw content dumps.

