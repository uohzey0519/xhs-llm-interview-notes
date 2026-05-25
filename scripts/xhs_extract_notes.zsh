#!/usr/bin/env zsh
set -euo pipefail

repo_root=${0:A:h:h}
mode=${1:-related}
if [[ "$mode" == <-> ]]; then
  limit="$mode"
  mode="related"
else
  limit=${2:-999999}
fi
out_dir="$repo_root/data/raw"
mkdir -p "$out_dir"
urls_file="$out_dir/xhs_extract_targets_2026-05-24.tsv"
out_file="$out_dir/xhs_note_extracts_2026-05-24.jsonl"
extract_js=$(mktemp -t xhs_extract.XXXXXX.js)
trap 'rm -f "$extract_js"' EXIT

python3 - "$repo_root/data/raw/xhs_inventory_2026-05-24.json" "$urls_file" "$mode" "$limit" <<'PY'
import json
import sys
from pathlib import Path

inventory = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
mode = sys.argv[3]
limit = int(sys.argv[4])

if mode == "all":
    targets = list(inventory)
elif mode == "related":
    targets = [x for x in inventory if x.get("is_llm_related_by_title")]
else:
    raise SystemExit(f"unknown mode: {mode}; use related, all, or a numeric related limit")
targets = targets[:limit]
with out.open("w", encoding="utf-8") as f:
    for item in targets:
        f.write("\t".join([
            item.get("note_id", ""),
            item.get("title", "").replace("\t", " "),
            item.get("author", "").replace("\t", " "),
            item.get("collect_url") or item.get("url", ""),
        ]) + "\n")
print(f"{len(targets)} {mode} targets -> {out}")
PY

total_targets=$(wc -l < "$urls_file" | tr -d ' ')

cat > "$extract_js" <<'JS'
(() => {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const pick = (sel) => {
    const el = document.querySelector(sel);
    return el ? clean(el.innerText || el.textContent || "") : "";
  };
  const title =
    pick("#detail-title") ||
    pick(".note-content #detail-title") ||
    clean(document.title.replace(/\s*-\s*小红书\s*$/, ""));
  const desc =
    pick("#detail-desc") ||
    pick("#detail-desc .note-text") ||
    pick(".note-content .desc") ||
    pick(".note-content");
  const date = pick(".bottom-container .date") || pick(".bottom-container");
  const author = pick(".author .name") || pick(".user-name") || "";
  const commentsText = pick(".comments-container");
  const imageCount = document.querySelectorAll("img").length;
  const noteTextCount = document.querySelectorAll("#detail-desc .note-text, .note-content .note-text").length;
  return JSON.stringify({
    url: location.href,
    page_title: document.title,
    title,
    desc,
    date,
    author,
    image_count: imageCount,
    note_text_count: noteTextCount,
    comments_excerpt: commentsText.slice(0, 1200)
  });
})();
JS

run_js() {
  local js_file="$1"
  /usr/bin/osascript \
    -e 'on run argv' \
    -e 'set jsPath to item 1 of argv' \
    -e 'set jsSource to read POSIX file jsPath' \
    -e 'tell application "Google Chrome" to tell active tab of front window to execute javascript (jsSource as text)' \
    -e 'end run' \
    "$js_file"
}

set_url() {
  local url="$1"
  /usr/bin/osascript -e 'on run argv' -e 'tell application "Google Chrome" to set URL of active tab of front window to item 1 of argv' -e 'end run' "$url" >/dev/null
}

: > "$out_file"
idx=0
while IFS=$'\t' read -r note_id title author url; do
  idx=$((idx + 1))
  print "extract $(printf '%03d' "$idx")/$total_targets $note_id $title"
  set_url "$url"
  sleep 2.2
  payload=$(run_js "$extract_js" || true)
  if [[ -z "$payload" ]]; then
    payload='{}'
  fi
  python3 - "$note_id" "$title" "$author" "$url" "$payload" <<'PY' >> "$out_file"
import json
import sys

note_id, title, author, source_url, payload = sys.argv[1:6]
try:
    data = json.loads(payload)
except Exception as exc:
    data = {"extract_error": str(exc), "raw_payload": payload[:1000]}
data.update({
    "note_id": note_id,
    "source_title": title,
    "source_author": author,
    "source_url": source_url,
})
desc_len = len(data.get("desc") or "")
data["extract_status"] = "text-ok" if desc_len >= 40 else ("title-only" if data.get("title") else "blocked")
print(json.dumps(data, ensure_ascii=False))
PY
done < "$urls_file"

print "$out_file"
