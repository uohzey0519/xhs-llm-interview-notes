#!/usr/bin/env zsh
set -euo pipefail

repo_root=${0:A:h:h}
out_dir="$repo_root/data/raw"
mkdir -p "$out_dir"
out_file="$out_dir/xhs_inventory_steps_2026-05-24.jsonl"
: > "$out_file"

if [[ -f "$repo_root/.env" ]]; then
  set -a
  source "$repo_root/.env"
  set +a
fi

profile_id="${XHS_PROFILE_ID:-}"
favorites_url="${XHS_FAVORITES_URL:-}"
if [[ -z "$favorites_url" && -n "$profile_id" ]]; then
  favorites_url="https://www.xiaohongshu.com/user/profile/$profile_id?tab=fav&subTab=note"
fi
if [[ -z "$favorites_url" ]]; then
  print -u2 "Set XHS_FAVORITES_URL or XHS_PROFILE_ID before running inventory."
  print -u2 "Example: XHS_FAVORITES_URL='https://www.xiaohongshu.com/user/profile/<profile-id>?tab=fav&subTab=note' zsh scripts/xhs_inventory_scroll.zsh"
  exit 2
fi
if [[ -z "$profile_id" && "$favorites_url" == *"/user/profile/"* ]]; then
  profile_id="${${favorites_url#*/user/profile/}%%[/?#]*}"
fi
collect_js=$(mktemp -t xhs_collect.XXXXXX.js)
scroll_js=$(mktemp -t xhs_scroll.XXXXXX.js)
top_js=$(mktemp -t xhs_top.XXXXXX.js)
trap 'rm -f "$collect_js" "$scroll_js" "$top_js"' EXIT

cat > "$top_js" <<'JS'
window.scrollTo(0, 0); 'ok';
JS

cat > "$scroll_js" <<'JS'
window.scrollBy(0, Math.floor(window.innerHeight * 0.50)); 'ok';
JS

cat > "$collect_js" <<JS
(() => {
  const profileId = "$profile_id";
  const noteIdRe = /\\/(?:explore|user\\/profile\\/[^/?#]+)\\/([0-9a-f]{24})(?:[/?#]|$)/;
  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();

  function noteIdFromHref(href) {
    const m = String(href || "").match(noteIdRe);
    return m ? m[1] : "";
  }

  function smallText(node) {
    return clean(node && (node.innerText || node.textContent || ""));
  }

  function bestCard(anchor) {
    let node = anchor;
    let best = anchor;
    for (let i = 0; i < 8 && node; i += 1) {
      const text = smallText(node);
      const links = Array.from(node.querySelectorAll ? node.querySelectorAll("a") : []);
      if (text.length > 0 && text.length < 500 && links.length >= 2) {
        best = node;
      }
      node = node.parentElement;
    }
    return best;
  }

  const selector = profileId ? \`a[href*="/user/profile/\${profileId}/"]\` : 'a[href*="/user/profile/"]';
  const titleLinks = Array.from(document.querySelectorAll(selector))
    .filter((a) => noteIdFromHref(a.href) && smallText(a));

  const items = [];
  for (const a of titleLinks) {
    const noteId = noteIdFromHref(a.href);
    const card = bestCard(a);
    const links = Array.from(card.querySelectorAll("a"));
    const authorLink = links.find((l) =>
      l !== a &&
      l.href.includes("/user/profile/") &&
      (!profileId || !l.href.includes(\`/user/profile/\${profileId}/\`)) &&
      smallText(l)
    );
    const nums = smallText(card).match(/(?:\\d+(?:\\.\\d+)?万?)/g) || [];
    const rect = a.getBoundingClientRect();
    items.push({
      note_id: noteId,
      title: smallText(a),
      author: authorLink ? smallText(authorLink) : "",
      url: \`https://www.xiaohongshu.com/explore/\${noteId}\`,
      collect_url: a.href,
      like_hint: nums.length ? nums[nums.length - 1] : "",
      y: Math.round(window.scrollY + rect.top)
    });
  }

  const metrics = {
    location: location.href,
    scroll_y: Math.round(window.scrollY),
    inner_height: window.innerHeight,
    document_height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
    visible_count: items.length
  };

  return JSON.stringify({metrics, items});
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

set_url "$favorites_url"
sleep 2
run_js "$top_js" >/dev/null
sleep 1

last_count=0
stable_rounds=0

for step in {0..179}; do
  payload=$(run_js "$collect_js")
  printf '%s\n' "$payload" >> "$out_file"

  counts=$(python3 - "$out_file" <<'PY'
import json
import sys
seen = set()
last = None
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    payload = json.loads(line)
    last = payload
    for item in payload.get("items", []):
        if item.get("note_id"):
            seen.add(item["note_id"])
metrics = last.get("metrics", {}) if last else {}
print(len(seen), metrics.get("visible_count", 0), metrics.get("scroll_y", 0), metrics.get("document_height", 0), metrics.get("inner_height", 0))
PY
)
  read collected visible scroll_y doc_h inner_h <<< "$counts"
  print "step=$(printf '%02d' "$step") collected=$collected visible=$visible scroll=$scroll_y/$doc_h"

  if [[ "$collected" == "$last_count" ]]; then
    stable_rounds=$((stable_rounds + 1))
  else
    stable_rounds=0
    last_count="$collected"
  fi

  if (( scroll_y + inner_h >= doc_h - 8 && stable_rounds >= 3 )); then
    break
  fi

  run_js "$scroll_js" >/dev/null
  sleep 1.15
done

print "$out_file"
