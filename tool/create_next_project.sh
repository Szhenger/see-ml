#!/usr/bin/env bash
# Materialize the Two-Plane Overhaul on GitHub: one Issue per body in
# docs/next-project/issues/, plus a Project (v2) board carrying the new
# issues and the existing roadmap issues, with Plane / Origin / Priority
# single-select fields set per item.
#
# Prereqs: gh (authenticated) + jq. Projects need the extra scope once:
#     gh auth refresh -s project
#
# Idempotent: labels are upserted, an issue whose exact title already
# exists is skipped, the project is reused if the title matches, existing
# fields/options are reused, and item-add is a no-op for present items.
set -euo pipefail

REPO="Szhenger/see-ml"
OWNER="Szhenger"
PROJECT_TITLE="SeeML Two-Plane Overhaul"
ISSUE_DIR="$(cd "$(dirname "$0")/../docs/next-project/issues" && pwd)"

# Existing roadmap issues to pull onto the board: number|Plane|Origin|Priority
EXISTING_ITEMS="
59|GPU backend|Roadmap|P0
60|GPU backend|Roadmap|P0
61|GPU backend|Roadmap|P0
62|GPU backend|Roadmap|P0
63|GPU backend|Roadmap|P0
64|GPU backend|Roadmap|P0
65|Gates & docs|Roadmap|P1
66|Core plane|Roadmap|P0
67|Gates & docs|Roadmap|P1
69|Python plane|Roadmap|P1
70|Core plane|Roadmap|P1
71|Core plane|Roadmap|P2
"

for bin in gh jq; do
  command -v "$bin" >/dev/null || { echo "error: $bin is required" >&2; exit 1; }
done
gh auth status >/dev/null 2>&1 || { echo "error: run 'gh auth login' first" >&2; exit 1; }

echo "== labels"
gh label create python-plane -R "$REPO" --force \
  -c 3572A5 -d "Build-host Python subsystem (the frontier plane)" >/dev/null
gh label create core-plane -R "$REPO" --force \
  -c 0E4D64 -d "Deterministic C++ core (device plane)" >/dev/null
gh label create doctrine -R "$REPO" --force \
  -c B34A2E -d "Touches a doctrinal guarantee; product decision required" >/dev/null
gh label create frontier-parity -R "$REPO" --force \
  -c 5319E7 -d "Parity with the training frontier (torch.compile, MLX-LM): SeeRL F1-F6" >/dev/null
echo "   ok: python-plane, core-plane, doctrine, frontier-parity"

# --- front-matter helpers -------------------------------------------------
fm() {  # fm <file> <key>
  awk -v key="$2" '
    /^---$/ { c++; next }
    c == 1 && $0 ~ "^" key ": " {
      sub("^" key ": ", ""); gsub(/^"|"$/, ""); print; exit
    }' "$1"
}
body() { awk '/^---$/ { c++; next } c >= 2 { print }' "$1"; }

# --- create the issues ----------------------------------------------------
echo "== issues"
ITEMS=""  # url|Plane|Origin|Priority
for f in "$ISSUE_DIR"/p*.md "$ISSUE_DIR"/e*.md "$ISSUE_DIR"/f*.md; do
  title=$(fm "$f" title)
  labels=$(fm "$f" labels)
  plane=$(fm "$f" plane)
  origin=$(fm "$f" origin)
  priority=$(fm "$f" priority)

  existing=$(gh issue list -R "$REPO" --state all --limit 200 \
      --json number,title |
    jq -r --arg t "$title" '.[] | select(.title == $t) | .number' | head -1)
  if [ -n "$existing" ]; then
    url="https://github.com/$REPO/issues/$existing"
    echo "   skip (exists as #$existing): $title"
  else
    label_args=()
    IFS=',' read -ra ls <<< "$labels"
    for l in "${ls[@]}"; do label_args+=(-l "$l"); done
    url=$(body "$f" | gh issue create -R "$REPO" -t "$title" -F - "${label_args[@]}")
    echo "   created: $url  ($(basename "$f"))"
  fi
  ITEMS+="$url|$plane|$origin|$priority"$'\n'
done
while IFS='|' read -r num plane origin priority; do
  [ -n "$num" ] || continue
  ITEMS+="https://github.com/$REPO/issues/$num|$plane|$origin|$priority"$'\n'
done <<< "$EXISTING_ITEMS"

# --- create or reuse the project ------------------------------------------
echo "== project"
proj_json=$(gh project list --owner "$OWNER" --format json --limit 100 |
  jq -c --arg t "$PROJECT_TITLE" 'first(.projects[] | select(.title == $t)) // empty')
if [ -n "$proj_json" ]; then
  echo "   reusing existing project"
else
  proj_json=$(gh project create --owner "$OWNER" --title "$PROJECT_TITLE" --format json)
fi
proj_num=$(jq -r '.number' <<< "$proj_json")
proj_id=$(jq -r '.id' <<< "$proj_json")
proj_url=$(jq -r '.url // empty' <<< "$proj_json")
echo "   project #$proj_num ${proj_url:+at $proj_url}"

ensure_field() {  # ensure_field <name> <comma-separated options>
  local have
  have=$(gh project field-list "$proj_num" --owner "$OWNER" --format json |
    jq -r --arg n "$1" '.fields[] | select(.name == $n) | .id' | head -1)
  if [ -z "$have" ]; then
    gh project field-create "$proj_num" --owner "$OWNER" --name "$1" \
      --data-type SINGLE_SELECT --single-select-options "$2" >/dev/null
    echo "   field created: $1"
  fi
}
ensure_field "Plane"    "Python plane,Core plane,GPU backend,Gates & docs"
ensure_field "Origin"   "Performance audit,Frontier bridge,Both audits,Nightly eval,Roadmap,Seam review"
ensure_field "Priority" "P0,P1,P2"

fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)

set_field() {  # set_field <item-id> <field-name> <option-name>
  local fid oid
  fid=$(jq -r --arg n "$2" '.fields[] | select(.name == $n) | .id' <<< "$fields_json")
  oid=$(jq -r --arg n "$2" --arg o "$3" \
    '.fields[] | select(.name == $n) | .options[] | select(.name == $o) | .id' \
    <<< "$fields_json")
  if [ -z "$fid" ] || [ -z "$oid" ]; then
    echo "   warn: no option '$3' for field '$2'" >&2; return 0
  fi
  gh project item-edit --id "$1" --project-id "$proj_id" \
    --field-id "$fid" --single-select-option-id "$oid" >/dev/null
}

echo "== board items"
while IFS='|' read -r url plane origin priority; do
  [ -n "$url" ] || continue
  item_id=$(gh project item-add "$proj_num" --owner "$OWNER" --url "$url" \
    --format json | jq -r '.id')
  set_field "$item_id" "Plane" "$plane"
  [ -n "$origin" ] && set_field "$item_id" "Origin" "$origin"  # the F bodies name no origin
  set_field "$item_id" "Priority" "$priority"
  echo "   $url  [$plane | $origin | $priority]"
done <<< "$ITEMS"

echo
echo "Done. Board: ${proj_url:-https://github.com/users/$OWNER/projects/$proj_num}"
