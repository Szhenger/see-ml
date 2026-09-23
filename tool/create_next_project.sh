#!/usr/bin/env bash
# Reconcile the Two-Plane Overhaul with GitHub. The bodies in
# docs/next-project/issues/ are the source of truth for one Issue each,
# plus a Project (v2) board carrying those issues and the existing roadmap
# issues, with Plane / Origin / Priority single-select fields per item.
#
# Prereqs: gh (authenticated) + jq. Projects need the extra scope once:
#     gh auth refresh -s project
#
# What the file owns, and what it does not (docs/next-project/README.md,
# "Materializing the board"):
#   * the issue's body, compared after the same normalization GitHub applies;
#   * its labels as a FLOOR: every label the file names is present, labels
#     added live for triage (sev:*, good first issue, ...) are kept;
#   * its milestone, only when the file names one (`milestone:`);
#   * an issue that is CLOSED is left alone — its body is history — and
#     only its board fields are reconciled.
# Every label and milestone a body names is upserted; the issue list and
# the board are each fetched once, paginated, so nothing stops at a page
# boundary; a board field is written only when its value differs. A
# single-select option that an existing field lacks cannot be added by the
# Projects CLI (the option-edit mutation refuses to append): the script
# names it as an ACTION for the web UI and carries on.
#
#     tool/create_next_project.sh              # reconcile
#     tool/create_next_project.sh --dry-run    # print every write it would make, touch nothing
set -euo pipefail

usage() {
  echo "usage: $0 [--dry-run | -n]" >&2
  exit "${1:-2}"
}
DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run|-n) DRY_RUN=1 ;;
  -h|--help) usage 0 ;;
  *) echo "error: unknown argument '$1'" >&2; usage ;;
esac
[ $# -le 1 ] || usage

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

# Labels the bodies use that GitHub does not create by default: name|color|description.
# Any other label a body names is upserted grey with an empty description.
LABEL_TABLE="
python-plane|3572A5|Build-host Python subsystem (the frontier plane)
core-plane|0E4D64|Deterministic C++ core (device plane)
doctrine|B34A2E|Touches a doctrinal guarantee; product decision required
frontier-parity|5319E7|Parity with the training frontier (torch.compile, MLX-LM): SeeRL F1-F6
"

for bin in gh jq; do
  command -v "$bin" >/dev/null || { echo "error: $bin is required" >&2; exit 1; }
done
gh auth status >/dev/null 2>&1 || { echo "error: run 'gh auth login' first" >&2; exit 1; }

# Reads call gh directly. Every WRITE goes through mutate, which under
# --dry-run prints the intent on stderr (so a caller's stdout redirect
# cannot swallow it) and does nothing; otherwise it runs the command and
# passes its stdout through for the caller to capture or discard.
mutate() {  # mutate "<what>" gh-args...
  local what="$1"; shift
  if [ "$DRY_RUN" = 1 ]; then echo "   would: $what" >&2; return 0; fi
  gh "$@"
}

# --- front-matter helpers -------------------------------------------------
fm() {  # fm <file> <key>  (empty when the key is absent)
  awk -v key="$2" '
    /^---$/ && c < 2 { c++; next }
    c == 1 && $0 ~ "^" key ": " {
      sub("^" key ": ", ""); gsub(/^"|"$/, ""); print; exit
    }' "$1"
}
# Only the first two `---` lines fence the front matter; a horizontal rule
# inside the body is body.
body() { awk '/^---$/ && c < 2 { c++; next } c >= 2 { print }' "$1"; }
# GitHub stores CRLF-free text and drops trailing blank lines: compare the
# same normalization on both sides.
norm() { tr -d '\r' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}'; }
# A comma-separated front-matter list as a sorted, trimmed, unique JSON
# array — all set arithmetic on labels happens in jq, in one ordering.
list_json() { jq -R -c 'split(",") | map(gsub("^\\s+|\\s+$"; "")) | map(select(. != "")) | unique'; }

BODIES=("$ISSUE_DIR"/*.md)

# --- labels: every label any body names ----------------------------------
echo "== labels"
have_labels=$(gh label list -R "$REPO" --limit 500 --json name | jq -c '[.[].name]')
wanted_labels=$(for f in "${BODIES[@]}"; do fm "$f" labels; done | paste -sd, - | list_json)
jq -r '.[]' <<< "$wanted_labels" | while IFS= read -r l; do
  row=$(awk -F'|' -v l="$l" '$1 == l' <<< "$LABEL_TABLE")
  if [ -n "$row" ]; then
    color=$(cut -d'|' -f2 <<< "$row"); desc=$(cut -d'|' -f3 <<< "$row")
    mutate "upsert label '$l'" label create "$l" -R "$REPO" --force -c "$color" -d "$desc" >/dev/null
  elif ! jq -e --arg l "$l" 'index($l) != null' <<< "$have_labels" >/dev/null; then
    mutate "create label '$l' (grey)" label create "$l" -R "$REPO" -c EDEDED -d "" >/dev/null
  fi
done
echo "   ok: $(jq -r 'join(" ")' <<< "$wanted_labels")"

# --- milestones: every milestone any body names --------------------------
echo "== milestones"
have_ms=$(gh api --paginate "repos/$REPO/milestones?state=all&per_page=100" | jq -c '[.[].title]')
for f in "${BODIES[@]}"; do fm "$f" milestone; done | sed '/^$/d' | LC_ALL=C sort -u |
while IFS= read -r m; do
  if jq -e --arg m "$m" 'index($m) != null' <<< "$have_ms" >/dev/null; then echo "   ok: $m"
  else mutate "create milestone '$m'" api "repos/$REPO/milestones" -f title="$m" >/dev/null; fi
done

# --- issues: one fetch, then create or edit per body ---------------------
echo "== issues"
LIVE=$(gh api --paginate "repos/$REPO/issues?state=all&per_page=100" |
  jq -c '.[] | select(.pull_request == null) |
         {number, title, state, body: (.body // ""), milestone: (.milestone.title // ""),
          labels: ([.labels[].name] | unique)}')
ITEMS=""  # url|Plane|Origin|Priority
for f in "${BODIES[@]}"; do
  title=$(fm "$f" title)
  labels_json=$(fm "$f" labels | list_json)
  labels_csv=$(jq -r 'join(",")' <<< "$labels_json")
  plane=$(fm "$f" plane)
  origin=$(fm "$f" origin)
  priority=$(fm "$f" priority)
  milestone=$(fm "$f" milestone)
  want_body=$(body "$f" | norm)

  live=$(jq -c --arg t "$title" 'select(.title == $t)' <<< "$LIVE" | head -1)
  if [ -z "$live" ]; then
    url=$(body "$f" | mutate "create issue '$title' [labels: ${labels_csv:-none}; milestone: ${milestone:-none}]" \
      issue create -R "$REPO" -t "$title" -F - \
      ${labels_csv:+-l "$labels_csv"} ${milestone:+-m "$milestone"})
    [ -n "$url" ] || url="https://github.com/$REPO/issues/new?title=$(basename "$f")"
    [ "$DRY_RUN" = 1 ] || echo "   created: $url  ($(basename "$f"))"
  else
    num=$(jq -r .number <<< "$live")
    url="https://github.com/$REPO/issues/$num"
    if [ "$(jq -r .state <<< "$live")" = "closed" ]; then
      echo "   closed, left alone (#$num): $title"
    else
      live_body=$(jq -r .body <<< "$live" | norm)
      add_labels=$(jq -r --argjson want "$labels_json" '($want - .labels) | join(",")' <<< "$live")
      live_ms=$(jq -r .milestone <<< "$live")
      what=""
      [ "$live_body" = "$want_body" ] || what="$what body"
      [ -z "$add_labels" ] || what="$what labels(+$add_labels)"
      if [ -n "$milestone" ] && [ "$live_ms" != "$milestone" ]; then what="$what milestone"; fi
      what="${what# }"
      if [ -z "$what" ]; then
        echo "   in sync (#$num): $title"
      else
        # bash 3.2 + set -u refuses "${empty[@]}": each flag is passed
        # through a guarded expansion instead of an array.
        body_flag=""; case "$what" in *body*) body_flag=1 ;; esac
        ms_flag=""; case "$what" in *milestone*) ms_flag=1 ;; esac
        body "$f" | mutate "edit #$num [$what]" issue edit "$num" -R "$REPO" \
          ${body_flag:+-F -} \
          ${add_labels:+--add-label "$add_labels"} \
          ${ms_flag:+-m "$milestone"} >/dev/null
        [ "$DRY_RUN" = 1 ] || echo "   edited #$num [$what]: $title"
      fi
    fi
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
elif [ "$DRY_RUN" = 1 ]; then
  echo "   would: create project '$PROJECT_TITLE' — nothing to reconcile against yet; stopping" >&2
  exit 0
else
  proj_json=$(gh project create --owner "$OWNER" --title "$PROJECT_TITLE" --format json)
fi
proj_num=$(jq -r '.number' <<< "$proj_json")
proj_id=$(jq -r '.id' <<< "$proj_json")
proj_url=$(jq -r '.url // empty' <<< "$proj_json")
echo "   project #$proj_num ${proj_url:+at $proj_url}"

fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)
PENDING_FIELDS=""  # fields a dry run would have created: their values cannot be set yet

ensure_field() {  # ensure_field <name> <comma-separated options>
  local have missing
  have=$(jq -r --arg n "$1" '.fields[] | select(.name == $n) | .id' <<< "$fields_json" | head -1)
  if [ -z "$have" ]; then
    mutate "create field '$1' with options [$2]" project field-create "$proj_num" --owner "$OWNER" \
      --name "$1" --data-type SINGLE_SELECT --single-select-options "$2" >/dev/null
    PENDING_FIELDS="$PENDING_FIELDS $1 "
    return 0
  fi
  # The field exists: an option it lacks has to be added by hand (the
  # Projects option-edit mutation refuses to append). Say which.
  missing=$(jq -r --arg n "$1" --argjson want "$(list_json <<< "$2")" \
    '[.fields[] | select(.name == $n) | .options[].name] as $have | ($want - $have) | join(", ")' \
    <<< "$fields_json")
  [ -z "$missing" ] || echo "   ACTION: field '$1' lacks option(s) '$missing' — add in the web UI (project settings → $1), then rerun" >&2
}
# The option sets are the union of what the bodies and the roadmap rows
# name, so a new origin or plane in a body is never silently unset.
opts() {  # opts <front-matter key> <EXISTING_ITEMS column>
  { for f in "${BODIES[@]}"; do fm "$f" "$1"; done
    sed '/^$/d' <<< "$EXISTING_ITEMS" | cut -d'|' -f"$2"; } | sed '/^$/d' | LC_ALL=C sort -u | paste -sd, -
}
ensure_field "Plane"    "$(opts plane 2)"
ensure_field "Origin"   "$(opts origin 3)"
ensure_field "Priority" "P0,P1,P2"

[ "$DRY_RUN" = 1 ] || fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)
# One fetch of the board: item ids and current field values, so a run
# that changes nothing writes nothing.
items_json=$(gh project item-list "$proj_num" --owner "$OWNER" --format json --limit 500)

set_field() {  # set_field <item-id> <item-url> <field-name> <option-name>
  local fid oid key current
  [ -n "$4" ] || return 0
  case "$PENDING_FIELDS" in *" $3 "*) return 0 ;; esac  # would be created first
  fid=$(jq -r --arg n "$3" '.fields[] | select(.name == $n) | .id' <<< "$fields_json")
  oid=$(jq -r --arg n "$3" --arg o "$4" \
    '.fields[] | select(.name == $n) | .options[] | select(.name == $o) | .id' \
    <<< "$fields_json")
  if [ -z "$fid" ] || [ -z "$oid" ]; then
    echo "   warn: no option '$4' for field '$3' (see ACTION above)" >&2; return 0
  fi
  # item-list keys a field's value by its lower-cased name.
  key=$(tr '[:upper:]' '[:lower:]' <<< "$3")
  current=$(jq -r --arg u "$2" --arg k "$key" --arg n "$3" \
    '.items[] | select(.content.url == $u) | (.[$k] // .[$n] // "")' <<< "$items_json" | head -1)
  [ "$current" = "$4" ] && return 0
  mutate "set $3 = '$4' on $2 (was '${current:-unset}')" project item-edit --id "$1" \
    --project-id "$proj_id" --field-id "$fid" --single-select-option-id "$oid" >/dev/null
}

echo "== items"
while IFS='|' read -r url plane origin priority; do
  [ -n "$url" ] || continue
  item_id=$(jq -r --arg u "$url" '.items[] | select(.content.url == $u) | .id' <<< "$items_json" | head -1)
  if [ -z "$item_id" ]; then
    item_id=$(mutate "add $url to the board" project item-add "$proj_num" --owner "$OWNER" \
      --url "$url" --format json | jq -r '.id // empty')
    [ -n "$item_id" ] || item_id="pending"
  fi
  set_field "$item_id" "$url" "Plane" "$plane"
  set_field "$item_id" "$url" "Origin" "$origin"
  set_field "$item_id" "$url" "Priority" "$priority"
  echo "   $url  [$plane | ${origin:-—} | $priority]"
done <<< "$ITEMS"

echo
echo "Done. Board: ${proj_url:-https://github.com/users/$OWNER/projects/$proj_num}"
