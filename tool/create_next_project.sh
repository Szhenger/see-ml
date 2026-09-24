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
#   * its IDENTITY: `number:` in the front matter names the live issue and
#     is the match key (a retitled issue still matches, and the file's title
#     is then pushed as an edit); a body without one is matched by exact
#     title, and the number the create returns is written back into the
#     file, so the second run matches by number. Two live issues with one
#     title never abort a run: the open, lowest-numbered one wins;
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
PROJECT_TITLE="SeeAI Two-Plane Overhaul"
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
# Any other label a body names is created grey, only when it is missing.
LABEL_TABLE="
python-plane|3572A5|Build-host Python subsystem (the frontier plane)
core-plane|0E4D64|Deterministic C++ core (device plane)
doctrine|B34A2E|Touches a doctrinal guarantee; product decision required
frontier-parity|5319E7|Parity with the training frontier (torch.compile, MLX-LM): SeeAI F1-F9
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
# A paginated read as ONE array (gh api --paginate emits one array per
# page; a test that reads them page by page sees only the last one).
gh_pages() { gh api --paginate "$1" | jq -s 'add // []'; }
# A body reaches gh through a file, never through a pipe into mutate: a
# pipe into a command that does not read it (a dry run, a labels-only
# edit) is a SIGPIPE race that aborts the script under pipefail.
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/next-project.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT

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
for f in "${BODIES[@]}"; do  # fail loudly, before any write, on a body that is not one
  for k in title labels plane priority; do
    [ -n "$(fm "$f" "$k")" ] || { echo "error: $(basename "$f") has no '$k:' in its front matter" >&2; exit 1; }
  done
done

# --- labels: every label any body names ----------------------------------
echo "== labels"
# GitHub label names are case-insensitive: every comparison lower-cases both sides.
have_labels=$(gh label list -R "$REPO" --limit 500 --json name,color,description |
  jq -c 'map({name: (.name | ascii_downcase), color: (.color | ascii_downcase), description: (.description // "")})')
wanted_labels=$(for f in "${BODIES[@]}"; do fm "$f" labels; done | paste -sd, - | list_json)
jq -r '.[]' <<< "$wanted_labels" | while IFS= read -r l; do
  live=$(jq -r --arg l "$l" 'first(.[] | select(.name == ($l | ascii_downcase)) | "\(.color)|\(.description)") // empty' <<< "$have_labels")
  row=$(awk -F'|' -v l="$l" '$1 == l' <<< "$LABEL_TABLE")
  if [ -n "$row" ]; then
    color=$(cut -d'|' -f2 <<< "$row"); desc=$(cut -d'|' -f3 <<< "$row")
    [ "$live" = "$(tr '[:upper:]' '[:lower:]' <<< "$color")|$desc" ] ||
      mutate "upsert label '$l' (#$color, '$desc')" label create "$l" -R "$REPO" --force -c "$color" -d "$desc" >/dev/null
  elif [ -z "$live" ]; then
    mutate "create label '$l' (grey)" label create "$l" -R "$REPO" -c EDEDED -d "" >/dev/null
  fi
done
echo "   ok: $(jq -r 'join(" ")' <<< "$wanted_labels")"

# --- milestones: every milestone any body names --------------------------
echo "== milestones"
have_ms=$(gh_pages "repos/$REPO/milestones?state=all&per_page=100" | jq -c '[.[].title]')
for f in "${BODIES[@]}"; do fm "$f" milestone; done | sed '/^$/d' | LC_ALL=C sort -u |
while IFS= read -r m; do
  if jq -e --arg m "$m" 'index($m) != null' <<< "$have_ms" >/dev/null; then echo "   ok: $m"
  else mutate "create milestone '$m'" api "repos/$REPO/milestones" -f title="$m" >/dev/null; fi
done

# --- issues: one fetch, then create or edit per body ---------------------
echo "== issues"
LIVE=$(gh_pages "repos/$REPO/issues?state=all&per_page=100" |
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
  number=$(fm "$f" number)
  body "$f" > "$TMPD/body"
  want_body=$(norm < "$TMPD/body")

  # Identity: `number:` when the file names one, else the exact title. A
  # title held by several live issues (the retitling hazard the 2026-09-24
  # review named) resolves to the open, lowest-numbered one; nothing is
  # piped into head, so no SIGPIPE under pipefail.
  if [ -n "$number" ]; then
    live=$(jq -c -s --argjson n "$number" 'first(.[] | select(.number == $n)) // empty' <<< "$LIVE")
    [ -n "$live" ] || { echo "error: $(basename "$f") names number: $number, which is not an issue of $REPO" >&2; exit 1; }
  else
    live=$(jq -c -s --arg t "$title" '[.[] | select(.title == $t)] | sort_by((.state != "open"), .number) | first // empty' <<< "$LIVE")
  fi
  if [ -z "$live" ]; then
    url=$(mutate "create issue '$title' [labels: ${labels_csv:-none}; milestone: ${milestone:-none}]" \
      issue create -R "$REPO" -t "$title" -F "$TMPD/body" \
      ${labels_csv:+-l "$labels_csv"} ${milestone:+-m "$milestone"})
    [ -n "$url" ] || url="https://github.com/$REPO/issues/new?title=$(basename "$f")"
    if [ "$DRY_RUN" != 1 ]; then
      num="${url##*/}"
      # The file now owns its identity: insert `number:` after `title:` so
      # the next run matches by number even if the issue is retitled.
      awk -v n="$num" 'p == 0 && /^title: / { print; print "number: " n; p = 1; next } { print }' "$f" > "$TMPD/fm" && cat "$TMPD/fm" > "$f"
      echo "   created: $url  ($(basename "$f"); wrote number: $num into the file)"
    fi
  else
    num=$(jq -r .number <<< "$live")
    url="https://github.com/$REPO/issues/$num"
    if [ "$(jq -r .state <<< "$live")" = "closed" ]; then
      echo "   closed, left alone (#$num): $title"
    else
      live_body=$(jq -r .body <<< "$live" | norm)
      add_labels=$(jq -r --argjson want "$labels_json" \
        '[.labels[] | ascii_downcase] as $have | [$want[] | . as $w | select(($have | index($w | ascii_downcase)) == null)] | join(",")' <<< "$live")
      live_ms=$(jq -r .milestone <<< "$live")
      # Each change sets its own flag here; nothing is re-parsed from the
      # printed list (a label named "milestone-blocker" is just a label).
      what=""; body_flag=""; ms_flag=""; title_flag=""
      [ "$live_body" = "$want_body" ] || { what="$what body"; body_flag=1; }
      [ "$(jq -r .title <<< "$live")" = "$title" ] || { what="$what title"; title_flag=1; }
      [ -z "$add_labels" ] || what="$what labels(+$add_labels)"
      if [ -n "$milestone" ] && [ "$live_ms" != "$milestone" ]; then what="$what milestone"; ms_flag=1; fi
      what="${what# }"
      if [ -z "$what" ]; then
        echo "   in sync (#$num): $title"
      else
        # bash 3.2 + set -u refuses "${empty[@]}": each flag is passed
        # through a guarded expansion instead of an array.
        mutate "edit #$num [$what]" issue edit "$num" -R "$REPO" \
          ${body_flag:+-F "$TMPD/body"} \
          ${title_flag:+-t "$title"} \
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
else
  proj_json=$(mutate "create project '$PROJECT_TITLE'" project create --owner "$OWNER" --title "$PROJECT_TITLE" --format json)
  if [ -z "$proj_json" ]; then
    echo "   (nothing to reconcile the board against yet; stopping)" >&2
    exit 0
  fi
fi
proj_num=$(jq -r '.number' <<< "$proj_json")
proj_id=$(jq -r '.id' <<< "$proj_json")
proj_url=$(jq -r '.url // empty' <<< "$proj_json")
echo "   project #$proj_num ${proj_url:+at $proj_url}"

fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)
WARNED=""
PENDING_FIELDS=""  # dry run only: fields it would have created, whose values it cannot preview

ensure_field() {  # ensure_field <name> <comma-separated options>
  local have missing
  have=$(jq -r --arg n "$1" 'first(.fields[] | select(.name == $n) | .id) // empty' <<< "$fields_json")
  if [ -z "$have" ]; then
    mutate "create field '$1' with options [$2]" project field-create "$proj_num" --owner "$OWNER" \
      --name "$1" --data-type SINGLE_SELECT --single-select-options "$2" >/dev/null
    [ "$DRY_RUN" = 1 ] && PENDING_FIELDS="$PENDING_FIELDS $1 "
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

[ "$DRY_RUN" = 1 ] || fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)  # sees the fields just created
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
  if [ -z "$fid" ] || [ -z "$oid" ]; then  # said once per missing option; the ACTION line above names the fix
    case "$WARNED" in *"|$3=$4|"*) ;; *) WARNED="$WARNED|$3=$4|"; echo "   warn: '$4' not set on any item: field '$3' lacks it (see ACTION above)" >&2 ;; esac
    return 0
  fi
  # item-list keys a field's value by its lower-cased name.
  key=$(tr '[:upper:]' '[:lower:]' <<< "$3")
  current=$(jq -r --arg u "$2" --arg k "$key" --arg n "$3" \
    'first(.items[] | select(.content.url == $u) | (.[$k] // .[$n] // "")) // ""' <<< "$items_json")
  [ "$current" = "$4" ] && return 0
  mutate "set $3 = '$4' on $2 (was '${current:-unset}')" project item-edit --id "$1" \
    --project-id "$proj_id" --field-id "$fid" --single-select-option-id "$oid" >/dev/null
}

echo "== items"
while IFS='|' read -r url plane origin priority; do
  [ -n "$url" ] || continue
  item_id=$(jq -r --arg u "$url" 'first(.items[] | select(.content.url == $u) | .id) // empty' <<< "$items_json")
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
