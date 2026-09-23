#!/usr/bin/env bash
# Reconcile the Two-Plane Overhaul with GitHub: the bodies in
# docs/next-project/issues/ are the source of truth for one Issue each
# (title, labels, milestone, body), plus a Project (v2) board carrying the
# new issues and the existing roadmap issues, with Plane / Origin /
# Priority single-select fields set per item.
#
# Prereqs: gh (authenticated) + jq. Projects need the extra scope once:
#     gh auth refresh -s project
#
# Idempotent and convergent: every label a body names is upserted, every
# milestone a body names is upserted, an issue is created when no issue
# carries its exact title and EDITED when the live body, labels or
# milestone differ from the file (so a committed edit to a body reaches
# GitHub on the next run), the project is reused if the title matches,
# existing fields / options are reused, and item-add is a no-op for
# present items. The issue list is fetched once, paginated, so the
# title match does not silently stop at a page boundary.
#
# A single-select option that an existing field lacks cannot be added by
# the Projects CLI (the option-edit mutation rejects it): the script says
# which option to add in the web UI and carries on.
#
#     tool/create_next_project.sh            # reconcile
#     tool/create_next_project.sh --dry-run  # say what would change, touch nothing
set -euo pipefail

REPO="Szhenger/see-ml"
OWNER="Szhenger"
PROJECT_TITLE="SeeML Two-Plane Overhaul"
ISSUE_DIR="$(cd "$(dirname "$0")/../docs/next-project/issues" && pwd)"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

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

run() {  # run <cmd...>: execute, or print under --dry-run
  if [ "$DRY_RUN" = 1 ]; then echo "   would: $*"; else "$@"; fi
}

# --- front-matter helpers -------------------------------------------------
fm() {  # fm <file> <key>  (empty when the key is absent)
  awk -v key="$2" '
    /^---$/ { c++; next }
    c == 1 && $0 ~ "^" key ": " {
      sub("^" key ": ", ""); gsub(/^"|"$/, ""); print; exit
    }' "$1"
}
body() { awk '/^---$/ { c++; next } c >= 2 { print }' "$1"; }
# GitHub stores CRLF-free text and drops a trailing newline: compare the
# same normalization on both sides.
norm() { tr -d '\r' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}'; }
sorted_csv() { tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -; }

BODIES=("$ISSUE_DIR"/*.md)

# --- labels: every label any body names ----------------------------------
echo "== labels"
have_labels=$(gh label list -R "$REPO" --limit 500 --json name | jq -r '.[].name')
wanted=$(for f in "${BODIES[@]}"; do fm "$f" labels; done | sorted_csv | tr ',' '\n')
while IFS= read -r l; do
  [ -n "$l" ] || continue
  row=$(grep -F "$l|" <<< "$LABEL_TABLE" | grep "^$l|" || true)
  if [ -n "$row" ]; then
    color=$(cut -d'|' -f2 <<< "$row"); desc=$(cut -d'|' -f3 <<< "$row")
    run gh label create "$l" -R "$REPO" --force -c "$color" -d "$desc" >/dev/null
  elif ! grep -qx "$l" <<< "$have_labels"; then
    run gh label create "$l" -R "$REPO" -c EDEDED -d "" >/dev/null
  fi
done <<< "$wanted"
echo "   ok: $(tr '\n' ' ' <<< "$wanted")"

# --- milestones: every milestone any body names --------------------------
echo "== milestones"
have_ms=$(gh api --paginate "repos/$REPO/milestones?state=all&per_page=100" | jq -r '.[].title')
for f in "${BODIES[@]}"; do fm "$f" milestone; done | sed '/^$/d' | sort -u |
while IFS= read -r m; do
  if grep -qxF "$m" <<< "$have_ms"; then echo "   ok: $m"
  else run gh api "repos/$REPO/milestones" -f title="$m" >/dev/null; echo "   $([ "$DRY_RUN" = 1 ] && echo would create || echo created): $m"; fi
done

# --- issues: one fetch, then create or edit per body ---------------------
echo "== issues"
LIVE=$(gh api --paginate "repos/$REPO/issues?state=all&per_page=100" |
  jq -c '.[] | select(.pull_request == null) |
         {number, title, body: (.body // ""), milestone: (.milestone.title // ""),
          labels: ([.labels[].name] | sort | join(","))}')
ITEMS=""  # url|Plane|Origin|Priority
for f in "${BODIES[@]}"; do
  title=$(fm "$f" title)
  labels=$(fm "$f" labels | sorted_csv)
  plane=$(fm "$f" plane)
  origin=$(fm "$f" origin)
  priority=$(fm "$f" priority)
  milestone=$(fm "$f" milestone)
  want_body=$(body "$f" | norm)

  live=$(jq -c --arg t "$title" 'select(.title == $t)' <<< "$LIVE" | head -1)
  if [ -z "$live" ]; then
    label_args=()
    IFS=',' read -ra ls <<< "$labels"
    for l in "${ls[@]}"; do label_args+=(-l "$l"); done
    ms_args=(); [ -n "$milestone" ] && ms_args=(-m "$milestone")
    if [ "$DRY_RUN" = 1 ]; then
      url="https://github.com/$REPO/issues/new"
      echo "   would create: $title  ($(basename "$f"))"
    else
      url=$(body "$f" | gh issue create -R "$REPO" -t "$title" -F - "${label_args[@]}" ${ms_args[@]+"${ms_args[@]}"})
      echo "   created: $url  ($(basename "$f"))"
    fi
  else
    num=$(jq -r .number <<< "$live")
    url="https://github.com/$REPO/issues/$num"
    live_body=$(jq -r .body <<< "$live" | norm)
    live_labels=$(jq -r .labels <<< "$live")
    live_ms=$(jq -r .milestone <<< "$live")
    changes=()  # bash 3.2 + set -u: an empty array is "unset", hence the guarded expansions below
    [ "$live_body" = "$want_body" ] || changes+=(body)
    [ "$live_labels" = "$labels" ] || changes+=(labels)
    [ "$live_ms" = "$milestone" ] || changes+=(milestone)
    if [ "${changes[*]+${#changes[@]}}" = "" ]; then
      echo "   in sync (#$num): $title"
    else
      edit_args=()
      for c in "${changes[@]}"; do
        case "$c" in
          labels)
            add=$(comm -13 <(tr ',' '\n' <<< "$live_labels" | sed '/^$/d') <(tr ',' '\n' <<< "$labels") | paste -sd, -)
            rm_=$(comm -23 <(tr ',' '\n' <<< "$live_labels" | sed '/^$/d') <(tr ',' '\n' <<< "$labels") | paste -sd, -)
            [ -n "$add" ] && edit_args+=(--add-label "$add")
            [ -n "$rm_" ] && edit_args+=(--remove-label "$rm_") ;;
          milestone)
            if [ -n "$milestone" ]; then edit_args+=(-m "$milestone"); else edit_args+=(--remove-milestone); fi ;;
          body) edit_args+=(-F -) ;;
        esac
      done
      if [ "$DRY_RUN" = 1 ]; then
        echo "   would edit #$num (${changes[*]}): $title"
      else
        body "$f" | gh issue edit "$num" -R "$REPO" ${edit_args[@]+"${edit_args[@]}"} >/dev/null
        echo "   edited #$num (${changes[*]}): $title"
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
  echo "   would create project '$PROJECT_TITLE'; stopping here (no project to reconcile against)"
  exit 0
else
  proj_json=$(gh project create --owner "$OWNER" --title "$PROJECT_TITLE" --format json)
fi
proj_num=$(jq -r '.number' <<< "$proj_json")
proj_id=$(jq -r '.id' <<< "$proj_json")
proj_url=$(jq -r '.url // empty' <<< "$proj_json")
echo "   project #$proj_num ${proj_url:+at $proj_url}"

fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)

ensure_field() {  # ensure_field <name> <comma-separated options>
  local have missing
  have=$(jq -r --arg n "$1" '.fields[] | select(.name == $n) | .id' <<< "$fields_json" | head -1)
  if [ -z "$have" ]; then
    run gh project field-create "$proj_num" --owner "$OWNER" --name "$1" \
      --data-type SINGLE_SELECT --single-select-options "$2" >/dev/null
    echo "   field created: $1 ($2)"
    return 0
  fi
  # The field exists: an option it lacks has to be added by hand (the
  # Projects option-edit mutation refuses to append). Say which.
  missing=$(comm -23 <(tr ',' '\n' <<< "$2" | sort -u) \
    <(jq -r --arg n "$1" '.fields[] | select(.name == $n) | .options[].name' <<< "$fields_json" | sort -u) |
    paste -sd, -)
  [ -z "$missing" ] || echo "   ACTION: field '$1' lacks option(s) '$missing' — add in the web UI (project settings → $1), then rerun" >&2
}
# The option sets are the union of what the bodies and the roadmap rows
# name, so a new origin or plane in a body is never silently unset.
opts() {  # opts <front-matter key> <EXISTING_ITEMS column>
  { for f in "${BODIES[@]}"; do fm "$f" "$1"; done
    sed '/^$/d' <<< "$EXISTING_ITEMS" | cut -d'|' -f"$2"; } | sed '/^$/d' | sort -u | paste -sd, -
}
ensure_field "Plane"    "$(opts plane 2)"
ensure_field "Origin"   "$(opts origin 3)"
ensure_field "Priority" "P0,P1,P2"

fields_json=$(gh project field-list "$proj_num" --owner "$OWNER" --format json)

set_field() {  # set_field <item-id> <field-name> <option-name>
  local fid oid
  [ -n "$3" ] || return 0
  fid=$(jq -r --arg n "$2" '.fields[] | select(.name == $n) | .id' <<< "$fields_json")
  oid=$(jq -r --arg n "$2" --arg o "$3" \
    '.fields[] | select(.name == $n) | .options[] | select(.name == $o) | .id' \
    <<< "$fields_json")
  if [ -z "$fid" ] || [ -z "$oid" ]; then
    echo "   warn: no option '$3' for field '$2' (see ACTION above)" >&2; return 0
  fi
  run gh project item-edit --id "$1" --project-id "$proj_id" \
    --field-id "$fid" --single-select-option-id "$oid" >/dev/null
}

echo "== items"
while IFS='|' read -r url plane origin priority; do
  [ -n "$url" ] || continue
  if [ "$DRY_RUN" = 1 ]; then item_id="dry"
  else
    item_id=$(gh project item-add "$proj_num" --owner "$OWNER" --url "$url" \
      --format json | jq -r '.id')
  fi
  set_field "$item_id" "Plane" "$plane"
  set_field "$item_id" "Origin" "$origin"
  set_field "$item_id" "Priority" "$priority"
  echo "   $url  [$plane | ${origin:-—} | $priority]"
done <<< "$ITEMS"

echo
echo "Done. Board: ${proj_url:-https://github.com/users/$OWNER/projects/$proj_num}"
