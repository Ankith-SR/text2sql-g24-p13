#!/usr/bin/env bash
# Creates one GitHub issue per unchecked item in docs/TASKS.md. Needs the GitHub CLI: `gh auth login` first.
# Run once from the repo root, after the repo exists on GitHub.
set -euo pipefail
owner=""
while IFS= read -r line; do
  case "$line" in
    "## "*) owner="${line#\#\# }";;
    "- [ ] "*) gh issue create --title "[$owner] ${line#- \[ \] }" --body "From docs/TASKS.md" --label task || true;;
  esac
done < docs/TASKS.md
