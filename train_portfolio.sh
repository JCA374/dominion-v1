#!/usr/bin/env bash
# Train the specialists of the portfolio model, one after another (ga/specialist.py), then
# build it:   python -m policy.portfolio build portfolio-v1 runs/<each run>
#
#   ./train_portfolio.sh              # every archetype below (skips runs that are done)
#   GENS=400 ./train_portfolio.sh witch militia
#
# One line per specialist: run name, then its key cards (empty = any kingdom).
set -euo pipefail
cd "$(dirname "$0")"
GENS=${GENS:-500}
WORKERS=${WORKERS:-6}

ARCHETYPES=(
  "generalist|"
  "witch|Witch"
  "militia|Militia"
  "smithy|Smithy"
  "laboratory|Laboratory"
  "council-room|Council Room"
  "library|Library"
  "chapel|Chapel"
  "market|Market"
  "festival|Festival"
  "bandit|Bandit"
  "gardens|Gardens,Workshop"
  "moneylender|Moneylender"
  "sentry|Sentry"
  "artisan|Artisan"
  "mine|Mine"
  "remodel|Remodel"
  "village-smithy|Village,Smithy"
  "throne-room|Throne Room"
  "bureaucrat|Bureaucrat"
  "generalist-safe|"
)

want=("$@")
for row in "${ARCHETYPES[@]}"; do
  name=${row%%|*}; must=${row#*|}
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $name "* ]]; then continue; fi
  if [ -f "runs/$name/progress.json" ] && grep -q '"done": true' "runs/$name/progress.json"; then
    echo "== $name: done, skipped"; continue
  fi
  extra=()
  [[ $name == *-safe ]] && extra=(--exclude-blocked)
  echo "== $name (must: ${must:-any}) $(date +%H:%M)"
  python3 -m ga.specialist "$name" --must "$must" --gens "$GENS" --pop 64 --kingdoms 8 --seeds 8 \
    --val-every 25 --val-kingdoms 60 --workers "$WORKERS" "${extra[@]}"
done
