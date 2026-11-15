#!/usr/bin/env bash
# Train the specialists of the portfolio model, one after another (ga/specialist.py), then
# build it:   python -m policy.portfolio build portfolio-v1 runs/<each run>
#
#   ./train_portfolio.sh              # every archetype below (skips runs that are done)
#   GENS=400 ./train_portfolio.sh witch militia
#   PREFIX=v2- SEED_FROM= HALL=runs/generalist,runs/remodel ./train_portfolio.sh
#        PREFIX     run names become runs/<PREFIX><name>
#        SEED_FROM  start from runs/<SEED_FROM><name>/strategy.json when it exists ("" = round 1 runs)
#        HALL       extra evolved opponents (ga/specialist.py --hall)
#
# One line per specialist: run name, then its key cards (empty = any kingdom).
set -euo pipefail
cd "$(dirname "$0")"
GENS=${GENS:-500}
WORKERS=${WORKERS:-6}
PREFIX=${PREFIX:-}
HALL=${HALL:-}

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
  "workshop|Workshop"
  "cellar|Cellar"
  "poacher|Poacher"
  "vassal|Vassal"
  "harbinger|Harbinger"
  "merchant|Merchant"
)

want=("$@")
for row in "${ARCHETYPES[@]}"; do
  name=${row%%|*}; must=${row#*|}
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $name "* ]]; then continue; fi
  run="$PREFIX$name"
  if [ -f "runs/$run/progress.json" ] && grep -q '"done": true' "runs/$run/progress.json"; then
    echo "== $run: done, skipped"; continue
  fi
  extra=()
  [[ $name == *-safe ]] && extra=(--exclude-blocked)
  [ -n "$HALL" ] && extra+=(--hall "$HALL")
  if [ -n "${SEED_FROM+x}" ] && [ -f "runs/$SEED_FROM$name/strategy.json" ] && [ "$SEED_FROM$name" != "$run" ]; then
    mkdir -p "runs/$run" && cp "runs/$SEED_FROM$name/strategy.json" "runs/$run/strategy.json"
    extra+=(--resume)
  fi
  echo "== $run (must: ${must:-any}) $(date +%H:%M)"
  python3 -m ga.specialist "$run" --must "$must" --gens "$GENS" --pop 64 --kingdoms 8 --seeds 8 \
    --val-every 25 --val-kingdoms 60 --workers "$WORKERS" "${extra[@]}"
done
