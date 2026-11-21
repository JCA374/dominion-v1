#!/usr/bin/env bash
# Measure each continued specialist (runs/c1-NAME) held-out once its run has finished (1000 gens in its progress),
# against its v2 genome: experiments/measure_specialist.py, 200 kingdoms x 20 seeds -> runs/meas/NAME.json.
#   experiments/measure_queue.sh harbinger artisan ...
set -uo pipefail
cd "$(dirname "$0")/.."
declare -A MUST=([harbinger]=Harbinger [artisan]=Artisan [bureaucrat]=Bureaucrat [chapel]=Chapel [council-room]="Council Room"
  [festival]=Festival [generalist]="" [laboratory]=Laboratory [library]=Library [moneylender]=Moneylender [remodel]=Remodel
  [sentry]=Sentry [smithy]=Smithy [throne-room]="Throne Room" [village-smithy]="Village,Smithy")
for name in "$@"; do
  until grep -q "\[c1-$name\] gen 1000 " runs-c23.log runs-c4.log 2>/dev/null; do sleep 60; done
  [ -f "runs/meas/$name.json" ] && continue
  echo "== $name $(date +%H:%M)"
  nice python3 experiments/measure_specialist.py --must "${MUST[$name]}" --ref "models/portfolio-v2/members/v2-$name.json" \
    "models/portfolio-v2/members/v2-$name.json" "runs/c1-$name/strategy.json" --kingdoms 200 --seeds 20 --workers 1 \
    --json "runs/meas/$name.json" | tail -2
done
