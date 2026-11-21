#!/usr/bin/env bash
# Continue v2 specialists from their released genomes (the "is it undertrained?" experiment).
#   PREFIX=c1- GENS=1000 ARGS="--rate 0.18" experiments/continue_runs.sh witch mine ...
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX=${PREFIX:-c1-}; GENS=${GENS:-1000}; WORKERS=${WORKERS:-7}; ARGS=${ARGS:-}
declare -A MUST=([generalist]="" [witch]=Witch [mine]=Mine [workshop]=Workshop [vassal]=Vassal [smithy]=Smithy
  [merchant]=Merchant [laboratory]=Laboratory [chapel]=Chapel [poacher]=Poacher [market]=Market [militia]=Militia [gardens]="Gardens,Workshop" [bandit]=Bandit [cellar]=Cellar [harbinger]=Harbinger
  [artisan]=Artisan [bureaucrat]=Bureaucrat [council-room]="Council Room" [festival]=Festival [library]=Library
  [moneylender]=Moneylender [remodel]=Remodel [sentry]=Sentry [throne-room]="Throne Room" [village-smithy]="Village,Smithy")
for name in "$@"; do
  run="$PREFIX$name"
  mkdir -p "runs/$run"
  cp "models/portfolio-v2/members/v2-$name.json" "runs/$run/strategy.json"
  echo "== $run $(date +%H:%M)"
  python3 -u -m ga.specialist "$run" --must "${MUST[$name]}" --gens "$GENS" --pop 64 --kingdoms 8 --seeds 8 \
    --val-every 25 --val-kingdoms 60 --workers "$WORKERS" --seed 2 --resume --snapshots \
    --hall runs/generalist,runs/remodel,runs/witch $ARGS
done
