#!/bin/bash
# Usage: ./bench.sh [data_dir=data/g20]  Builds image, runs 3x like the scorer, prints
# median wall time + stderr stage timings of the median run, then compares to out-ref.
set -u; cd "$(dirname "$0")"; ROOT=$PWD
DATA=${1:-data/g20}; NAME=$(basename "$DATA"); STARTER=$ROOT/starter-duckdb/starter-duckdb
docker build -q -t shopping-sessions:latest . >/dev/null || { echo "docker build failed"; exit 1; }
for i in 1 2 3; do
  O=$ROOT/out/bench$i; rm -rf "$O"; mkdir -p "$O"
  s=$(date +%s.%N); IMAGE=shopping-sessions:latest "$STARTER/scripts/docker_run.sh" "$DATA" "$O" >/dev/null 2>"$O.err"
  echo "$(echo "$(date +%s.%N) - $s" | bc) $i"
done | sort -n > out/bench.times
read -r T I < <(sed -n 2p out/bench.times)
echo "median wall $(printf %.3f "$T")s  (all: $(awk '{printf "%.3f ",$1}' out/bench.times))"
tr '\n' ' ' < "$ROOT/out/bench$I.err"; echo
if [ -d "out-ref/$NAME" ]; then
  docker run --rm --network none -u "$(id -u)" -v "$STARTER":/starter:ro -v "$ROOT/out":/out:ro \
    -v "$ROOT/out-ref":/ref:ro -v "$ROOT/tools":/tools:ro shop-tester \
    python -I /tools/cmpref.py "/ref/$NAME" "/out/bench$I"
fi
