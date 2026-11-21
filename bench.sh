#!/bin/bash
# Usage: ./bench.sh [data_dir=data/g20] [runs=5]  Builds image, runs like the scorer
# (8 logical CPUs PINNED = 4 physical cores + their HT siblings, 24g), prints median wall
# time + stderr stage timings of the median run, then compares to out-ref.
set -u; cd "$(dirname "$0")"; ROOT=$PWD
DATA=$(cd "${1:-data/g20}" && pwd); NAME=$(basename "$DATA"); RUNS=${2:-5}; STARTER=$ROOT/starter-duckdb/starter-duckdb
# first 4 physical cores (skip core 0: OS noise) and their siblings
CPUSET=$(for c in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do cat "$c"; done \
  | sort -u -t- -k1,1n | sed -n 2,5p | paste -sd,)
docker build -q -t shopping-sessions:latest . >/dev/null || { echo "docker build failed"; exit 1; }
for i in $(seq "$RUNS"); do
  O=$ROOT/out/bench$i; rm -rf "$O" "$O.err"; mkdir -p "$O"; chmod 777 "$O"
  s=$(date +%s.%N)
  docker run --rm --network none --read-only --tmpfs /tmp:rw,size=64m,mode=1777 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
    --cpuset-cpus "$CPUSET" --memory 24g --memory-swap 24g --user 1000:1000 \
    -v "$DATA":/data:ro -v "$O":/output -e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/tmp \
    -e NCPU=8 -e MEM_GB=24 shopping-sessions:latest >/dev/null 2>"$O.err"
  echo "$(echo "$(date +%s.%N) - $s" | bc) $i"
done | sort -n > out/bench.times
read -r T I < <(sed -n "$(( (RUNS+1)/2 ))p" out/bench.times)
echo "cpuset $CPUSET  median wall $(printf %.3f "$T")s  (all: $(awk '{printf "%.3f ",$1}' out/bench.times))"
echo "internal total median $(cat out/bench*.err | awk '/^total/{print $2}' | sort -n | sed -n "$(( (RUNS+1)/2 ))p")"
tr '\n' ' ' < "$ROOT/out/bench$I.err"; echo
if [ -d "out-ref/$NAME" ]; then
  docker run --rm --network none -u "$(id -u)" -v "$STARTER":/starter:ro -v "$ROOT/out":/out:ro \
    -v "$ROOT/out-ref":/ref:ro -v "$ROOT/tools":/tools:ro shop-tester \
    python -I /tools/cmpref.py "/ref/$NAME" "/out/bench$I"
fi
