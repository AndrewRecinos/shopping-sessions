#!/bin/bash
# Usage: ./test.sh [local|docker]   (default local). Runs 36 fixtures + dev check.
set -u
cd "$(dirname "$0")"
MODE=${1:-local}
ROOT=$PWD
STARTER=$ROOT/starter-duckdb/starter-duckdb
FX=$STARTER/fixtures
OUT=$ROOT/out
rm -rf "$OUT"; mkdir -p "$OUT/fx"
if [ "$MODE" = docker ]; then
  docker build -q -t shopping-sessions:latest . >/dev/null || { echo "docker build failed"; exit 1; }
  run() { IMAGE=shopping-sessions:latest "$STARTER/scripts/docker_run.sh" "$1" "$2" >/dev/null 2>"$2.err"; }
else
  cargo build --release -q 2>&1 | grep -E '^error' -A5 | head -20
  run() { DATA_DIR="$1" OUTPUT_DIR="$2" ${WRAP:-} "$ROOT/target/release/app" 2>"$2.err"; }  # WRAP: e.g. "sde64 -clx --"
fi
for d in "$FX"/*/; do
  c=$(basename "$d"); data="$d/input"; [ -d "$d/data" ] && data="$d/data"
  mkdir -p "$OUT/fx/$c"; run "$data" "$OUT/fx/$c" || echo "FAIL $c: exit $? $(tail -c 300 "$OUT/fx/$c.err")"
done
DEV=-
if [ -d data/dev ]; then
  mkdir -p "$OUT/dev"; s=$(date +%s.%N)
  run "$ROOT/data/dev" "$OUT/dev" || echo "dev run failed: $(tail -c 300 "$OUT/dev.err")"
  echo "dev run time: $(echo "$(date +%s.%N) - $s" | bc | cut -c1-5)s"
  DEV=/out/dev
fi
docker run --rm --network none -u "$(id -u)" -v "$STARTER":/starter:ro -v "$OUT":/out:ro \
  -v "$ROOT/dev-answer/dev-answer":/ans:ro -v "$ROOT/tools":/tools:ro shop-tester \
  python -I /tools/cmp.py /starter/fixtures /out/fx "$DEV" /ans
