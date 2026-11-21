#!/bin/bash
# Usage: ./cmp.sh <out dir> [ref name=g20]  Compare a local output dir to out-ref/<name>.
cd "$(dirname "$0")"; ROOT=$PWD; O=$(cd "$1" && pwd)
docker run --rm --network none -u "$(id -u)" -v "$ROOT/starter-duckdb/starter-duckdb":/starter:ro -v "$O":/out:ro \
  -v "$ROOT/out-ref":/ref:ro -v "$ROOT/tools":/tools:ro shop-tester python -I /tools/cmpref.py "/ref/${2:-g20}" /out
