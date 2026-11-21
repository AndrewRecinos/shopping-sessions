# Compare two output dirs (parquet tables) for exact row equality. Short report.
import sys
sys.path.insert(0, "/starter/scripts")
from run_fixtures import SCHEMAS, load_actual
ref, act = sys.argv[1:3]; ok = True
for t, sc in SCHEMAS.items():
    a = load_actual(f"{ref}/{t}", sc); b = load_actual(f"{act}/{t}", sc)
    a = a.sort(list(sc), nulls_last=True); b = b.sort(list(sc), nulls_last=True)
    if not a.equals(b):
        ok = False; oa = a.join(b, on=list(sc), how="anti", nulls_equal=True)
        print(f"{t}: rows ref {a.height} act {b.height}, missing {oa.height}")
        for r in oa.head(5).rows(): print("  -", r)
print("ref-compare:", "IDENTICAL" if ok else "DIFFERENT")
