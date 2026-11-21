# Compare fixture outputs and dev output; short report. Runs inside shop-tester.
import os, sys
sys.path.insert(0, "/starter/scripts")
from run_fixtures import SCHEMAS, load_expected, load_actual, rows
import polars as pl
fx, fxout, devout, ans = sys.argv[1:5]
fails = 0; cases = sorted(d for d in os.listdir(fx) if os.path.isdir(os.path.join(fx, d)))
shown = 0
for c in cases:
    bad = []
    for t, sc in SCHEMAS.items():
        try:
            exp = rows(load_expected(f"{fx}/{c}/expected/{t}.csv", sc))
            act = rows(load_actual(f"{fxout}/{c}/{t}", sc))
        except Exception as e:
            bad.append(f"{t}: {str(e)[:200]}"); continue
        if exp != act:
            se, sa = set(exp), set(act)
            d = [f"-{r}" for r in exp if r not in sa][:5] + [f"+{r}" for r in act if r not in se][:5]
            bad.append(f"{t}: " + "; ".join(d[:5]))
    if bad:
        fails += 1
        if shown < 5:
            shown += 1; print(f"FAIL {c}: " + " | ".join(bad))
print(f"fixtures: {len(cases)-fails}/{len(cases)} passed")
if devout != "-":
    ok = True
    for t, sc in SCHEMAS.items():
        a = load_expected(f"{ans}/{t}.csv", sc)
        try:
            b = load_actual(f"{devout}/{t}", sc)
        except Exception as e:
            print(f"dev {t}: {e}"); ok = False; continue
        keys = [k for k in ("user_id", "campaign_id", "day") if k in sc]
        oa = a.join(b, on=list(sc), how="anti", nulls_equal=True)
        ob = b.join(a, on=list(sc), how="anti", nulls_equal=True)
        dup = b.height - b.n_unique(subset=keys)
        if oa.height or ob.height or dup:
            ok = False
            print(f"dev {t}: rows exp {a.height} act {b.height}, missing {oa.height}, extra {ob.height}, dupkeys {dup}")
            for r in oa.sort(keys).head(5).rows(): print("  -", r)
            for r in ob.sort(keys).head(5).rows(): print("  +", r)
    print("dev: " + ("1/1 passed" if ok else "0/1 passed"))
