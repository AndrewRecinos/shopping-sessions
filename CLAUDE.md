# Shopping Sessions — Builders Buildathon (Databricks track)

Goal: **win on run time**. Score = container wall time (start → exit) on ~20M events
(practice) / ~60M events (final, 3 min hard cap per run, any overrun = no score).
Current leader: 0.93 s. A wrong row = score 0, so **correctness first, then speed**.

`rules.md` in this folder is the spec and always wins. Read it fully before coding.
The starter in `starter-duckdb/` has the 36 edge-case fixtures, test scripts and
`make check-dev`; read its Makefile/scripts to learn how tests are run and where
outputs go. `generator/` is the Rust source of the data generator; `dev-answer/` has
the expected CSVs for the 1M dev set (seed 1).

## Working rules (to save usage)

- Keep test/bench output SHORT: print pass/fail counts and at most the first 5
  differing rows. Never dump whole files or long logs into the conversation.
- Log timings to stderr as one line per stage (`parse_events 0.21s`).
- Do big chunks of work per step: implement → run fixtures → fix → repeat, without
  asking me between iterations unless truly blocked.
- Don't add features beyond rules.md. No config, no CLI flags beyond what's needed.

## Deliverable

- A single Rust binary (crate at repo root) + `Dockerfile` at repo root.
- Reads `DATA_DIR` (default /data), writes `OUTPUT_DIR` (default /output),
  may spill to `SCRATCH_DIR`. 8 vCPU, 24 GB RAM, no swap, no network at run time.
- Root FS read-only except /tmp (64 MB). Runs as uid 1000. Log to stderr only.
- Image linux/amd64, ≤ 8 GB. Repo ≤ 50 MB, **no data committed** (.gitignore it).

## Dockerfile plan (fast startup)

- Multi-stage: build in `rust:1` with target `x86_64-unknown-linux-musl`,
  `--release`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`.
- Use `mimalloc` as global allocator (musl's malloc is slow).
- Final stage `FROM scratch`, copy only the static binary, `USER 1000`,
  `ENTRYPOINT ["/app"]`.
- **Do NOT use `-C target-cpu=native` or x86-64-v3**: the build machine may differ
  from the run machine and SIGILL = score 0. Use default target; `memchr` does
  runtime SIMD dispatch already.
- If `FROM scratch` causes any problem in the starter's test harness, fall back to
  `gcr.io/distroless/static`.

## Architecture

All files are memory-mapped (`memmap2`) and IDs are kept as `&[u8]` slices into the
maps (zero-copy; quotes stripped by adjusting the slice). Never allocate a String
per row. Use `std::thread::scope` with 8 threads (or rayon). Split each big file
into ~line-aligned chunks so work is balanced (files have Zipf sizes; one holds
~20% of rows). Header line = first line of each file, skip it. Last line may lack
a trailing `\n`. Some files are header-only.

Time base: `base = window_start - 7 days` (ms). All valid ts (events and clicks)
fit in `u32` as `ts - base` (21 days < 2^32 ms). Use that for compact sort keys.

### 0. Small inputs
- `manifest.json`: parse window_start/window_end (F1 strings) by hand.
- `products.csv` → map product_id bytes → category index (u8). Category names list
  + `unknown` (index = len). Pre-sort category names byte-wise so "smallest
  category" tie-break = smallest index.
- `campaigns.csv` → map campaign_id bytes → (campaign idx u16, channel).

### 1. Links → persons (union-find)
- Parse links (2 fields exactly). Drop malformed lines and lines with an empty side.
- Intern every ID → u32. Union-find with path compression + union by size.
- Person name = byte-wise smallest ID string in the component (track min per root).
- IDs seen only in events/clicks (not in links) are their own person.

### 2. Events
Per copy, parse and compute everything that only depends on that copy:
`(event_key, ingest_ts, ok, person, ts_rel u32, type u8, cat u8, price u32)`
where `ok` = passes all validation (user non-empty, ts valid & in
[window_start, window_end), type exactly view/cart/purchase, product_id non-empty,
purchase price digits-only in 1..=10_000_000) AND not late
(`ingest_ts - ts <= 86_400_000`; ingest before ts is fine).
- Malformed lines and empty event_id → drop before dedup.
- Dedup on event_id: keep copy with smallest ingest_ts (equal ingest_ts copies are
  identical). Then keep it only if `ok`. A later valid copy NEVER rescues an
  invalid earliest copy. Copies may name different users — use the kept copy's.
- Dedup is 2–4%: do it by partitioning copies by hash(event_id) into 8×k buckets,
  then per bucket sort/hash and resolve. Compare real bytes, not just hashes.

### 3. Clicks
- Parse; drop malformed / empty click_id; validate (user non-empty, ts valid & in
  [window_start - 7d, window_end), campaign_id exactly in campaigns.csv).
- **Click dedup can be skipped**: copies are identical in value, attribution picks
  latest ts then smallest click_id (identical copies give the same winner), and
  assists count distinct campaigns. Verify this reasoning against rules.md.
- Keep per click: person, ts_rel, campaign idx, and a handle to the click_id bytes
  (needed only for tie-break on equal ts; byte-wise compare).

### 4. Sessionize + attribute (per person)
- Group events and clicks by person (radix/partition by person into buckets, one
  bucket set per thread), sort each person's events by ts and clicks by
  (ts, click_id bytes).
- New session when `ts - prev_ts > 1_800_000` (strictly greater; equal ts share).
- Session day = UTC date of the session start; ALL its events count on that day.
- Credit: latest click with `start - 7d <= click.ts <= start`; tie on ts →
  smallest click_id byte-wise. None → `organic`.
- Assists: every distinct campaign with a click in that same window (including the
  credited one). Use a two-pointer sliding window over the person's sorted clicks
  (session starts are increasing) with a per-thread campaign count array.
  `organic` never assists.

### 5. Aggregate + write
- user_daily per (person, day): sessions, events, revenue_cents, top_category
  (max summed purchase price per category; tie → byte-wise smallest name; null if
  no purchases; `unknown` competes).
- campaign_daily per (campaign, day): sessions, purchasing_sessions,
  revenue_cents, assisted_sessions. Dense arrays [campaigns+1][days] per thread,
  merged at the end. Include rows where only assisted > 0 (credited cols = 0).
  organic: channel `organic`, assisted 0.
- Write Parquet with the `parquet` crate (low-level writer is fine, arrow optional).
  Columns exactly as rules.md; `day` = DATE (INT32 logical Date); integers as
  INT64 is fine; strings BYTE_ARRAY UTF8. **No compression.** Each thread can write
  its own `part-N.parquet` for user_daily. File names must not start with `.`/`_`.
- After writing, flush and `std::process::exit(0)` — skip freeing memory.

## Correctness traps (all from rules.md — test each)

CSV:
- Only `\n` ends a line. `\r` is DATA (so `1000\r` price is bad, `...Z\r` ts bad).
- Exact field count (events 7, clicks 4, links 2) else malformed. Empty line =
  malformed.
- Quotes only allowed wrapping a WHOLE field: `"u1"` → `u1`, `""` → empty. A quote
  anywhere else (`a"b`, `"abc`, `abc"`, `"a"b"`) → whole line malformed.
- Spaces are significant (` u1`, `u1 `, `u1` are three IDs). Case-sensitive.
- Digits = ASCII 0-9 only.

Timestamps (events + clicks `ts`):
- F1 `YYYY-MM-DDTHH:MM:SSZ` exactly (20 bytes). F2 `YYYY-MM-DDTHH:MM:SS±HH:MM`
  (25 bytes), offset hours 00–23, minutes 00–59, UTC = local − offset; `-00:00` ok.
- F3 digits only, any length, leading zeros allowed (`0001791071890000`). Strip
  leading zeros before overflow check; too big for i64 → out of window (drop).
- Bad: lowercase z/t, space instead of T, fractional secs, no zone, `+0200`,
  `+24:00`, `+05:60`, month 13, Feb 30, Feb 29 in non-leap year, hour 24, sec 60,
  empty, `1.79e12`, `-5`. Leap years: div by 4, not 100 unless 400 (2028 is leap).
- Event window `[ws, we)`; click window `[ws - 7d, we)`. Click exactly at
  `ws - 7d` is valid; at `we` is not.

Events:
- Dedup BEFORE validation (see above). Late = `ingest_ts - ts > 24h` exactly
  86_400_000 ms; exactly 24h kept.
- `price_cents` only checked for purchases; `0250` = 250; 0 bad; >10_000_000 bad;
  empty/decimal/signed bad. Long leading-zero strings must not overflow.
- event_type exact: `Purchase`, `cart ` are bad.
- Unknown product (incl. case variants) → category `unknown`, still counts.

Links:
- Self-links and repeats (either order) are no-ops. Upper-case IDs sort before
  lowercase (byte-wise). An ID only in links can be the person's name.

Attribution:
- Click at exactly session start counts; 1 ms after does not. Exactly 7d before
  counts; 7d + 1 ms does not.

Self-check (cheap, run in debug/dev only): Σsessions and Σrevenue equal across both
tables; Σassisted ≥ Σ non-organic credited sessions.

## Speed plan (only after fixtures + check-dev pass and first submission is in)

1. Instrument per-stage timings (stderr). Generate a 20M set with the generator and
   bench inside Docker: `docker run --rm --cpus 8 --memory 24g --read-only
   -v $DATA:/data:ro -v $OUT:/output ...` and time it. Optimize the biggest stage.
2. **ID fast paths**: read `generator/` source to learn the exact formats of
   user/event/click/product/campaign IDs. If they're like `u_12345`, parse to an
   integer and index dense arrays instead of hashing. ALWAYS keep a general
   fallback (hash map on bytes) for anything that doesn't match the pattern exactly
   (quotes, spaces, upper case, leading zeros, extra chars). Byte-wise ordering of
   names/click_ids must still be computed on the real bytes (numeric order ≠
   byte order unless fixed width).
3. Parsing: process each chunk with `memchr` for `\n`, then split fields by hand;
   fast path for lines with no `"`. Parse F1/F2/F3 with hand-written code using
   days-from-civil; no chrono.
4. Fuse passes: parse events straight into per-bucket (by person or by event-hash)
   vectors to avoid a separate scatter pass.
5. Try mmap vs. big `read()` into per-thread buffers (page-fault cost on cold
   mmaps can dominate); try `MAP_POPULATE` / `madvise(SEQUENTIAL)`.
6. Parquet: uncompressed, plain encoding, write files in parallel.
7. Before the final (Sun 11:00 AM EST): generate a ~60M set, run 3 times, confirm
   < ~10 s and memory well under 24 GB. A run over 3 min = no final score.

## Status

- [x] Rust crate + Dockerfile skeleton writing empty tables
- [x] All 36 fixtures pass (local binary and Docker image)
- [x] `make check-dev` passes (dev 1M: ~0.15 s local binary, ~0.33 s via docker_run.sh incl. startup)
- Testing: `./test.sh [local|docker]`. Comparison runs in the `shop-tester` image
  (tools/tester, python+polars) since the host has no pip. Dev set: data/dev.
- [ ] First submission
- [ ] Per-stage timings on 20M local set
- [x] Per-stage timings on 20M local set (data/g20, seed 2; reference output in out-ref/g20; `./bench.sh`)
- [ ] Optimizations (record before/after times here) — bench.sh median wall, docker, 20M:
  - baseline: 3.37 s
  - u64 packed keys for u/e IDs (+exact sharded interner fallback), dense p/c_ maps: 1.39 s
  - two-stage SIMD CSV scanner (specials index + branchless field walk, exact split fallback; fuzz-tested), 1MB chunks, NCPU env honored: 1.22 s
  - parallel sharded link interning, key-order min, drop self-maps (links 0.15->0.07s): 1.37 s vs 1.38 s for previous build in same (noisier) session
- [ ] 60M safety run
