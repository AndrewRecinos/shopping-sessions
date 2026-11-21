use memchr::memchr;
use memmap2::Mmap;
use rustc_hash::{FxHashMap, FxHasher};
use std::fs::File;
use std::hash::Hasher;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DAY: i64 = 86_400_000;
const WEEK: i64 = 7 * DAY;
const GAP: i64 = 1_800_000;
const NB: usize = 256;
const CHUNK: usize = 1 << 20;

// ---------------------------------------------------------------- CSV

/// Split a line into exactly N fields, stripping whole-field quotes.
#[inline]
fn split<'a, const N: usize>(line: &'static [u8], out: &mut [&'static [u8]; N]) -> bool {
    let len = line.len();
    let mut start = 0;
    let mut n = 0;
    loop {
        let end = match memchr(b',', &line[start..]) {
            Some(p) => start + p,
            None => len,
        };
        if n == N {
            return false;
        }
        let mut f = &line[start..end];
        if memchr(b'"', f).is_some() {
            let l = f.len();
            if l >= 2 && f[0] == b'"' && f[l - 1] == b'"' && memchr(b'"', &f[1..l - 1]).is_none() {
                f = &f[1..l - 1];
            } else {
                return false;
            }
        }
        out[n] = f;
        n += 1;
        if end == len {
            break;
        }
        start = end + 1;
    }
    n == N
}

/// Iterate the non-empty lines of a chunk.
#[inline]
fn for_lines<'a>(data: &'static [u8], mut f: impl FnMut(&'static [u8])) {
    let mut s = 0;
    while s < data.len() {
        let e = match memchr(b'\n', &data[s..]) {
            Some(p) => s + p,
            None => data.len(),
        };
        if e > s {
            f(&data[s..e]);
        }
        s = e + 1;
    }
}

/// Bitmask of bytes in `p[0..16]` equal to ',', '"' or '\n'.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn special16(p: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    debug_assert!(p.len() >= 16);
    unsafe {
        let v = _mm_loadu_si128(p.as_ptr() as *const __m128i);
        let a = _mm_cmpeq_epi8(v, _mm_set1_epi8(b',' as i8));
        let b = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'"' as i8));
        let c = _mm_cmpeq_epi8(v, _mm_set1_epi8(b'\n' as i8));
        _mm_movemask_epi8(_mm_or_si128(_mm_or_si128(a, b), c)) as u32
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn special16(p: &[u8]) -> u32 {
    let mut m = 0;
    for i in 0..16 {
        m |= ((p[i] == b',' || p[i] == b'"' || p[i] == b'\n') as u32) << i;
    }
    m
}

/// Call `f` with the N fields of every well-formed row of a line-aligned chunk.
/// Stage 1 indexes every ',', '"', '\n' (SIMD); stage 2 walks fields. Any line that
/// doesn't match the strict fast shape is re-parsed with the exact `split`.
#[inline(always)]
fn for_rows<const N: usize>(data: &'static [u8], f: impl FnMut(&[&'static [u8]; N])) {
    thread_local!(static IDX: std::cell::RefCell<Vec<u32>> = Default::default());
    IDX.with(|c| for_rows_idx(data, &mut c.borrow_mut(), f))
}

#[inline(always)]
fn for_rows_idx<const N: usize>(data: &'static [u8], idx: &mut Vec<u32>, mut f: impl FnMut(&[&'static [u8]; N])) {
    let len = data.len();
    idx.clear();
    idx.reserve(len + 3 * N + 64);
    unsafe {
        let p = idx.as_mut_ptr();
        let mut n = 0usize;
        let mut base = 0usize;
        while base + 64 <= len {
            let d = &data[base..];
            let mut m = special16(d) as u64
                | (special16(&d[16..]) as u64) << 16
                | (special16(&d[32..]) as u64) << 32
                | (special16(&d[48..]) as u64) << 48;
            let cnt = m.count_ones() as usize;
            let b = base as u32;
            let mut w = p.add(n);
            loop {
                for q in 0..8 {
                    *w.add(q) = b + m.trailing_zeros();
                    m &= m.wrapping_sub(1);
                }
                if m == 0 {
                    break;
                }
                w = w.add(8);
            }
            n += cnt;
            base += 64;
        }
        while base < len {
            let c = data[base];
            if c == b',' || c == b'"' || c == b'\n' {
                *p.add(n) = base as u32;
                n += 1;
            }
            base += 1;
        }
        for _ in 0..3 * N + 8 {
            *p.add(n) = len as u32;
            n += 1;
        }
        idx.set_len(n);
    }
    let at = |p: usize| -> u8 { if p < len { data[p] } else { b'\n' } };
    let mut fields = [&b""[..]; N];
    let mut ls = 0usize; // line start
    let mut k = 0usize; // index into idx at line start
    while ls < len {
        let mut j = k;
        let mut s = ls;
        let mut good = true;
        let mut sep = 0usize;
        let mut se = [(0usize, 0usize); N];
        for fi in 0..N {
            let p0 = idx[j] as usize;
            let p1 = idx[j + 1] as usize;
            let p2 = idx[j + 2] as usize;
            let c0 = at(p0);
            let quoted = (p0 == s) & (c0 == b'"');
            let ok = if quoted { (at(p1) == b'"') & (p2 == p1 + 1) & (at(p2) != b'"') } else { c0 != b'"' };
            let start = s + quoted as usize;
            let end = if quoted { p1 } else { p0 };
            sep = if quoted { p2 } else { p0 };
            j += 1 + 2 * quoted as usize;
            good &= ok & ((at(sep) == b'\n') == (fi == N - 1));
            se[fi] = (start, end);
            s = sep + 1;
        }
        if good {
            for fi in 0..N {
                fields[fi] = unsafe { data.get_unchecked(se[fi].0..se[fi].1) };
            }
        }
        if good {
            f(&fields);
            ls = sep + 1;
            k = j;
        } else {
            // slow path: find this line's end from its first special
            let mut j = k;
            while at(idx[j] as usize) != b'\n' {
                j += 1;
            }
            let nl = idx[j] as usize;
            if nl > ls && split(&data[ls..nl], &mut fields) {
                f(&fields);
            }
            ls = nl + 1;
            k = j + 1;
        }
    }
}

// ---------------------------------------------------------------- timestamps

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[inline]
fn d2(b: &[u8], i: usize) -> Option<i64> {
    let a = b[i].wrapping_sub(b'0');
    let c = b[i + 1].wrapping_sub(b'0');
    if a < 10 && c < 10 {
        Some(a as i64 * 10 + c as i64)
    } else {
        None
    }
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Parse a digits-only string as u64, None if empty, not digits, or too large.
#[inline]
fn parse_digits(b: &[u8]) -> Option<u64> {
    if b.is_empty() {
        return None;
    }
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    let s = &b[i..];
    let mut v: u64 = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
    }
    if s.len() > 19 {
        return Some(u64::MAX);
    }
    for &c in s {
        v = v * 10 + (c - b'0') as u64;
    }
    Some(v)
}

/// Epoch ms, or None for a bad timestamp (out-of-range F3 maps to i64::MAX).
fn parse_ts(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    if b.iter().all(|c| c.is_ascii_digit()) {
        let v = parse_digits(b)?;
        return Some(if v > i64::MAX as u64 { i64::MAX } else { v as i64 });
    }
    if b.len() != 20 && b.len() != 25 {
        return None;
    }
    let y = d2(b, 0)? * 100 + d2(b, 2)?;
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let mo = d2(b, 5)?;
    let d = d2(b, 8)?;
    let hh = d2(b, 11)?;
    let mi = d2(b, 14)?;
    let ss = d2(b, 17)?;
    if !(1..=12).contains(&mo) || hh > 23 || mi > 59 || ss > 59 {
        return None;
    }
    let mdays = [31, if is_leap(y) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if d < 1 || d > mdays[(mo - 1) as usize] {
        return None;
    }
    let mut t = ((days_from_civil(y, mo, d) * 24 + hh) * 60 + mi) * 60 + ss;
    t *= 1000;
    if b.len() == 20 {
        if b[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if b[22] != b':' || oh > 23 || om > 59 {
            return None;
        }
        t -= sign * (oh * 60 + om) * 60_000;
    }
    Some(t)
}

// ---------------------------------------------------------------- helpers

#[inline]
fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(b);
    h.finish().wrapping_mul(0x9E37_79B9_7F4A_7C15)
}


/// Pack `prefix[0-9a-z]{1,8}` into a u64 (big-endian, zero padded; top bit 0).
/// Injective, and preserves byte order among packed IDs of the same prefix.
#[inline]
fn pack(prefix: u8, b: &[u8]) -> Option<u64> {
    let n = b.len();
    if n < 2 || n > 9 || b[0] != prefix {
        return None;
    }
    let mut v = 0u64;
    for &c in &b[1..] {
        if !(c.is_ascii_digit() || c.is_ascii_lowercase()) {
            return None;
        }
        v = (v << 8) | c as u64;
    }
    Some(v << (8 * (9 - n)))
}

fn unpack(prefix: u8, k: u64, out: &mut Vec<u8>) {
    out.push(prefix);
    for c in k.to_be_bytes() {
        if c != 0 {
            out.push(c);
        }
    }
}

const FB: u64 = 1 << 63;

/// Exact interning for IDs that don't match the fast pattern: key = FB | idx<<6 | shard.
struct Interner {
    shards: Vec<std::sync::Mutex<(FxHashMap<&'static [u8], u64>, Vec<&'static [u8]>)>>,
}

impl Interner {
    fn new() -> Self {
        Interner { shards: (0..64).map(|_| Default::default()).collect() }
    }
    fn key(&self, b: &'static [u8]) -> u64 {
        let s = (hash_bytes(b) >> 58) as usize;
        let mut g = self.shards[s].lock().unwrap();
        if let Some(&k) = g.0.get(b) {
            return k;
        }
        let k = FB | ((g.1.len() as u64) << 6) | s as u64;
        g.1.push(b);
        g.0.insert(b, k);
        k
    }
    fn bytes(&self, k: u64) -> &'static [u8] {
        let g = self.shards[(k & 63) as usize].lock().unwrap();
        g.1[((k & !FB) >> 6) as usize]
    }
}

#[inline]
fn hash64(k: u64) -> u64 {
    (k ^ (k >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn map_file(p: &Path) -> Option<&'static Mmap> {
    map_file0(p).map(|m| &*Box::leak(Box::new(m)))
}

fn map_file0(p: &Path) -> Option<Mmap> {
    let f = File::open(p).ok()?;
    let len = f.metadata().ok()?.len();
    if len == 0 {
        return None;
    }
    unsafe { Mmap::map(&f).ok() }
}

fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().map_or(false, |x| x == "csv"))
            .collect(),
        Err(_) => vec![],
    };
    v.sort();
    v
}

/// File body after the header line.
fn body(m: &'static Mmap) -> &'static [u8] {
    match memchr(b'\n', m) {
        Some(p) => &m[p + 1..],
        None => &[],
    }
}

/// Line-aligned chunks of all bodies.
fn chunks(maps: &[&'static Mmap]) -> Vec<&'static [u8]> {
    let mut out = vec![];
    for m in maps {
        let b = body(m);
        let mut s = 0;
        while s < b.len() {
            let mut e = (s + CHUNK).min(b.len());
            if e < b.len() {
                e = match memchr(b'\n', &b[e..]) {
                    Some(p) => e + p + 1,
                    None => b.len(),
                };
            }
            out.push(&b[s..e]);
            s = e;
        }
    }
    // largest first for balance
    out.sort_by_key(|c| std::cmp::Reverse(c.len()));
    out
}

/// Run `work(item, &mut state)` over items 0..n on nt threads; returns per-thread states.
fn par<S: Send, I: Fn() -> S + Sync, F: Fn(usize, &mut S) + Sync>(n: usize, nt: usize, init: I, work: F) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|_| {
                sc.spawn(|| {
                    let mut s = init();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        work(i, &mut s);
                    }
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn t(stage: &str, t0: &mut Instant) {
    eprintln!("{stage} {:.3}s", t0.elapsed().as_secs_f64());
    *t0 = Instant::now();
}

// ---------------------------------------------------------------- records

#[derive(Clone, Copy)]
struct ERaw {
    ekey: u64,
    ingest: i64,
    pkey: u64,
    ts: u32,
    price: u32,
    cat: u16,
    typ: u8,
    ok: bool,
}

#[derive(Clone, Copy)]
struct Ev {
    pkey: u64,
    ts: u32,
    price: u32,
    cat: u16,
    typ: u8, // 2 = purchase
}

#[derive(Clone, Copy)]
struct Ck {
    pkey: u64,
    cid: &'static [u8],
    ts: u32,
    camp: u32,
}

struct URow {
    pkey: u64,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u16>,
}

/// `prefix` followed by 1..=7 digits without leading zero.
#[inline]
fn num(prefix: &[u8], b: &[u8]) -> Option<usize> {
    let p = prefix.len();
    if b.len() <= p || b.len() > p + 7 || &b[..p] != prefix || b[p] == b'0' {
        return None;
    }
    let mut v = 0usize;
    for &c in &b[p..] {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        v = v * 10 + d as usize;
    }
    Some(v)
}

/// Exact ID → value lookup: dense array for `prefix<number>` IDs, hash map otherwise.
struct IdMap<V: Copy> {
    prefix: &'static [u8],
    dense: Vec<Option<V>>,
    other: FxHashMap<&'static [u8], V>,
}

impl<V: Copy> IdMap<V> {
    fn new(prefix: &'static [u8], items: impl Iterator<Item = (&'static [u8], V)>) -> Self {
        let mut m = IdMap { prefix, dense: vec![], other: FxHashMap::default() };
        for (k, v) in items {
            match num(prefix, k) {
                Some(n) if n < (1 << 22) => {
                    if m.dense.len() <= n {
                        m.dense.resize(n + 1, None);
                    }
                    m.dense[n].get_or_insert(v);
                }
                _ => {
                    m.other.entry(k).or_insert(v);
                }
            }
        }
        m
    }
    #[inline]
    fn get(&self, k: &[u8]) -> Option<V> {
        match num(self.prefix, k) {
            Some(n) if n < (1 << 22) => self.dense.get(n).copied().flatten(),
            _ => self.other.get(k).copied(),
        }
    }
}

// ---------------------------------------------------------------- main

fn main() {
    run();
}

fn run() {
    let t_all = Instant::now();
    let mut t0 = Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let outd = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let nt = std::env::var("NCPU").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)).clamp(1, 64);

    // manifest
    let man = std::fs::read(data.join("manifest.json")).expect("manifest");
    let get = |key: &str| -> i64 {
        let k = format!("\"{key}\"");
        let p = man.windows(k.len()).position(|w| w == k.as_bytes()).expect("key") + k.len();
        let rest = &man[p..];
        let q1 = memchr(b'"', rest).unwrap() + 1;
        let q2 = memchr(b'"', &rest[q1..]).unwrap() + q1;
        parse_ts(&rest[q1..q2]).expect("manifest ts")
    };
    let ws = get("window_start");
    let we = get("window_end");
    let base = ws - WEEK;
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;

    // products / categories
    let pm = map_file(&data.join("products.csv"));
    let mut prod_raw: Vec<(&[u8], &[u8])> = vec![];
    if let Some(m) = pm {
        for_lines(body(m), |l| {
            let mut f = [&b""[..]; 2];
            if split(l, &mut f) {
                prod_raw.push((f[0], f[1]));
            }
        });
    }
    let mut cats: Vec<&[u8]> = prod_raw.iter().map(|p| p.1).collect();
    cats.push(b"unknown");
    cats.sort_unstable();
    cats.dedup();
    let cat_idx: FxHashMap<&[u8], u16> = cats.iter().enumerate().map(|(i, c)| (*c, i as u16)).collect();
    let unknown = cat_idx[&b"unknown"[..]];
    let products: IdMap<u16> = IdMap::new(b"p", prod_raw.iter().map(|(p, c)| (*p, cat_idx[c])));
    let ncat = cats.len();

    // campaigns
    let cm = map_file(&data.join("campaigns.csv"));
    let mut camps: Vec<(&[u8], &[u8])> = vec![];
    if let Some(m) = cm {
        for_lines(body(m), |l| {
            let mut f = [&b""[..]; 2];
            if split(l, &mut f) {
                camps.push((f[0], f[1]));
            }
        });
    }
    let camp_idx: IdMap<u32> = IdMap::new(b"c_", camps.iter().enumerate().map(|(i, c)| (c.0, i as u32)));
    let ncamp = camps.len();
    t("small_inputs", &mut t0);

    // map big files
    let lmaps: Vec<&Mmap> = list_csv(&data.join("links")).iter().filter_map(|p| map_file(p)).collect();
    let emaps: Vec<&Mmap> = list_csv(&data.join("events")).iter().filter_map(|p| map_file(p)).collect();
    let cmaps: Vec<&Mmap> = list_csv(&data.join("clicks")).iter().filter_map(|p| map_file(p)).collect();
    let lch = chunks(&lmaps);
    let ech = chunks(&emaps);
    let cch = chunks(&cmaps);
    t("map_files", &mut t0);
    let users = Interner::new();
    let ukey = |b: &'static [u8]| -> u64 { pack(b'u', b).unwrap_or_else(|| users.key(b)) };
    let eids = Interner::new();

    // ---------------- links
    let pairs = par(lch.len(), nt, Vec::new, |i, v: &mut Vec<(u64, u64, &[u8], &[u8])>| {
        for_rows::<2>(lch[i], |f| {
            if !f[0].is_empty() && !f[1].is_empty() {
                v.push((ukey(f[0]), ukey(f[1]), f[0], f[1]));
            }
        });
    });
    let mut ids: FxHashMap<u64, u32> = FxHashMap::default();
    let mut id_bytes: Vec<(u64, &[u8])> = vec![];
    let mut parent: Vec<u32> = vec![];
    let mut size: Vec<u32> = vec![];
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        let mut r = x;
        while parent[r as usize] != r {
            r = parent[r as usize];
        }
        while parent[x as usize] != r {
            let n = parent[x as usize];
            parent[x as usize] = r;
            x = n;
        }
        r
    }
    for v in &pairs {
        for &(ka, kb, a, b) in v {
            let mut intern = |k: u64, s: &'static [u8]| -> u32 {
                *ids.entry(k).or_insert_with(|| {
                    id_bytes.push((k, s));
                    parent.push(parent.len() as u32);
                    size.push(1);
                    (id_bytes.len() - 1) as u32
                })
            };
            let ia = intern(ka, a);
            let ib = intern(kb, b);
            let ra = find(&mut parent, ia);
            let rb = find(&mut parent, ib);
            if ra != rb {
                let (big, small) = if size[ra as usize] >= size[rb as usize] { (ra, rb) } else { (rb, ra) };
                parent[small as usize] = big;
                size[big as usize] += size[small as usize];
            }
        }
    }
    let nid = id_bytes.len();
    let mut minid: Vec<u32> = (0..nid as u32).collect();
    for i in 0..nid as u32 {
        let r = find(&mut parent, i) as usize;
        if id_bytes[i as usize].1 < id_bytes[minid[r] as usize].1 {
            minid[r] = i;
        }
    }
    let mut link_map: FxHashMap<u64, u64> = FxHashMap::default();
    for i in 0..nid {
        let m = minid[parent[i] as usize] as usize;
        if m != i {
            link_map.insert(id_bytes[i].0, id_bytes[m].0);
        }
    }
    drop(ids);
    t("links", &mut t0);
    let link_map = &link_map;
    let person = |u: &'static [u8]| -> u64 {
        let k = ukey(u);
        link_map.get(&k).copied().unwrap_or(k)
    };
    // ---------------- events: parse into eid buckets
    let eraw = par(
        ech.len(),
        nt,
        || (0..NB).map(|_| Vec::<ERaw>::new()).collect::<Vec<_>>(),
        |i, bk| {
            for_rows::<7>(ech[i], |f| {
                if f[0].is_empty() {
                    return;
                }
                let ingest = parse_ts(f[3]).unwrap_or(i64::MAX);
                let user = f[1];
                let mut ok = !user.is_empty() && !f[5].is_empty();
                let ts = parse_ts(f[2]);
                let tsv = ts.unwrap_or(0);
                ok &= ts.is_some() && tsv >= ws && tsv < we;
                let typ = match f[4] {
                    b"view" => 0u8,
                    b"cart" => 1,
                    b"purchase" => 2,
                    _ => {
                        ok = false;
                        3
                    }
                };
                let mut price = 0u32;
                if typ == 2 {
                    match parse_digits(f[6]) {
                        Some(p) if (1..=10_000_000).contains(&p) => price = p as u32,
                        _ => ok = false,
                    }
                }
                if ok && ingest.saturating_sub(tsv) > DAY {
                    ok = false;
                }
                let (pkey, cat, tsr) = if ok {
                    (person(user), products.get(f[5]).unwrap_or(unknown), (tsv - base) as u32)
                } else {
                    (0, 0, 0)
                };
                let ekey = pack(b'e', f[0]).unwrap_or_else(|| eids.key(f[0]));
                bk[(hash64(ekey) >> 56) as usize].push(ERaw { ekey, ingest, pkey, ts: tsr, price, cat, typ, ok });
            });
        },
    );
    t("parse_events", &mut t0);

    // ---------------- dedup -> person buckets
    let evb = par(
        NB,
        nt,
        || (0..NB).map(|_| Vec::<Ev>::new()).collect::<Vec<_>>(),
        |b, out| {
            let mut v: Vec<ERaw> = Vec::with_capacity(eraw.iter().map(|t| t[b].len()).sum());
            for tv in &eraw {
                v.extend_from_slice(&tv[b]);
            }
            v.sort_unstable_by_key(|x| (x.ekey, x.ingest));
            let mut i = 0;
            while i < v.len() {
                let r = v[i];
                let mut j = i + 1;
                while j < v.len() && v[j].ekey == r.ekey {
                    j += 1;
                }
                if r.ok {
                    out[(hash64(r.pkey) >> 56) as usize].push(Ev { pkey: r.pkey, ts: r.ts, price: r.price, cat: r.cat, typ: r.typ });
                }
                i = j;
            }
        },
    );
    drop(eraw);
    t("dedup_events", &mut t0);

    // ---------------- clicks
    let ckb = par(
        cch.len(),
        nt,
        || (0..NB).map(|_| Vec::<Ck>::new()).collect::<Vec<_>>(),
        |i, out| {
            for_rows::<4>(cch[i], |f| {
                if f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                let ts = match parse_ts(f[3]) {
                    Some(t) if t >= ws - WEEK && t < we => t,
                    _ => return,
                };
                let camp = match camp_idx.get(f[2]) {
                    Some(c) => c,
                    None => return,
                };
                let pkey = person(f[1]);
                out[(hash64(pkey) >> 56) as usize].push(Ck { pkey, cid: f[0], ts: (ts - base) as u32, camp });
            });
        },
    );
    t("parse_clicks", &mut t0);
    // ---------------- sessionize + attribute
    let nc1 = ncamp + 1;
    let results = par(
        NB,
        nt,
        || (Vec::<URow>::new(), vec![[0i64; 4]; nc1 * ndays], vec![0u32; nc1], vec![0u32; nc1], Vec::<u32>::new(), vec![0i64; ncat]),
        |b, (urows, carr, cnt, pos, act, catsum)| {
            let mut ev: Vec<Ev> = Vec::with_capacity(evb.iter().map(|t| t[b].len()).sum());
            for tv in &evb {
                ev.extend_from_slice(&tv[b]);
            }
            let mut ck: Vec<Ck> = Vec::with_capacity(ckb.iter().map(|t| t[b].len()).sum());
            for tv in &ckb {
                ck.extend_from_slice(&tv[b]);
            }
            ev.sort_unstable_by_key(|x| (x.pkey, x.ts));
            ck.sort_unstable_by(|x, y| (x.pkey, x.ts).cmp(&(y.pkey, y.ts)).then_with(|| x.cid.cmp(y.cid)));
            let mut ci = 0;
            let mut i = 0;
            while i < ev.len() {
                let pkey = ev[i].pkey;
                let mut iend = i + 1;
                while iend < ev.len() && ev[iend].pkey == pkey {
                    iend += 1;
                }
                while ci < ck.len() && ck[ci].pkey < pkey {
                    ci += 1;
                }
                let mut cend = ci;
                while cend < ck.len() && ck[cend].pkey == pkey {
                    cend += 1;
                }
                let e = &ev[i..iend];
                let c = &ck[ci..cend];
                // person
                let (mut lo, mut hi) = (0usize, 0usize);
                act.clear();
                let mut cur_day = i32::MIN;
                let mut row = URow { pkey, day: 0, sessions: 0, events: 0, revenue: 0, top: None };
                let mut any_purchase = false;
                fn flush(row: &mut URow, catsum: &mut [i64], any: bool, urows: &mut Vec<URow>) {
                    let top = if any {
                        let mut best = 0;
                        for k in 1..catsum.len() {
                            if catsum[k] > catsum[best] {
                                best = k;
                            }
                        }
                        Some(best as u16)
                    } else {
                        None
                    };
                    urows.push(URow { pkey: row.pkey, day: row.day, sessions: row.sessions, events: row.events, revenue: row.revenue, top });
                    catsum.iter_mut().for_each(|x| *x = 0);
                }
                let mut k = 0;
                while k < e.len() {
                    let start = e[k].ts as i64;
                    let mut j = k + 1;
                    let mut prev = start;
                    while j < e.len() && (e[j].ts as i64) - prev <= GAP {
                        prev = e[j].ts as i64;
                        j += 1;
                    }
                    let day = (start + base).div_euclid(DAY) as i32;
                    if day != cur_day {
                        if cur_day != i32::MIN {
                            flush(&mut row, catsum, any_purchase, urows);
                        }
                        cur_day = day;
                        row = URow { pkey, day, sessions: 0, events: 0, revenue: 0, top: None };
                        any_purchase = false;
                    }
                    let mut rev = 0i64;
                    let mut purch = false;
                    for x in &e[k..j] {
                        if x.typ == 2 {
                            rev += x.price as i64;
                            catsum[x.cat as usize] += x.price as i64;
                            purch = true;
                        }
                    }
                    row.sessions += 1;
                    row.events += (j - k) as i64;
                    row.revenue += rev;
                    any_purchase |= purch;
                    // clicks window [start-WEEK, start]
                    while hi < c.len() && (c[hi].ts as i64) <= start {
                        let cp = c[hi].camp as usize;
                        if cnt[cp] == 0 {
                            pos[cp] = act.len() as u32;
                            act.push(cp as u32);
                        }
                        cnt[cp] += 1;
                        hi += 1;
                    }
                    while lo < hi && (c[lo].ts as i64) < start - WEEK {
                        let cp = c[lo].camp as usize;
                        cnt[cp] -= 1;
                        if cnt[cp] == 0 {
                            let p = pos[cp] as usize;
                            let last = *act.last().unwrap();
                            act[p] = last;
                            pos[last as usize] = p as u32;
                            act.pop();
                        }
                        lo += 1;
                    }
                    let di = (day as i64 - day0) as usize;
                    let credit = if hi > lo {
                        let tt = c[hi - 1].ts;
                        let mut q = hi - 1;
                        while q > lo && c[q - 1].ts == tt {
                            q -= 1;
                        }
                        c[q].camp as usize
                    } else {
                        ncamp
                    };
                    let cell = &mut carr[credit * ndays + di];
                    cell[0] += 1;
                    cell[1] += purch as i64;
                    cell[2] += rev;
                    for &a in act.iter() {
                        carr[a as usize * ndays + di][3] += 1;
                    }
                    k = j;
                }
                if cur_day != i32::MIN {
                    flush(&mut row, catsum, any_purchase, urows);
                }
                // reset click counts
                for &a in act.iter() {
                    cnt[a as usize] = 0;
                }
                act.clear();
                i = iend;
                ci = cend;
            }
        },
    );
    t("sessionize", &mut t0);

    // ---------------- write
    let ud = outd.join("user_daily");
    let cd = outd.join("campaign_daily");
    std::fs::create_dir_all(&ud).expect("mkdir");
    std::fs::create_dir_all(&cd).expect("mkdir");
    let mut camp_tot = vec![[0i64; 4]; nc1 * ndays];
    for r in &results {
        for (a, b) in camp_tot.iter_mut().zip(r.1.iter()) {
            for q in 0..4 {
                a[q] += b[q];
            }
        }
    }
    std::thread::scope(|sc| {
        for (ti, r) in results.iter().enumerate() {
            let ud = &ud;
            let cats = &cats;
            let users = &users;
            sc.spawn(move || write_user(&ud.join(format!("part-{ti}.parquet")), &r.0, cats, users));
        }
        write_camp(&cd.join("part-0.parquet"), &camp_tot, &camps, ndays, day0);
    });
    t("write", &mut t0);
    eprintln!("total {:.3}s", t_all.elapsed().as_secs_f64());
    std::process::exit(0);
}

// ---------------------------------------------------------------- parquet

use parquet::basic::Compression;
use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .build(),
    )
}

enum Col {
    S(Vec<ByteArray>, Option<Vec<i16>>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

fn write_cols(path: &Path, schema: &str, cols: Vec<Col>, nrows: usize) {
    let schema = Arc::new(parse_message_type(schema).expect("schema"));
    let f = BufWriter::with_capacity(1 << 20, File::create(path).expect("create"));
    let mut w = SerializedFileWriter::new(f, schema, props()).expect("writer");
    if nrows > 0 {
        let mut rg = w.next_row_group().expect("rg");
        let mut it = cols.into_iter();
        while let Some(mut col) = rg.next_column().expect("col") {
            match it.next().expect("col data") {
                Col::S(v, d) => {
                    col.typed::<ByteArrayType>().write_batch(&v, d.as_deref(), None).expect("w");
                }
                Col::I32(v) => {
                    col.typed::<Int32Type>().write_batch(&v, None, None).expect("w");
                }
                Col::I64(v) => {
                    col.typed::<Int64Type>().write_batch(&v, None, None).expect("w");
                }
            }
            col.close().expect("close col");
        }
        rg.close().expect("close rg");
    }
    let mut inner = w.into_inner().expect("finish");
    inner.flush().expect("flush");
}

fn write_user(path: &Path, rows: &[URow], cats: &[&[u8]], users: &Interner) {
    let mut defs = Vec::with_capacity(rows.len());
    let mut tops = vec![];
    for r in rows {
        match r.top {
            Some(c) => {
                defs.push(1i16);
                tops.push(ByteArray::from(cats[c as usize].to_vec()));
            }
            None => defs.push(0),
        }
    }
    let cols = vec![
        Col::S(
            rows.iter()
                .map(|r| {
                    if r.pkey & FB != 0 {
                        ByteArray::from(users.bytes(r.pkey).to_vec())
                    } else {
                        let mut v = Vec::with_capacity(9);
                        unpack(b'u', r.pkey, &mut v);
                        ByteArray::from(v)
                    }
                })
                .collect(),
            None,
        ),
        Col::I32(rows.iter().map(|r| r.day).collect()),
        Col::I64(rows.iter().map(|r| r.sessions).collect()),
        Col::I64(rows.iter().map(|r| r.events).collect()),
        Col::I64(rows.iter().map(|r| r.revenue).collect()),
        Col::S(tops, Some(defs)),
    ];
    write_cols(
        path,
        "message user_daily { required binary user_id (UTF8); required int32 day (DATE); required int64 sessions; \
         required int64 events; required int64 revenue_cents; optional binary top_category (UTF8); }",
        cols,
        rows.len(),
    );
}

fn write_camp(path: &Path, tot: &[[i64; 4]], camps: &[(&[u8], &[u8])], ndays: usize, day0: i64) {
    let (mut id, mut ch, mut day, mut s, mut p, mut r, mut a) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for c in 0..=camps.len() {
        let (cid, chn): (&[u8], &[u8]) = if c < camps.len() { camps[c] } else { (b"organic", b"organic") };
        for d in 0..ndays {
            let x = tot[c * ndays + d];
            if x[0] == 0 && x[3] == 0 {
                continue;
            }
            id.push(ByteArray::from(cid.to_vec()));
            ch.push(ByteArray::from(chn.to_vec()));
            day.push((day0 + d as i64) as i32);
            s.push(x[0]);
            p.push(x[1]);
            r.push(x[2]);
            a.push(x[3]);
        }
    }
    let n = id.len();
    write_cols(
        path,
        "message campaign_daily { required binary campaign_id (UTF8); required binary channel (UTF8); \
         required int32 day (DATE); required int64 sessions; required int64 purchasing_sessions; \
         required int64 revenue_cents; required int64 assisted_sessions; }",
        vec![Col::S(id, None), Col::S(ch, None), Col::I32(day), Col::I64(s), Col::I64(p), Col::I64(r), Col::I64(a)],
        n,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference<const N: usize>(d: &'static [u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = vec![];
        for_lines(d, |l| {
            let mut f = [&b""[..]; N];
            if split(l, &mut f) {
                out.push(f.iter().map(|x| x.to_vec()).collect());
            }
        });
        out
    }
    fn fast<const N: usize>(d: &'static [u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = vec![];
        for_rows::<N>(d, |f| out.push(f.iter().map(|x| x.to_vec()).collect()));
        out
    }

    #[test]
    fn rows_match_split() {
        let mut x: u64 = 12345;
        let alpha = b"ab,,\"\"\n\r x";
        for it in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let n = (x % 300) as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                v.push(alpha[(x % alpha.len() as u64) as usize]);
            }
            let d: &'static [u8] = Box::leak(v.into_boxed_slice());
            assert_eq!(reference::<2>(d), fast::<2>(d), "N=2 it={it} {:?}", String::from_utf8_lossy(d));
            assert_eq!(reference::<4>(d), fast::<4>(d), "N=4 it={it}");
            assert_eq!(reference::<7>(d), fast::<7>(d), "N=7 it={it}");
        }
    }
}
