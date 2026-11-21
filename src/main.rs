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

/// Category field value meaning "category >= ESC, see the side tables" (only with > 255 categories).
const ESC: u32 = 255;
const DAY: i64 = 86_400_000;
const WEEK: i64 = 7 * DAY;
const GAP: i64 = 1_800_000;
const NBB: u32 = 10;
const NB: usize = 1 << NBB;
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

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

/// Exact lookup of `YYYY-MM-DD` strings for days near the window, indexed by the DD digits.
/// An entry only matches its own (valid) date bytes, so a hit is always correct; a miss
/// falls back to full validation.
struct DateTab {
    e: [(u64, u16, i64); 166],
}

impl DateTab {
    fn new(first_day: i64, last_day: i64) -> Self {
        let key = |day: i64| -> (u64, u16, i64) {
            let (y, m, d) = civil_from_days(day);
            let s = format!("{y:04}-{m:02}-{d:02}");
            let b = s.as_bytes();
            (u64::from_le_bytes(b[0..8].try_into().unwrap()), u16::from_le_bytes(b[8..10].try_into().unwrap()), day)
        };
        let mut t = DateTab { e: [key(10957); 166] };
        for day in first_day..=last_day {
            let k = key(day);
            if (1000..=9999).contains(&civil_from_days(day).0) {
                let (_, _, d) = civil_from_days(day);
                t.e[d as usize] = k;
            }
        }
        t
    }
}

/// Same result as `parse_ts`, with a table lookup for dates near the window and a fast
/// path for short digit strings.
#[inline(always)]
fn parse_ts_c(b: &[u8], c: &DateTab) -> Option<i64> {
    match parse_ts_f(b, c) {
        Some(t) => Some(t),
        None => parse_ts(b),
    }
}

#[inline(always)]
fn parse_ts_f(b: &[u8], c: &DateTab) -> Option<i64> {
    let n = b.len();
    if n == 20 || n == 25 {
        let lo = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let hi = u16::from_le_bytes(b[8..10].try_into().unwrap());
        let e = &c.e[(b[8] & 15) as usize * 10 + (b[9] & 15) as usize];
        if lo != e.0 || hi != e.1 || b[10] != b'T' {
            return None;
        }
        // "HH:MM:SS": digits and colons, then pairwise 10*a+b
        let v = u64::from_le_bytes(b[11..19].try_into().unwrap());
        let x = v ^ 0x3030_3030_3030_3030 ^ (0x0A << 16 | 0x0A << 40);
        if (x | x.wrapping_add(0x7676_7676_7676_7676)) & 0x8080_8080_8080_8080 != 0 || x & 0x0000_FF00_00FF_0000 != 0 {
            return None;
        }
        let y = x * 10 + (x >> 8);
        let (hh, mi, ss) = ((y & 0xFF) as i64, (y >> 24 & 0xFF) as i64, (y >> 48 & 0xFF) as i64);
        if hh > 23 || mi > 59 || ss > 59 {
            return None;
        }
        let mut t = (((e.2 * 24 + hh) * 60 + mi) * 60 + ss) * 1000;
        if n == 20 {
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
        return Some(t);
    }
    if n >= 1 && n <= 18 {
        let mut v = 0i64;
        let mut bad = 0u8;
        for &ch in b {
            let d = ch.wrapping_sub(b'0');
            bad |= (d >= 10) as u8;
            v = v * 10 + d as i64;
        }
        if bad == 0 {
            return Some(v);
        }
    }
    None
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
#[inline(always)]
fn pack(prefix: u8, b: &[u8]) -> Option<u64> {
    let n = b.len();
    if n < 2 || n > 9 || b[0] != prefix {
        return None;
    }
    let end = b.as_ptr() as usize + n;
    if (end - 8) >> 12 == (end - 1) >> 12 {
        // SWAR: load the 8 bytes ending at the field end (same page as its last byte)
        let w = u64::from_be_bytes(unsafe { std::ptr::read_unaligned((end - 8) as *const [u8; 8]) });
        let k = n - 1;
        let m80 = 0x8080_8080_8080_8080u64 >> (8 * (8 - k));
        let mask = u64::MAX >> (8 * (8 - k));
        let x = w & mask;
        let lo7 = x & 0x7F7F_7F7F_7F7F_7F7F;
        let ge0 = lo7.wrapping_add(0x5050_5050_5050_5050);
        let lt9 = !lo7.wrapping_add(0x4646_4646_4646_4646);
        let gea = lo7.wrapping_add(0x1F1F_1F1F_1F1F_1F1F);
        let ltz = !lo7.wrapping_add(0x0505_0505_0505_0505);
        let valid = ((ge0 & lt9) | (gea & ltz)) & !x & m80;
        return if valid == m80 { Some(x << (8 * (9 - n))) } else { None };
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

const EMPTY: u64 = u64::MAX;

/// Leaked anonymous mapping advised for transparent huge pages (fewer TLB misses).
fn huge_slice<T: Copy>(n: usize, init: T) -> &'static mut [T] {
    let bytes = (n * std::mem::size_of::<T>()).max(1);
    let m = memmap2::MmapMut::map_anon(bytes).expect("anon map");
    #[cfg(target_os = "linux")]
    let _ = m.advise(memmap2::Advice::HugePage);
    let m = Box::leak(Box::new(m));
    let s = unsafe { std::slice::from_raw_parts_mut(m.as_mut_ptr() as *mut T, n) };
    s.fill(init);
    s
}

/// Lock-free open-addressing table of link IDs: slot index = node id, value = person key.
/// Linear probing over the whole table; a missing key maps to itself.
#[repr(C)]
struct Slot {
    k: std::sync::atomic::AtomicU64,
    v: std::sync::atomic::AtomicU64,
}

struct LTab {
    slots: &'static [Slot],
}

impl LTab {
    fn new(n: usize) -> Self {
        let raw = huge_slice(n * 2, EMPTY);
        LTab { slots: unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const Slot, n) } }
    }
    #[inline(always)]
    fn home(&self, k: u64) -> usize {
        ((hash64(k) as u128 * self.slots.len() as u128) >> 64) as usize
    }
    /// Slot of `k`, inserting it if absent.
    fn insert(&self, k: u64) -> u32 {
        let n = self.slots.len();
        let mut i = self.home(k);
        loop {
            let s = &self.slots[i];
            let cur = s.k.load(Ordering::Relaxed);
            if cur == k {
                return i as u32;
            }
            if cur == EMPTY {
                match s.k.compare_exchange(EMPTY, k, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => return i as u32,
                    Err(c) if c == k => return i as u32,
                    Err(_) => {}
                }
            }
            i += 1;
            if i == n {
                i = 0;
            }
        }
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        let n = self.slots.len();
        let mut i = self.home(k);
        loop {
            let s = unsafe { self.slots.get_unchecked(i) };
            let kk = s.k.load(Ordering::Relaxed);
            if kk == k {
                return s.v.load(Ordering::Relaxed);
            }
            if kk == EMPTY {
                return k;
            }
            i += 1;
            if i == n {
                i = 0;
            }
        }
    }
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            _mm_prefetch(self.slots.as_ptr().add(self.home(k)) as *const i8, _MM_HINT_T0);
        }
    }
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
    pkey: u64, // 0 = invalid / late
    ts: u32,
    pc: u32, // purchase price (24 bits, 0 = not a purchase) | category << 24
}

#[derive(Clone, Copy)]
struct Ev {
    pkey: u64,
    ts: u32,
    pc: u32,
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
    let dtab = DateTab::new(day0 - 8, (we - 1).div_euclid(DAY) + 2);
    let dtab = &dtab;

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
    let nlines: usize = lmaps.iter().map(|m| memchr::memchr_iter(b'\n', m).count() + 1).sum();
    let ltab = LTab::new(nlines * 5 / 2 + 64);
    let nslot = ltab.slots.len();
    let upairs = par(lch.len(), nt, Vec::new, |i, pv: &mut Vec<(u32, u32)>| {
        for_rows::<2>(lch[i], |f| {
            if !f[0].is_empty() && !f[1].is_empty() {
                pv.push((ltab.insert(ukey(f[0])), ltab.insert(ukey(f[1]))));
            }
        });
    });
    // lock-free union-find: link the larger root under the smaller (CAS), path halving
    use std::sync::atomic::AtomicU32;
    let parent: Vec<AtomicU32> = (0..nslot as u32).map(AtomicU32::new).collect();
    let find = |mut x: u32| -> u32 {
        loop {
            let p = parent[x as usize].load(Ordering::Relaxed);
            if p == x {
                return x;
            }
            let g = parent[p as usize].load(Ordering::Relaxed);
            if g != p {
                let _ = parent[x as usize].compare_exchange_weak(p, g, Ordering::Relaxed, Ordering::Relaxed);
            }
            x = g;
        }
    };
    par(upairs.len(), nt, || (), |t, _| {
        for &(a, b) in &upairs[t] {
            let (mut a, mut b) = (a, b);
            loop {
                a = find(a);
                b = find(b);
                if a == b {
                    break;
                }
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if parent[hi as usize].compare_exchange(hi, lo, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                    break;
                }
            }
        }
    });
    let parent: Vec<u32> = parent.into_iter().map(|x| x.into_inner()).collect();
    // component root and byte-wise minimum ID (atomic min), then value = that ID's key
    const RB: usize = 1 << 16;
    let nrb = nslot.div_ceil(RB);
    let minid: Vec<AtomicU32> = (0..nslot).map(|_| AtomicU32::new(u32::MAX)).collect();
    {
        let (parent, minid, users) = (&parent, &minid, &users);
        let key = |i: u32| ltab.slots[i as usize].k.load(Ordering::Relaxed);
        let less = |i: u32, j: u32| {
            let (a, b) = (key(i), key(j));
            if (a | b) & FB == 0 {
                a < b
            } else {
                let (mut x, mut y) = (Vec::new(), Vec::new());
                let ab: &[u8] = if a & FB != 0 { users.bytes(a) } else { unpack(b'u', a, &mut x); &x };
                let bb: &[u8] = if b & FB != 0 { users.bytes(b) } else { unpack(b'u', b, &mut y); &y };
                ab < bb
            }
        };
        par(nrb, nt, || (), |q, _| {
            for i in q * RB..((q + 1) * RB).min(nslot) {
                if key(i as u32) == EMPTY {
                    continue;
                }
                let mut r = i as u32;
                while parent[r as usize] != r {
                    r = parent[r as usize];
                }
                let m = &minid[r as usize];
                let mut cur = m.load(Ordering::Relaxed);
                while cur == u32::MAX || less(i as u32, cur) {
                    match m.compare_exchange_weak(cur, i as u32, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(c) => cur = c,
                    }
                }
            }
        });
        par(nrb, nt, || (), |q, _| {
            for i in q * RB..((q + 1) * RB).min(nslot) {
                if key(i as u32) == EMPTY {
                    continue;
                }
                let mut r = i as u32;
                while parent[r as usize] != r {
                    r = parent[r as usize];
                }
                let s = &ltab.slots[i];
                s.v.store(key(minid[r as usize].load(Ordering::Relaxed)), Ordering::Relaxed);
            }
        });
    }
    drop(upairs);
    drop(parent);
    drop(minid);
    // compact lookup table holding only IDs whose person key differs from their own
    let moved = |s: &Slot| {
        let k = s.k.load(Ordering::Relaxed);
        k != EMPTY && s.v.load(Ordering::Relaxed) != k
    };
    let nmoved: usize = par(nrb, nt, || 0usize, |q, c| *c += ltab.slots[q * RB..((q + 1) * RB).min(nslot)].iter().filter(|s| moved(s)).count()).iter().sum();
    let ctab = LTab::new(nmoved * 2 + 64);
    par(nrb, nt, || (), |q, _| {
        for s in &ltab.slots[q * RB..((q + 1) * RB).min(nslot)] {
            if moved(s) {
                let i = ctab.insert(s.k.load(Ordering::Relaxed));
                ctab.slots[i as usize].v.store(s.v.load(Ordering::Relaxed), Ordering::Relaxed);
            }
        }
    });
    t("links", &mut t0);
    let ltab = &ctab;
    // ---------------- events: parse into eid buckets
    let ecap = ech.iter().map(|c| c.len()).sum::<usize>() / 64 / nt / NB;
    let wide1: std::sync::Mutex<FxHashMap<(u64, i64), u16>> = Default::default();
    let wide2: std::sync::Mutex<Vec<(u64, u32, u32, u16)>> = Default::default();
    let eraw = par(
        ech.len(),
        nt,
        || (0..NB).map(|_| Vec::<ERaw>::with_capacity(ecap)).collect::<Vec<_>>(),
        |i, bk| {
            
            for_rows::<7>(ech[i], |f| {
                if f[0].is_empty() {
                    return;
                }
                let ingest = parse_ts_c(f[3], dtab).unwrap_or(i64::MAX);
                let user = f[1];
                let mut ok = !user.is_empty() && !f[5].is_empty();
                let ts = parse_ts_c(f[2], dtab);
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
                    (ukey(user), products.get(f[5]).unwrap_or(unknown), (tsv - base) as u32)
                } else {
                    (0, 0, 0)
                };
                let ekey = pack(b'e', f[0]).unwrap_or_else(|| eids.key(f[0]));
                let mut c = cat as u32;
                if c >= ESC {
                    if pkey != 0 && price != 0 {
                        wide1.lock().unwrap().insert((ekey, ingest), cat);
                    }
                    c = ESC;
                }
                bk[(hash64(ekey) >> (64 - NBB)) as usize].push(ERaw { ekey, ingest, pkey, ts: tsr, pc: price | c << 24 });
            });
        },
    );
    t("parse_events", &mut t0);

    // ---------------- dedup -> person buckets
    let pcap = ech.iter().map(|c| c.len()).sum::<usize>() / 70 / nt / NB + 64;
    let evb = par(
        NB,
        nt,
        || ((0..NB).map(|_| Vec::<Ev>::with_capacity(pcap)).collect::<Vec<_>>(), Vec::<u32>::new()),
        |b, (out, tab)| {
            let mut v: Vec<ERaw> = Vec::with_capacity(eraw.iter().map(|t| t[b].len()).sum());
            for tv in &eraw {
                v.extend_from_slice(&tv[b]);
            }
            // hash dedup: earliest ingest wins (equal ingest copies are identical); losers get pkey 0
            let sz = (v.len() * 2).next_power_of_two().max(64);
            let mask = sz - 1;
            tab.clear();
            tab.resize(sz, u32::MAX);
            for i in 0..v.len() {
                let k = v[i].ekey;
                let mut h = (hash64(k) >> 16) as usize & mask;
                loop {
                    let j = tab[h];
                    if j == u32::MAX {
                        tab[h] = i as u32;
                        break;
                    }
                    let j = j as usize;
                    if v[j].ekey == k {
                        if v[i].ingest < v[j].ingest {
                            v[j].pkey = 0;
                            tab[h] = i as u32;
                        } else {
                            v[i].pkey = 0;
                        }
                        break;
                    }
                    h = (h + 1) & mask;
                }
            }
            const D: usize = 16;
            for i in 0..v.len() {
                if i + D < v.len() {
                    ltab.prefetch(v[i + D].pkey);
                }
                let r = v[i];
                if r.pkey != 0 {
                    let pkey = ltab.get(r.pkey);
                    if r.pc >> 24 == ESC && r.pc & 0xFF_FFFF != 0 {
                        let c = wide1.lock().unwrap()[&(r.ekey, r.ingest)];
                        wide2.lock().unwrap().push((pkey, r.ts, r.pc & 0xFF_FFFF, c));
                    }
                    out[(hash64(pkey) >> (64 - NBB)) as usize].push(Ev { pkey, ts: r.ts, pc: r.pc });
                }
            }
        },
    );
    let evb: Vec<Vec<Vec<Ev>>> = evb.into_iter().map(|x| x.0).collect();
    drop(eraw);
    t("dedup_events", &mut t0);

    // ---------------- clicks
    let ckb = par(
        cch.len(),
        nt,
        || ((0..NB).map(|_| Vec::<Ck>::new()).collect::<Vec<_>>(), Vec::<Ck>::new()),
        |i, (out, tmp)| {
            tmp.clear();
            
            for_rows::<4>(cch[i], |f| {
                if f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                let ts = match parse_ts_c(f[3], dtab) {
                    Some(t) if t >= ws - WEEK && t < we => t,
                    _ => return,
                };
                let camp = match camp_idx.get(f[2]) {
                    Some(c) => c,
                    None => return,
                };
                tmp.push(Ck { pkey: ukey(f[1]), cid: f[0], ts: (ts - base) as u32, camp });
            });
            const D: usize = 16;
            for q in 0..tmp.len() {
                if q + D < tmp.len() {
                    ltab.prefetch(tmp[q + D].pkey);
                }
                let mut c = tmp[q];
                c.pkey = ltab.get(c.pkey);
                out[(hash64(c.pkey) >> (64 - NBB)) as usize].push(c);
            }
        },
    );
    let ckb: Vec<Vec<Vec<Ck>>> = ckb.into_iter().map(|x| x.0).collect();
    t("parse_clicks", &mut t0);
    // ---------------- sessionize + attribute
    let nc1 = ncamp + 1;
    // wide-category purchases per person bucket, in (person, ts) order
    let mut wide: Vec<Vec<(u64, u32, u32, u16)>> = (0..NB).map(|_| Vec::new()).collect();
    for w in wide2.into_inner().unwrap() {
        wide[(hash64(w.0) >> (64 - NBB)) as usize].push(w);
    }
    wide.iter_mut().for_each(|v| v.sort_unstable());
    #[derive(Clone, Copy)]
    struct CS {
        ts: u32,
        camp: u32,
        idx: u32,
    }
    #[derive(Default)]
    struct SW {
        tab: Vec<u32>,
        prov: Vec<u64>,
        rank: Vec<u32>,
        order: Vec<u32>,
        lid: Vec<u32>,
        eoff: Vec<u32>,
        coff: Vec<u32>,
        es: Vec<u64>,
        cs: Vec<CS>,
    }
    let results = par(
        NB,
        nt,
        || (Vec::<URow>::new(), vec![[0i64; 4]; nc1 * ndays], vec![0u32; nc1], vec![0u32; nc1], Vec::<u32>::new(), vec![0i64; ncat], SW::default()),
        |b, (urows, carr, cnt, pos, act, catsum, sw)| {
            let ne: usize = evb.iter().map(|t| t[b].len()).sum();
            if ne == 0 {
                return;
            }
            // distinct persons of this bucket -> provisional ids
            let sz = (ne * 2).next_power_of_two().max(64);
            let mask = sz - 1;
            sw.tab.clear();
            sw.tab.resize(sz, u32::MAX);
            sw.prov.clear();
            sw.lid.clear();
            let (tab, prov) = (&mut sw.tab, &mut sw.prov);
            let mut lookup = |k: u64, insert: bool| -> u32 {
                let mut h = (hash64(k) >> 16) as usize & mask;
                loop {
                    let j = tab[h];
                    if j == u32::MAX {
                        if !insert {
                            return u32::MAX;
                        }
                        tab[h] = prov.len() as u32;
                        prov.push(k);
                        return tab[h];
                    }
                    if prov[j as usize] == k {
                        return j;
                    }
                    h = (h + 1) & mask;
                }
            };
            for tv in &evb {
                for x in &tv[b] {
                    let l = lookup(x.pkey, true);
                    sw.lid.push(l);
                }
            }
            let nck: usize = ckb.iter().map(|t| t[b].len()).sum();
            let mut clid: Vec<u32> = Vec::with_capacity(nck);
            for tv in &ckb {
                for x in &tv[b] {
                    clid.push(lookup(x.pkey, false));
                }
            }
            let np = sw.prov.len();
            // rank persons by key so rows come out in person-key order
            sw.order.clear();
            sw.order.extend(0..np as u32);
            {
                let prov = &sw.prov;
                sw.order.sort_unstable_by_key(|&i| prov[i as usize]);
            }
            sw.rank.clear();
            sw.rank.resize(np, 0);
            for (r, &i) in sw.order.iter().enumerate() {
                sw.rank[i as usize] = r as u32;
            }
            // counting sort events (ts<<32|pc) and clicks by person rank
            sw.eoff.clear();
            sw.eoff.resize(np + 1, 0);
            sw.coff.clear();
            sw.coff.resize(np + 1, 0);
            for l in sw.lid.iter_mut() {
                *l = sw.rank[*l as usize];
                sw.eoff[*l as usize + 1] += 1;
            }
            for l in clid.iter_mut() {
                if *l != u32::MAX {
                    *l = sw.rank[*l as usize];
                    sw.coff[*l as usize + 1] += 1;
                }
            }
            for q in 0..np {
                sw.eoff[q + 1] += sw.eoff[q];
                sw.coff[q + 1] += sw.coff[q];
            }
            sw.es.clear();
            sw.es.resize(ne, 0);
            let ncs = sw.coff[np] as usize;
            sw.cs.clear();
            sw.cs.resize(ncs, CS { ts: 0, camp: 0, idx: 0 });
            let ck: Vec<Ck> = {
                let mut v = Vec::with_capacity(nck);
                for tv in &ckb {
                    v.extend_from_slice(&tv[b]);
                }
                v
            };
            {
                let mut wp: Vec<u32> = sw.eoff[..np].to_vec();
                let mut i = 0;
                for tv in &evb {
                    for x in &tv[b] {
                        let l = sw.lid[i] as usize;
                        sw.es[wp[l] as usize] = (x.ts as u64) << 32 | x.pc as u64;
                        wp[l] += 1;
                        i += 1;
                    }
                }
                wp.copy_from_slice(&sw.coff[..np]);
                for (i, x) in ck.iter().enumerate() {
                    let l = clid[i];
                    if l != u32::MAX {
                        sw.cs[wp[l as usize] as usize] = CS { ts: x.ts, camp: x.camp, idx: i as u32 };
                        wp[l as usize] += 1;
                    }
                }
            }
            let mut wi = 0;
            for r in 0..np {
                let pkey = sw.prov[sw.order[r] as usize];
                let e = &mut sw.es[sw.eoff[r] as usize..sw.eoff[r + 1] as usize];
                let c = &mut sw.cs[sw.coff[r] as usize..sw.coff[r + 1] as usize];
                if e.len() <= 24 {
                    for x in 1..e.len() {
                        let v = e[x];
                        let mut y = x;
                        while y > 0 && e[y - 1] > v {
                            e[y] = e[y - 1];
                            y -= 1;
                        }
                        e[y] = v;
                    }
                } else {
                    e.sort_unstable();
                }
                c.sort_unstable_by_key(|x| x.ts);
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
                    let start = (e[k] >> 32) as i64;
                    let mut j = k + 1;
                    let mut prev = start;
                    while j < e.len() && ((e[j] >> 32) as i64) - prev <= GAP {
                        prev = (e[j] >> 32) as i64;
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
                    for &x in &e[k..j] {
                        let xpc = x as u32;
                        let mut price = (xpc & 0xFF_FFFF) as i64;
                        let mut cat = (xpc >> 24) as usize;
                        if cat == ESC as usize && price != 0 {
                            let w = wide[b][wi];
                            wi += 1;
                            debug_assert!(w.0 == pkey && w.1 == (x >> 32) as u32);
                            (price, cat) = (w.2 as i64, w.3 as usize);
                        }
                        if price != 0 {
                            rev += price;
                            catsum[cat] += price;
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
                        let mut best = q;
                        while q > lo && c[q - 1].ts == tt {
                            q -= 1;
                            if ck[c[q].idx as usize].cid < ck[c[best].idx as usize].cid {
                                best = q;
                            }
                        }
                        c[best].camp as usize
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
    if let Ok(st) = std::fs::read_to_string("/proc/self/status") {
        let kb = |k: &str| st.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0) / 1024;
        eprintln!("mem peak {} MiB (anon {} MiB, file-mapped {} MiB)", kb("VmHWM:"), kb("RssAnon:"), kb("RssFile:"));
    }
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

    #[test]
    fn ts_cache_matches() {
        let seeds: [&[u8]; 8] = [b"2026-09-16T18:24:20Z", b"2028-02-29T23:59:59+05:30", b"2027-02-29T00:00:00Z",
            b"1791071890000", b"0001791071890000", b"2026-12-31T00:00:00-00:00", b"99999999999999999999", b"2026-09-16T18:24:20z"];
        let alpha = b"0123456789-:TZ+ z.";
        let mut x: u64 = 777;
        let dd = |y, m, d| days_from_civil(y, m, d);
        let tabs = [DateTab::new(dd(2026, 9, 1), dd(2026, 10, 3)), DateTab::new(dd(2028, 2, 10), dd(2028, 3, 20)),
            DateTab::new(dd(2026, 12, 20), dd(2027, 3, 5))];
        for it in 0..6_000_000 {
            let tc = &tabs[it % 3];
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let mut v = seeds[(x % 8) as usize].to_vec();
            let k = (x >> 8) % 4;
            for _ in 0..k {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let p = (x as usize >> 3) % (v.len() + 1);
                match x % 3 {
                    0 if p < v.len() => v[p] = alpha[(x >> 20) as usize % alpha.len()],
                    1 if p < v.len() => { v.remove(p); }
                    _ => v.insert(p.min(v.len()), alpha[(x >> 20) as usize % alpha.len()]),
                }
            }
            assert_eq!(parse_ts(&v), parse_ts_c(&v, tc), "it={it} {:?}", String::from_utf8_lossy(&v));
        }
    }

    #[test]
    fn pack_swar_matches() {
        fn slow(prefix: u8, b: &[u8]) -> Option<u64> {
            let n = b.len();
            if n < 2 || n > 9 || b[0] != prefix { return None; }
            let mut v = 0u64;
            for &c in &b[1..] {
                if !(c.is_ascii_digit() || c.is_ascii_lowercase()) { return None; }
                v = (v << 8) | c as u64;
            }
            Some(v << (8 * (9 - n)))
        }
        let mut x: u64 = 99;
        let buf: Vec<u8> = (0..20000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            let a = b"u0123456789abcxyz/:`{AZ \x80\xff"; a[(x % a.len() as u64) as usize] }).collect();
        for i in 0..buf.len() - 12 {
            for n in 0..11 {
                let mut b = buf[i..i + n].to_vec();
                if !b.is_empty() { b[0] = b'u'; }
                let mut pad = vec![b'x'; 16]; pad.extend_from_slice(&b);
                let f = &pad[16..];
                assert_eq!(slow(b'u', f), pack(b'u', f), "{:?}", f);
            }
        }
    }
}
