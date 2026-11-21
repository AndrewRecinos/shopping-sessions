// Diagnostic probe: environment info, sub-stage timers, memory-bandwidth microbenchmark.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

fn rd(p: &str) -> String {
    std::fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_else(|_| "-".into())
}

#[inline(always)]
pub fn tsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Sub-stage accumulators (TSC ticks summed over threads).
pub const NAMES: [&str; 14] = [
    "dedup.gather", "dedup.hash", "dedup.lookup+scatter", "clicks.parse", "clicks.lookup+scatter",
    "sess.person_hash", "sess.rank+count_sort", "sess.person_sorts", "sess.sessions+credit", "", "", "", "", "",
];
pub static ACC: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
#[inline(always)]
pub fn acc(i: usize, t: &mut u64) {
    let n = tsc();
    ACC[i].fetch_add(n.wrapping_sub(*t), Relaxed);
    *t = n;
}
#[inline(always)]
pub fn lap(l: &mut u64, t: &mut u64) {
    let n = tsc();
    *l += n.wrapping_sub(*t);
    *t = n;
}

// par() statistics since the last stage print
pub static PAR_WALL: AtomicU64 = AtomicU64::new(0); // ns
pub static PAR_BUSY: AtomicU64 = AtomicU64::new(0); // ns summed over threads
pub static PAR_SPAWN: AtomicU64 = AtomicU64::new(0); // max ns from par() call to thread first work
pub static PAR_N: AtomicU64 = AtomicU64::new(0);

pub fn par_suffix(nt: usize) -> String {
    let w = PAR_WALL.swap(0, Relaxed);
    let b = PAR_BUSY.swap(0, Relaxed);
    let s = PAR_SPAWN.swap(0, Relaxed);
    let n = PAR_N.swap(0, Relaxed);
    if n == 0 || w == 0 {
        return String::new();
    }
    format!("  [par x{n} eff {:.0}% spawn<={:.2}ms]", 100.0 * b as f64 / (w as f64 * nt as f64), s as f64 / 1e6)
}

pub struct Clock {
    pub i0: Instant,
    pub c0: u64,
}
impl Clock {
    pub fn new() -> Self {
        Clock { i0: Instant::now(), c0: tsc() }
    }
    pub fn ticks_per_ns(&self) -> f64 {
        (tsc() - self.c0) as f64 / self.i0.elapsed().as_nanos().max(1) as f64
    }
}

pub fn print_acc(clk: &Clock, nt: usize) {
    let tpn = clk.ticks_per_ns();
    let mut s = String::from("substages (thread-avg ms):");
    for (i, n) in NAMES.iter().enumerate() {
        if n.is_empty() {
            continue;
        }
        let v = ACC[i].load(Relaxed) as f64 / tpn / 1e6 / nt as f64;
        s += &format!(" {n} {v:.1}");
        if i == 4 {
            eprintln!("{s}");
            s = String::from("  ");
        }
    }
    eprintln!("{s}  (tsc {tpn:.2} GHz)");
}

/// Seconds from process start (kernel) to now, plus clock tick size.
pub fn start_to_now() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut ts) };
    let now = ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9;
    let st = rd("/proc/self/stat");
    let f: Vec<&str> = st.rsplit_once(')').map(|x| x.1).unwrap_or("").split_whitespace().collect();
    let tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    let start = f.get(19).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) / tck;
    now - start
}

pub fn sysinfo(nt: usize, data: &std::path::Path, out: &std::path::Path) {
    let ci = rd("/proc/cpuinfo");
    let field = |k: &str| ci.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_once(':')).map(|x| x.1.trim().to_string()).unwrap_or("-".into());
    let nlog = ci.lines().filter(|l| l.starts_with("processor")).count();
    let mut cores = std::collections::BTreeSet::new();
    let (mut pid, mut socks) = (String::new(), std::collections::BTreeSet::new());
    for l in ci.lines() {
        if let Some((k, v)) = l.split_once(':') {
            match k.trim() {
                "physical id" => {
                    pid = v.trim().to_string();
                    socks.insert(pid.clone());
                }
                "core id" => {
                    cores.insert((pid.clone(), v.trim().to_string()));
                }
                _ => {}
            }
        }
    }
    eprintln!("cpu: {} | logical {nlog} phys-cores {} sockets {} | siblings {} cpu-cores {} | {} MHz | cache {}",
        field("model name"), cores.len(), socks.len(), field("siblings"), field("cpu cores"), field("cpu MHz"), field("cache size"));
    let flags = field("flags");
    let fl: Vec<&str> = flags.split_whitespace().collect();
    let want = ["avx", "avx2", "bmi1", "bmi2", "fma", "sse4_2", "popcnt", "erms", "fsrm", "hypervisor", "pdpe1gb", "vaes", "vpclmulqdq", "sha_ni", "lzcnt", "movdiri", "amx_tile"];
    let have: Vec<&str> = fl.iter().copied().filter(|f| want.contains(f) || f.starts_with("avx512")).collect();
    let missing: Vec<&str> = want.iter().copied().filter(|w| !fl.contains(w)).collect();
    eprintln!("flags: {} | missing: {}", have.join(" "), missing.join(" "));
    eprintln!("cpu: vendor {} family {} model {} stepping {} microcode {} | bugs: {}", field("vendor_id"), field("cpu family"), field("model\t"), field("stepping"), field("microcode"), field("bugs"));
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let numa = std::fs::read_dir("/sys/devices/system/node").map(|d| d.filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("node")).count()).unwrap_or(0);
    eprintln!("kernel {} | page {ps} B | THP enabled {} defrag {} | hugepagesize {} | numa nodes {numa}",
        rd("/proc/sys/kernel/osrelease"), rd("/sys/kernel/mm/transparent_hugepage/enabled"), rd("/sys/kernel/mm/transparent_hugepage/defrag"),
        rd("/proc/meminfo").lines().find(|l| l.starts_with("Hugepagesize")).map(|l| l.split_whitespace().skip(1).collect::<Vec<_>>().join("")).unwrap_or("-".into()));
    let mi = rd("/proc/meminfo");
    let mf = |k: &str| mi.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0) / 1024;
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    let aff: Vec<usize> = (0..1024).filter(|&c| unsafe { libc::CPU_ISSET(c, &set) }).collect();
    let affs = if aff.len() > 16 { format!("{}..{}", aff[0], aff[aff.len() - 1]) } else { format!("{aff:?}") };
    eprintln!("affinity {} cpus {affs} | avail_par {} | threads used {nt} | MemTotal {} MiB MemAvail {} MiB | uptime {}",
        aff.len(), std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0), mf("MemTotal:"), mf("MemAvailable:"), rd("/proc/uptime"));
    let cg = "/sys/fs/cgroup";
    if std::path::Path::new(&format!("{cg}/cgroup.controllers")).exists() {
        eprintln!("cgroup2: cpu.max {} cpu.weight {} cpuset {} | memory.max {} high {} swap.max {} | pids.max {}",
            rd(&format!("{cg}/cpu.max")), rd(&format!("{cg}/cpu.weight")), rd(&format!("{cg}/cpuset.cpus.effective")),
            rd(&format!("{cg}/memory.max")), rd(&format!("{cg}/memory.high")), rd(&format!("{cg}/memory.swap.max")), rd(&format!("{cg}/pids.max")));
        let ps = rd(&format!("{cg}/cpu.stat"));
        eprintln!("cgroup2 cpu.stat: {}", ps.lines().filter(|l| l.contains("throttled") || l.starts_with("nr_periods")).collect::<Vec<_>>().join(", "));
    } else {
        eprintln!("cgroup1: cfs_quota {} period {} | mem limit {} | cpuset {} | self {}",
            rd(&format!("{cg}/cpu/cpu.cfs_quota_us")), rd(&format!("{cg}/cpu/cpu.cfs_period_us")), rd(&format!("{cg}/memory/memory.limit_in_bytes")),
            rd(&format!("{cg}/cpuset/cpuset.cpus")), rd("/proc/self/cgroup").replace('\n', ";"));
    }
    // filesystems holding the data/output
    let mnt = rd("/proc/self/mountinfo");
    let fs_of = |p: &std::path::Path| -> String {
        let p = p.to_string_lossy();
        let mut best = ("".to_string(), "-".to_string());
        for l in mnt.lines() {
            let f: Vec<&str> = l.split_whitespace().collect();
            let Some(sep) = f.iter().position(|x| *x == "-") else { continue };
            let mp = f.get(4).copied().unwrap_or("");
            if p.starts_with(mp) && mp.len() >= best.0.len() {
                best = (mp.to_string(), format!("{} {} ({})", mp, f.get(sep + 1).unwrap_or(&"-"), f.get(5).unwrap_or(&"-")));
            }
        }
        best.1
    };
    let envs: Vec<String> = std::env::vars().map(|(k, _)| k).collect();
    eprintln!("fs: data {} | out {} | env: {}", fs_of(data), fs_of(out), { let e = envs.join(","); e.chars().take(300).collect::<String>() });
}

pub fn proc_stats(tag: &str) {
    let st = rd("/proc/self/stat");
    let f: Vec<&str> = st.rsplit_once(')').map(|x| x.1).unwrap_or("").split_whitespace().collect();
    let tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    let g = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    let s = rd("/proc/self/status");
    let k = |n: &str| s.lines().find(|l| l.starts_with(n)).and_then(|l| l.split_whitespace().nth(1)).unwrap_or("-").to_string();
    let sr = rd("/proc/self/smaps_rollup");
    let thp = sr.lines().find(|l| l.starts_with("AnonHugePages")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()).unwrap_or(0) / 1024;
    eprintln!("{tag}: utime {:.2}s stime {:.2}s minflt {} majflt {} | ctxsw vol {} invol {} | HWM {} MiB anon {} MiB file {} MiB THP {thp} MiB",
        g(11) / tck, g(12) / tck, g(7), g(9), k("voluntary_ctxt_switches:"), k("nonvoluntary_ctxt_switches:"),
        k("VmHWM:").parse::<u64>().unwrap_or(0) / 1024, k("RssAnon:").parse::<u64>().unwrap_or(0) / 1024, k("RssFile:").parse::<u64>().unwrap_or(0) / 1024);
}

// ---------------------------------------------------------------- bandwidth

struct Buf(*mut u64);
unsafe impl Sync for Buf {}
unsafe impl Send for Buf {}
impl Buf {
    fn p(&self) -> *mut u64 {
        self.0
    }
}

fn on_threads<R: Send>(nt: usize, f: impl Fn(usize) -> R + Sync) -> (f64, Vec<R>) {
    let t = Instant::now();
    let r = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt).map(|i| { let f = &f; sc.spawn(move || f(i)) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    (t.elapsed().as_secs_f64(), r)
}

pub fn membench(nt: usize) {
    const BYTES: usize = 1 << 30;
    let n = BYTES / 8;
    let p = unsafe { libc::mmap(std::ptr::null_mut(), BYTES, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
    if p == libc::MAP_FAILED {
        eprintln!("membench: mmap failed");
        return;
    }
    unsafe { libc::madvise(p, BYTES, libc::MADV_HUGEPAGE) };
    let b = Buf(p as *mut u64);
    let gb = BYTES as f64 / 1e9;
    let seg = |i: usize, k: usize| -> &mut [u64] { let s = n / k; unsafe { std::slice::from_raw_parts_mut(b.p().add(i * s), s) } };
    let write = |k: usize, v: u64| on_threads(k, |i| { for (j, x) in seg(i, k).iter_mut().enumerate() { *x = v ^ j as u64; } }).0;
    let read = |k: usize| on_threads(k, |i| { let mut s = 0u64; for &x in seg(i, k).iter() { s = s.wrapping_add(x); } std::hint::black_box(s) }).0;
    let first = write(nt, 1);
    let w1 = write(1, 2);
    let wn = write(nt, 3);
    let r1 = read(1);
    let rn = read(nt);
    eprintln!("membench 1GiB (THP-advised): first-touch write {nt}T {:.2} GB/s | seq write 1T {:.2} {nt}T {:.2} GB/s | seq read 1T {:.2} {nt}T {:.2} GB/s",
        gb / first, gb / w1, gb / wn, gb / r1, gb / rn);
    // random independent 8-byte reads
    const R: usize = 1 << 24;
    let rnd = |k: usize| on_threads(k, |i| {
        let all = unsafe { std::slice::from_raw_parts(b.p(), n) };
        let mut x = 0x9E3779B97F4A7C15u64 ^ (i as u64 + 1) * 0x1234567;
        let mut s = 0u64;
        for _ in 0..R / k {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            s = s.wrapping_add(all[(x as usize) & (n - 1)]);
        }
        std::hint::black_box(s)
    }).0;
    let t1 = rnd(1);
    let tn = rnd(nt);
    let reads_n = (R / nt) * nt;
    eprintln!("random 8B reads over 1GiB: 1T {:.1} M/s ({:.2} GB/s) | {nt}T {:.1} M/s ({:.2} GB/s)",
        R as f64 / t1 / 1e6, R as f64 * 8.0 / t1 / 1e9, reads_n as f64 / tn / 1e6, reads_n as f64 * 8.0 / tn / 1e9);
    // dependent pointer chase over 2^21 cache lines (128 MiB span) and 2^14 lines (1 MiB, ~L2)
    for &lines in &[1usize << 21, 1 << 14] {
        let all = unsafe { std::slice::from_raw_parts_mut(b.p(), n) };
        let mut perm: Vec<u32> = (0..lines as u32).collect();
        let mut x = 88172645463325252u64;
        for i in (1..lines).rev() {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            perm.swap(i, (x % (i as u64 + 1)) as usize);
        }
        for i in 0..lines {
            all[perm[i] as usize * 8] = perm[(i + 1) % lines] as u64 * 8;
        }
        let steps = 1usize << 22;
        let t = Instant::now();
        let mut c = perm[0] as usize * 8;
        for _ in 0..steps {
            c = all[c] as usize;
        }
        std::hint::black_box(c);
        eprintln!("pointer chase {} MiB span: {:.1} ns/load", lines * 64 >> 20, t.elapsed().as_nanos() as f64 / steps as f64);
    }
    unsafe { libc::munmap(p, BYTES) };
}
