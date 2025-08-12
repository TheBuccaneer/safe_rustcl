// examples/stm_abort.rs
//
// Minimaler Stub: parst CLI (--threads, --conflict, --duration, --seed),
// simuliert eine STM-Workload (Dummy), deterministisch per Seed.
// Ziel für heute: baut & läuft. Logging der Abort-Events folgt im nächsten Schritt.

use std::env;
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

// ---- CLI ----

#[derive(Clone, Copy, Debug)]
enum Conflict {
    Low,
    Med,
    High,
}
impl FromStr for Conflict {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Ok(Conflict::Low),
            "med" | "medium" => Ok(Conflict::Med),
            "high" => Ok(Conflict::High),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
struct Config {
    threads: usize,
    conflict: Conflict,
    duration_s: u64,
    seed: u64,
}

fn parse_args() -> Config {
    let mut threads = 4usize;
    let mut conflict = Conflict::Low;
    let mut duration_s = 5u64;
    let mut seed = 1u64;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => {
                if let Some(v) = args.next() {
                    threads = v.parse().unwrap_or(4);
                }
            }
            "--conflict" => {
                if let Some(v) = args.next() {
                    conflict = v.parse().unwrap_or(Conflict::Low);
                }
            }
            "--duration" => {
                if let Some(v) = args.next() {
                    duration_s = v.parse().unwrap_or(5);
                }
            }
            "--seed" => {
                if let Some(v) = args.next() {
                    seed = v.parse().unwrap_or(1);
                }
            }
            _ => {}
        }
    }
    Config { threads, conflict, duration_s, seed }
}

// ---- sehr einfacher, deterministischer PRNG ----
#[derive(Clone)]
struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) / (u32::MAX as f32)
    }
}

// ---- Dummy-STM-Workload ----
// Wir simulieren Transactions mit Konfliktwahrscheinlichkeit je nach Level.
// Heute nur Zähler & deterministisches Verhalten. Abort-Logging folgt separat.

fn main() {
    let cfg = parse_args();
    eprintln!(
        "stm_abort: threads={}, conflict={:?}, duration={}s, seed={}",
        cfg.threads, cfg.conflict, cfg.duration_s, cfg.seed
    );

    // Konfliktwahrscheinlichkeit grob (wird später feiner justiert)
    let p_conflict = match cfg.conflict {
        Conflict::Low => 0.02_f32,
        Conflict::Med => 0.15_f32,
        Conflict::High => 0.40_f32,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let aborts = Arc::new(AtomicU64::new(0));
    let commits = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let mut handles = Vec::with_capacity(cfg.threads);

    for i in 0..cfg.threads {
        let stop = stop.clone();
        let aborts = aborts.clone();
        let commits = commits.clone();
        // pro Thread deterministischer Seed
        let thread_seed = cfg.seed ^ (((i as u64) + 1) << 32) ^ 0x9E3779B97F4A7C15u64;
        let mut rng = XorShift64::new(thread_seed);

        let h = thread::spawn(move || {
            // Kleine "kritische Sektion"
            while !stop.load(Ordering::Relaxed) {
                // Transaktion beginnt
                // Arbeit simulieren
                spin_for_ns(1500 + (rng.next_u32() % 1500) as u64);

                // Konfliktsampling
                let r = rng.next_f32();
                if r < p_conflict {
                    // Abort
                    aborts.fetch_add(1, Ordering::Relaxed);
                    // kurzer Backoff (deterministisch)
                    spin_for_ns(10_000 + ((i as u64) * 1_000));
                    // retry (direkt weiter zur nächsten Iteration)

                    #[cfg(feature = "memtrace")]
                    hpc_core::memtracer::trace_abort(
                    /*tx_id*/ 0,
                    /*cause*/ "conflict",
                    /*retries*/ 1,
                /*conflict_sz*/ 1,
                    /*abort_token*/ "stm",
);
                    continue;
                } else {
                    // Commit
                    commits.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        handles.push(h);
    }

    // Laufzeit begrenzen
    let dur = Duration::from_secs(cfg.duration_s);
    while start.elapsed() < dur {
        thread::sleep(Duration::from_millis(5));
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }

    let a = aborts.load(Ordering::Relaxed);
    let c = commits.load(Ordering::Relaxed);

    println!("STM run finished.");
    println!("aborts_total: {}", a);
    println!("commits_total: {}", c);

    // Optional: Summary schreiben lassen, falls Feature aktiv (kein Abort-CSV hier!)
    #[cfg(feature = "memtrace")]
    {
        // ruft deine bereits implementierte Summary-Erzeugung auf
        hpc_core::memtracer::flush_csv();
        println!("memtrace.csv / memtrace_summary.txt geschrieben (falls Events vorhanden).");
    }
}

// sehr kleiner, portabler Busy-Wait (für deterministische Mikro-Sleeps)
#[inline(always)]
fn spin_for_ns(nanos: u64) {
    let start = Instant::now();
    // coarse spin (keine systemgenauen Nanos)
    while start.elapsed().as_nanos() < nanos as u128 {
        core::hint::spin_loop();
    }
}
