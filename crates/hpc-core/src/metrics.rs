#![cfg(feature = "metrics")]

use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

/* ───────────── Roh‑Latenzen ─────────────────────────── */

static TIMES: Lazy<Mutex<Vec<(&'static str, u128)>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

/// Im Wrapper aufrufen: `record("enqueue_write", Instant::now());`
pub fn record(name: &'static str, start: Instant) {
    let dur = start.elapsed().as_micros();
    TIMES.lock().unwrap().push((name, dur));
}

/* ───────────── Buffer‑Allokationen ───────────────────── */

pub static ALLOCS:      AtomicUsize = AtomicUsize::new(0);
pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

/* ───────────── Zusammenfassung ausgeben ─────────────── */

/// Am Programmende aufrufen, z. B. in `main()`
pub fn summary() {
    /* API‑Latenzen gruppiert */
    let mut map: HashMap<&str, Vec<u128>> = HashMap::new();
    {
        let mut times = TIMES.lock().unwrap();
        for (name, us) in times.drain(..) {
            map.entry(name).or_default().push(us);
        }
    }

    println!("── metrics summary ──");
    for (name, mut v) in map {
        v.sort_unstable();
        let mean = v.iter().sum::<u128>() / v.len() as u128;
        let p95  = v[((v.len() * 95) / 100).saturating_sub(1)];
        println!("{:<18} mean={:>5} µs   p95={:>5} µs", name, mean, p95);
    }

    /* Allokations‑Zähler */
    let allocs = ALLOCS.load(Ordering::Relaxed);
    let bytes  = ALLOC_BYTES.load(Ordering::Relaxed);
    println!("GPU allocations: {}   ({} MiB)", allocs, bytes / 1024 / 1024);
}
