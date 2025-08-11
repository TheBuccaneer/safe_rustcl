#![cfg(feature = "memtrace")]

mod copytoken;
mod aborttoken;

pub use copytoken::*;
pub use aborttoken::*;

// Gemeinsame Hilfstypen (Dir, LOG, T0 etc.)
use once_cell::sync::Lazy;
use std::{
    fs::File,
    io::Write,
    sync::{atomic::{AtomicBool, Ordering}, Mutex},
    time::Instant,
};

#[derive(Clone, Copy, Debug)]
pub enum Dir {
    H2D,
    D2H,
    Kernel,
}
impl Dir {
    pub fn as_str(self) -> &'static str {
        match self {
            Dir::H2D => "H2D",
            Dir::D2H => "D2H",
            Dir::Kernel => "KRN",
        }
    }
}

pub static T0: Lazy<Instant> = Lazy::new(Instant::now);
pub static AUTO_TRACE: AtomicBool = AtomicBool::new(true);

#[inline]
pub fn enable_auto_trace() {
    AUTO_TRACE.store(true, Ordering::Relaxed);
}
#[inline]
pub fn disable_auto_trace() {
    AUTO_TRACE.store(false, Ordering::Relaxed);
}
#[inline]
pub fn is_auto_trace_enabled() -> bool {
    AUTO_TRACE.load(Ordering::Relaxed)
}

/// Log-Eintrag
#[derive(Debug)]
pub struct Record {
    t_start_us: u64,
    t_end_us: u64,
    bytes: usize,
    dir: Dir,
    idle_us: u64,
    abort_token: Option<String>,
    // NEU:
    phase: Phase,
    tx_id: Option<u64>,
    cause: Option<String>,
    retries: Option<u32>,
    conflict_sz: Option<usize>,
}

pub static LOG: Lazy<Mutex<Vec<Record>>> = Lazy::new(|| Mutex::new(Vec::with_capacity(4096)));

// crates/hpc-core/src/memtracer/mod.rs

#[cfg(feature = "memtrace")]
pub fn flush_csv() {
    use std::fs::File;
    use std::io::Write;

    let log = LOG.lock().unwrap();

    // A) Transfer/Kernel Events → memtrace.csv (nur relevante Felder)
    let mut f = File::create("memtrace.csv").expect("memtrace.csv nicht anlegbar");
    
    // Header nur mit relevanten Feldern für Transfer/Kernel
    writeln!(
        f,
        "t_start_us,t_end_us,bytes,dir,idle_us,abort_token,phase"
    ).unwrap();

    for r in log.iter().filter(|r| !matches!(r.phase, Phase::Abort)) {
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            r.t_start_us,
            r.t_end_us,
            r.bytes,
            r.dir.as_str(),
            r.idle_us,
            r.abort_token.as_deref().unwrap_or(""),
            r.phase.as_str()
        )
        .unwrap();
    }
    
    // Statistiken für normale Events
    let transfer_count = log.iter()
        .filter(|r| matches!(r.phase, Phase::Transfer))
        .count();
    let kernel_count = log.iter()
        .filter(|r| matches!(r.phase, Phase::Kernel))
        .count();
    
    println!("✓ memtrace.csv geschrieben ({} transfers, {} kernels)", 
             transfer_count, kernel_count);

    // B) Abort Events → memtrace_abort.csv (nur wenn Aborts vorhanden)
    let abort_events: Vec<_> = log.iter()
        .filter(|r| matches!(r.phase, Phase::Abort))
        .collect();
    
    if !abort_events.is_empty() {
        let mut fa = File::create("memtrace_abort.csv")
            .expect("memtrace_abort.csv nicht anlegbar");
        
        // Header mit allen Abort-relevanten Feldern
        writeln!(
            fa,
            "t_start_us,t_end_us,tx_id,cause,retries,conflict_sz,idle_us,abort_token"
        ).unwrap();

        for r in abort_events.iter() {
            writeln!(
                fa,
                "{},{},{},{},{},{},{},{}",
                r.t_start_us,
                r.t_end_us,
                r.tx_id.map(|v| v.to_string()).unwrap_or_default(),
                r.cause.as_deref().unwrap_or(""),
                r.retries.map(|v| v.to_string()).unwrap_or_default(),
                r.conflict_sz.map(|v| v.to_string()).unwrap_or_default(),
                r.idle_us,
                r.abort_token.as_deref().unwrap_or("")
            )
            .unwrap();
        }
        
        println!("✓ memtrace_abort.csv geschrieben ({} aborts)", abort_events.len());
    }

    // C) Optional: Summary Statistics → memtrace_summary.txt
    if log.len() > 0 {
        let mut fs = File::create("memtrace_summary.txt")
            .expect("memtrace_summary.txt nicht anlegbar");
        
        // Zeitstatistiken
        let total_time = log.last().unwrap().t_end_us - log.first().unwrap().t_start_us;
        let total_idle: u64 = log.iter().map(|r| r.idle_us).sum();
        let utilization = 100.0 * (1.0 - (total_idle as f64 / total_time as f64));
        
        // Transfer-Statistiken
        let h2d_bytes: usize = log.iter()
            .filter(|r| matches!(r.dir, Dir::H2D))
            .map(|r| r.bytes)
            .sum();
        let d2h_bytes: usize = log.iter()
            .filter(|r| matches!(r.dir, Dir::D2H))
            .map(|r| r.bytes)
            .sum();
        
        writeln!(fs, "=== MemTrace Summary ===").unwrap();
        writeln!(fs, "Total events:     {}", log.len()).unwrap();
        writeln!(fs, "Total time:       {} µs", total_time).unwrap();
        writeln!(fs, "Total idle:       {} µs", total_idle).unwrap();
        writeln!(fs, "Utilization:      {:.1}%", utilization).unwrap();
        writeln!(fs, "").unwrap();
        writeln!(fs, "Transfer events:  {}", transfer_count).unwrap();
        writeln!(fs, "Kernel events:    {}", kernel_count).unwrap();
        writeln!(fs, "Abort events:     {}", abort_events.len()).unwrap();
        writeln!(fs, "").unwrap();
        writeln!(fs, "H2D total:        {} bytes ({:.2} MB)", 
                 h2d_bytes, h2d_bytes as f64 / 1024.0 / 1024.0).unwrap();
        writeln!(fs, "D2H total:        {} bytes ({:.2} MB)", 
                 d2h_bytes, d2h_bytes as f64 / 1024.0 / 1024.0).unwrap();
        
        // Abort-Token Statistiken
        let unique_tokens: std::collections::HashSet<_> = log.iter()
            .filter_map(|r| r.abort_token.as_ref())
            .collect();
        if !unique_tokens.is_empty() {
            writeln!(fs, "").unwrap();
            writeln!(fs, "Unique abort tokens: {}", unique_tokens.len()).unwrap();
            for token in unique_tokens.iter().take(10) {
                writeln!(fs, "  - {}", token).unwrap();
            }
        }
        
        println!("✓ memtrace_summary.txt geschrieben");
    }
}



pub fn reset() {
    LOG.lock().unwrap().clear();
}



#[derive(Clone, Copy, Debug)]
pub enum Phase { Transfer, Kernel, Abort }
impl Phase {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self { Phase::Transfer=>"Transfer", Phase::Kernel=>"Kernel", Phase::Abort=>"Abort" }
    }
}


#[derive(Debug)]
pub struct TracingScope {
    prev: bool,
}

impl TracingScope {
    #[inline]
    pub fn new(enable: bool) -> Self {
        use std::sync::atomic::Ordering;
        let prev = AUTO_TRACE.swap(enable, Ordering::Relaxed);
        TracingScope { prev }
    }
    #[inline] pub fn enabled() -> Self { Self::new(true) }
    #[inline] pub fn disabled() -> Self { Self::new(false) }
}

impl Drop for TracingScope {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        AUTO_TRACE.store(self.prev, Ordering::Relaxed);
    }
}

#[inline]
pub fn now_us() -> u64 {
    Instant::now().duration_since(*T0).as_micros() as u64
}