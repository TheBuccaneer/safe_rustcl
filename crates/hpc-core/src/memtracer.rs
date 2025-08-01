//! src/memtracer.rs
#![cfg(feature = "memtrace")]

use once_cell::sync::Lazy;
use std::{
    fs::File, 
    io::Write, 
    sync::{Mutex, atomic::{AtomicBool, Ordering}}, 
    time::Instant
};

/// Transfer‑Richtung oder Kernel‑Event
#[derive(Clone, Copy)]
pub enum Dir { H2D, D2H, Kernel }
impl Dir {
    fn as_str(self) -> &'static str {
        match self {
            Dir::H2D    => "H2D",
            Dir::D2H    => "D2H",
            Dir::Kernel => "Kernel",
        }
    }
}

/// globaler Nullpunkt – wird beim ersten start() initialisiert
static T0: Lazy<Instant> = Lazy::new(Instant::now);

/// Log‑Puffer: (start, end, bytes, dir, idle)
static LOG: Lazy<Mutex<Vec<(u128, u128, usize, &'static str, u128)>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

/// Global flag für automatisches Tracing
static AUTO_TRACE: AtomicBool = AtomicBool::new(true);

/// Token hält Startzeit, Größe & Richtung
pub struct CopyToken {
    start: Instant,
    bytes: usize,
    dir: Dir,
}

/// Scoped guard für temporäre Tracing-Kontrolle
pub struct TracingScope {
    prev_state: bool,
}

impl TracingScope {
    pub fn disabled() -> Self {
        let prev_state = AUTO_TRACE.swap(false, Ordering::Relaxed);
        Self { prev_state }
    }
    
    pub fn enabled() -> Self {
        let prev_state = AUTO_TRACE.swap(true, Ordering::Relaxed);
        Self { prev_state }
    }
}

impl Drop for TracingScope {
    fn drop(&mut self) {
        AUTO_TRACE.store(self.prev_state, Ordering::Relaxed);
    }
}

/// Prüft ob Auto-Tracing aktiviert ist
pub fn is_auto_trace_enabled() -> bool {
    AUTO_TRACE.load(Ordering::Relaxed)
}

/// Aktiviert Auto-Tracing global
pub fn enable_auto_trace() {
    AUTO_TRACE.store(true, Ordering::Relaxed);
}

/// Deaktiviert Auto-Tracing global
pub fn disable_auto_trace() {
    AUTO_TRACE.store(false, Ordering::Relaxed);
}

/// Start eines Transfers/Kernels – ruft Lazy::force(&T0) auf
pub fn start(dir: Dir, bytes: usize) -> CopyToken {
    Lazy::force(&T0);
    CopyToken { start: Instant::now(), bytes, dir }
}

impl CopyToken {
    /// Ende eines Transfers/Kernels – schreibt eine Zeile mit idle_us
    pub fn finish(self) {
        let t0 = *T0;
        let s  = self.start.duration_since(t0).as_micros();
        let e  = Instant::now().duration_since(t0).as_micros();
        
        // Idle‑Time berechnen
        let mut log = LOG.lock().unwrap();
        let prev_end = log.last().map(|entry| entry.1).unwrap_or(0);
        let idle = if s > prev_end { s - prev_end } else { 0 };
        
        log.push((s, e, self.bytes, self.dir.as_str(), idle));
    }
}

/// CSV schreiben – einmal am Programmende aufrufen
pub fn flush_csv() {
    let mut f = File::create("memtrace.csv").expect("konnte memtrace.csv nicht anlegen");
    writeln!(f, "t_start_us,t_end_us,bytes,dir,idle_us").unwrap();
    for (s, e, b, d, idle) in LOG.lock().unwrap().iter() {
        writeln!(f, "{},{},{},{},{}", s, e, b, d, idle).unwrap();
    }
}