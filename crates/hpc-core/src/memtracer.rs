//! src/memtracer.rs – kombiniert CopyToken + Abort-Token + alte API-Shims
#![cfg(feature = "memtrace")]

use once_cell::sync::Lazy;
use std::{
    fs::File,
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::Instant,
};

/// Richtung/Typ
#[derive(Clone, Copy, Debug)]
pub enum Dir {
    H2D,
    D2H,
    Kernel,
}
impl Dir {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Dir::H2D => "H2D",
            Dir::D2H => "D2H",
            Dir::Kernel => "KRN",
        }
    }
}

/// T0 für relative Zeitstempel
static T0: Lazy<Instant> = Lazy::new(Instant::now);

/// Auto-Trace
static AUTO_TRACE: AtomicBool = AtomicBool::new(true);
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

/// Aktueller Abort-Token
static CURRENT_ABORT: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));
#[inline]
pub fn set_abort_token<S: Into<String>>(token: S) {
    *CURRENT_ABORT.lock().unwrap() = Some(token.into());
}
#[inline]
pub fn clear_abort_token() {
    *CURRENT_ABORT.lock().unwrap() = None;
}
pub struct AbortTokenGuard(Option<String>);
impl AbortTokenGuard {
    #[inline]
    pub fn new<S: Into<String>>(token: S) -> Self {
        let mut lock = CURRENT_ABORT.lock().unwrap();
        let prev = lock.take();
        *lock = Some(token.into());
        AbortTokenGuard(prev)
    }
}
impl Drop for AbortTokenGuard {
    fn drop(&mut self) {
        let mut lock = CURRENT_ABORT.lock().unwrap();
        *lock = self.0.take();
    }
}

/// Ein Log-Eintrag
#[derive(Debug)]
struct Record {
    t_start_us: u64,
    t_end_us: u64,
    bytes: usize,
    dir: Dir,
    idle_us: u64,
    abort_token: Option<String>,
}

/// Zentrales Log
static LOG: Lazy<Mutex<Vec<Record>>> = Lazy::new(|| Mutex::new(Vec::with_capacity(4096)));

/// Copy-/Kernel-Token – loggt bei `finish()` oder `Drop`
pub struct CopyToken {
    start: Instant,
    bytes: usize,
    dir: Dir,
    finished: bool,
}
impl CopyToken {
    /// Konsumierendes Finish (kompatibel zu `Box<CopyToken>.finish()`).
    pub fn finish(mut self) {
        if !AUTO_TRACE.load(Ordering::Relaxed) {
            self.finished = true;
            return;
        }
        self.log_once();
    }

    #[inline]
    fn log_once(&mut self) {
        if self.finished {
            return;
        }
        let s = self.start.duration_since(*T0).as_micros() as u64;
        let e = Instant::now().duration_since(*T0).as_micros() as u64;

        let mut log = LOG.lock().unwrap();
        let prev_end = log.last().map(|r| r.t_end_us).unwrap_or(0);
        let idle = if s > prev_end { s - prev_end } else { 0 };
        let abort = CURRENT_ABORT.lock().unwrap().clone();

        log.push(Record {
            t_start_us: s,
            t_end_us: e,
            bytes: self.bytes,
            dir: self.dir,
            idle_us: idle,
            abort_token: abort,
        });

        self.finished = true;
    }
}
impl Drop for CopyToken {
    fn drop(&mut self) {
        if AUTO_TRACE.load(Ordering::Relaxed) {
            self.log_once();
        } else {
            self.finished = true;
        }
    }
}

/// Start eines Transfers/Kernels – gibt Token zurück
#[inline]
pub fn start(dir: Dir, bytes: usize) -> CopyToken {
    Lazy::force(&T0);
    CopyToken {
        start: Instant::now(),
        bytes,
        dir,
        finished: false,
    }
}

/// Direkte Logging-API (extern gemessene Zeitpunkte)
#[inline]
pub fn log_transfer(t_start_us: u64, t_end_us: u64, bytes: usize, dir: Dir) {
    if !AUTO_TRACE.load(Ordering::Relaxed) {
        return;
    }
    let mut log = LOG.lock().unwrap();
    let prev_end = log.last().map(|r| r.t_end_us).unwrap_or(0);
    let idle = if t_start_us > prev_end {
        t_start_us - prev_end
    } else {
        0
    };
    let abort = CURRENT_ABORT.lock().unwrap().clone();

    log.push(Record {
        t_start_us,
        t_end_us,
        bytes,
        dir,
        idle_us: idle,
        abort_token: abort,
    });
}

/// CSV schreiben
pub fn flush_csv() {
    let log = LOG.lock().unwrap();
    let mut f = File::create("memtrace.csv").expect("konnte memtrace.csv nicht anlegen");
    writeln!(f, "t_start_us,t_end_us,bytes,dir,idle_us,abort_token").unwrap();
    for r in log.iter() {
        writeln!(
            f,
            "{},{},{},{},{},{}",
            r.t_start_us,
            r.t_end_us,
            r.bytes,
            r.dir.as_str(),
            r.idle_us,
            r.abort_token.as_deref().unwrap_or("")
        )
        .unwrap();
    }
}

/// Log leeren
#[inline]
pub fn reset() {
    LOG.lock().unwrap().clear();
}

/// Kompatibler RAII-Scope (stellt vorherigen Auto-Trace-Zustand bei Drop wieder her)
pub struct TracingScope {
    prev: bool,
}
impl TracingScope {
    #[inline]
    pub fn new(enable: bool) -> Self {
        let prev = is_auto_trace_enabled();
        if enable {
            enable_auto_trace();
        } else {
            disable_auto_trace();
        }
        TracingScope { prev }
    }
}
impl Default for TracingScope {
    fn default() -> Self {
        TracingScope { prev: is_auto_trace_enabled() }
    }
}
impl Drop for TracingScope {
    fn drop(&mut self) {
        if self.prev {
            enable_auto_trace();
        } else {
            disable_auto_trace();
        }
    }
}
