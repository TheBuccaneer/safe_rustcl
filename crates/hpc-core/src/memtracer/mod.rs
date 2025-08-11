#![cfg(feature = "memtrace")]

//! Memory transfer tracing utilities

mod copytoken;
mod aborttoken;

pub use copytoken::{CopyToken, start, log_transfer};
pub use aborttoken::{
    AbortEvent, log_abort, 
    set_abort_token, clear_abort_token, AbortTokenGuard,
    CURRENT_ABORT,
};

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

/// Transfer direction
#[derive(Clone, Copy, Debug)]
pub enum Dir {
    H2D,
    D2H,
    Kernel,
}

/// Operation type (semantically clearer)
#[derive(Clone, Copy, Debug)]
pub enum Operation {
    H2D,
    D2H,
    Kernel,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Operation::H2D => "H2D",
            Operation::D2H => "D2H",
            Operation::Kernel => "KRN",
        }
    }
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

/// Phase of operation
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    Transfer,
    Kernel,
    Abort,
}

impl Phase {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Transfer => "Transfer",
            Phase::Kernel => "Kernel",
            Phase::Abort => "Abort",
        }
    }
}

/// Global start time reference
pub static T0: Lazy<Instant> = Lazy::new(Instant::now);

/// Auto-trace enable flag
pub static AUTO_TRACE: AtomicBool = AtomicBool::new(true);

/// Enable auto-tracing
#[inline]
pub fn enable_auto_trace() {
    AUTO_TRACE.store(true, Ordering::Relaxed);
}

/// Disable auto-tracing
#[inline]
pub fn disable_auto_trace() {
    AUTO_TRACE.store(false, Ordering::Relaxed);
}

/// Check if auto-tracing is enabled
#[inline]
pub fn is_auto_trace_enabled() -> bool {
    AUTO_TRACE.load(Ordering::Relaxed)
}

/// Log record
#[derive(Debug)]
pub struct Record {
    pub t_start_us: u64,
    pub t_end_us: u64,
    pub bytes: usize,
    pub dir: Dir,
    pub idle_us: u64,
    pub abort_token: Option<String>,
    pub phase: Phase,
    pub tx_id: Option<u64>,
    pub cause: Option<String>,
    pub retries: Option<u32>,
    pub conflict_sz: Option<usize>,
}

/// Global log storage
pub static LOG: Lazy<Mutex<Vec<Record>>> = 
    Lazy::new(|| Mutex::new(Vec::with_capacity(4096)));

/// Flush logs to CSV files
pub fn flush_csv() {
    let log = LOG.lock().unwrap();
    if log.is_empty() {
        println!("⚠ No MemTrace events to export");
        return;
    }

    // Normal events
    let mut f = File::create("memtrace.csv").expect("Cannot create memtrace.csv");
    writeln!(f, "t_start_us,t_end_us,bytes,dir,idle_us,abort_token,phase").unwrap();
    
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
        ).unwrap();
    }

    // Abort events (if any)
    let abort_events: Vec<_> = log.iter()
        .filter(|r| matches!(r.phase, Phase::Abort))
        .collect();
    
    if !abort_events.is_empty() {
        let mut fa = File::create("memtrace_abort.csv")
            .expect("Cannot create memtrace_abort.csv");
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
                r.tx_id.unwrap_or(0),
                r.cause.as_deref().unwrap_or(""),
                r.retries.unwrap_or(0),
                r.conflict_sz.unwrap_or(0),
                r.idle_us,
                r.abort_token.as_deref().unwrap_or("")
            ).unwrap();
        }
        println!("✓ memtrace_abort.csv written ({} aborts)", abort_events.len());
    }
    
    println!("✓ memtrace.csv written ({} events)", 
             log.len() - abort_events.len());
}

/// Reset all logs
pub fn reset() {
    LOG.lock().unwrap().clear();
}

/// RAII scope for temporarily changing trace state
#[derive(Debug)]
pub struct TracingScope {
    prev: bool,
}

impl TracingScope {
    #[inline]
    pub fn new(enable: bool) -> Self {
        let prev = AUTO_TRACE.swap(enable, Ordering::Relaxed);
        TracingScope { prev }
    }
    
    #[inline]
    pub fn enabled() -> Self {
        Self::new(true)
    }
    
    #[inline]
    pub fn disabled() -> Self {
        Self::new(false)
    }
}

impl Drop for TracingScope {
    fn drop(&mut self) {
        AUTO_TRACE.store(self.prev, Ordering::Relaxed);
    }
}

/// Get current time in microseconds since T0
#[inline]
pub fn now_us() -> u64 {
    Instant::now().duration_since(*T0).as_micros() as u64
}