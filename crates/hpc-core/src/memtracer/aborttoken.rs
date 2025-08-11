#![cfg(feature = "memtrace")]

use once_cell::sync::Lazy;
use std::sync::Mutex;

use super::{LOG, Record, Dir, Phase, AUTO_TRACE};

pub struct AbortEvent {
    pub tx_id: u64,
    pub cause: String,
    pub retries: u32,
    pub conflict_sz: usize,
    pub t_start_us: u64,
    pub t_end_us: u64,
}

pub fn log_abort(ev: &AbortEvent) {
    if !AUTO_TRACE.load(std::sync::atomic::Ordering::Relaxed) { return; }
    let mut log = LOG.lock().unwrap();
    let prev_end = log.last().map(|r| r.t_end_us).unwrap_or(0);
    let idle = if ev.t_start_us > prev_end { ev.t_start_us - prev_end } else { 0 };
    let abort_tok = CURRENT_ABORT.lock().unwrap().clone();

    log.push(Record {
        t_start_us: ev.t_start_us,
        t_end_us: ev.t_end_us,
        bytes: 0,
        dir: Dir::Kernel,          // Platzhalter; Phase steuert die Semantik
        idle_us: idle,
        abort_token: abort_tok,
        phase: Phase::Abort,
        tx_id: Some(ev.tx_id),
        cause: Some(ev.cause.clone()),
        retries: Some(ev.retries),
        conflict_sz: Some(ev.conflict_sz),
    });
}

pub static CURRENT_ABORT: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

pub fn set_abort_token<S: Into<String>>(token: S) {
    *CURRENT_ABORT.lock().unwrap() = Some(token.into());
}

pub fn clear_abort_token() {
    *CURRENT_ABORT.lock().unwrap() = None;
}

pub struct AbortTokenGuard(Option<String>);
impl AbortTokenGuard {
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
