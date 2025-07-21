#[cfg(feature = "metrics")]
mod metrics {
    use std::{sync::Mutex, time::Instant};
    lazy_static::lazy_static! {
        static ref TIMES: Mutex<Vec<(&'static str, u128)>> = Mutex::new(Vec::new());
    }
    pub fn record(name: &'static str, start: Instant) {
        let dur = start.elapsed().as_micros();
        TIMES.lock().unwrap().push((name, dur));
    }
    pub fn dump() {
        for (name, us) in TIMES.lock().unwrap().iter() {
            println!("{}: {} µs", name, us);
        }
    }
}

#[cfg(feature = "metrics")]
pub use metrics::*;
