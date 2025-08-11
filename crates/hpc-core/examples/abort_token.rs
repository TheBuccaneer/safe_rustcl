// examples/abort_token.rs
#![cfg(feature = "memtrace")]

use std::thread::sleep;
use std::time::Duration;

use hpc_core::memtracer::{self, Dir, AbortTokenGuard};

fn main() {
    // 1) Auto-Trace aktivieren
    memtracer::enable_auto_trace();

    // 2) Ohne Token loggen
    {
        let _t = memtracer::start(Dir::H2D, 1024);
        sleep(Duration::from_millis(10));
    }

    // 3) Mit dauerhaft gesetztem Token
    memtracer::set_abort_token("permanent-123");
    {
        let _t = memtracer::start(Dir::D2H, 2048);
        sleep(Duration::from_millis(15));
    }
    memtracer::clear_abort_token();

    // 4) Mit temporärem Token (Guard)
    {
        let _guard = AbortTokenGuard::new("guarded-XYZ");
        let _t = memtracer::start(Dir::Kernel, 4096);
        sleep(Duration::from_millis(5));
    }

    // 5) Flush ins CSV
    memtracer::flush_csv();

    println!("memtrace.csv geschrieben – bitte Datei prüfen.");


}