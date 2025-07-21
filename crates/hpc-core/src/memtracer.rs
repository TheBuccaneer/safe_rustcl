#[cfg(feature = "memtrace")]
mod memtrace {
    use std::{fs::File, io::Write, sync::Mutex, time::Instant};

    /// Richtungs-Enum
    #[derive(Clone, Copy)]
    pub enum Dir { H2D, D2H }

    /// Protokoll-Eintrag: (Startµs, Endµs, Bytes, Dir)
    static LOG: Mutex<Vec<(u128, u128, usize, Dir)>> = Mutex::new(Vec::new());

    /// Token für einen Kopiervorgang
    pub struct CopyToken {
        start: Instant,
        bytes: usize,
        dir: Dir,
    }

    /// Startet ein Logging für `bytes` in Richtung `dir`
    pub fn start(dir: Dir, bytes: usize) -> CopyToken {
        CopyToken { start: Instant::now(), bytes, dir }
    }

    impl CopyToken {
        /// Schließt das Logging ab und fügt den Eintrag in den globalen Log
        pub fn finish(self) {
            let end = Instant::now();
            let mut log = LOG.lock().unwrap();
            log.push((
                self.start.elapsed().as_micros(),
                end.elapsed().as_micros(),
                self.bytes,
                self.dir,
            ));
        }
    }

    /// Schreibt alle gesammelten Einträge in memtrace.csv
    pub fn flush_csv() {
        let data = LOG.lock().unwrap();
        let mut f = File::create("memtrace.csv").unwrap();
        writeln!(f, "t_start_us,t_end_us,bytes,dir").unwrap();
        for &(s,e,b,d) in data.iter() {
            let dir_str = match d { Dir::H2D => "H2D", Dir::D2H => "D2H" };
            writeln!(f, "{},{},{},{}", s, e, b, dir_str).unwrap();
        }
    }
}
// Export übernehmen, wenn Feature aktiv
#[cfg(feature = "memtrace")]
pub use memtrace::*;