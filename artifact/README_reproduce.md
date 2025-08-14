How to reproduce…


bench_setup.sh — Systemfixierung vor Benchmarks

Setzt GPU in den Persistenzmodus, fixiert SM-Clocks und Power-Cap (NVIDIA), stellt CPU-Governor auf performance und führt den übergebenen Befehl auf festen CPU-Kernen aus. Benötigt sudo, nvidia-smi, cpupower.
Aufruf: ./bench_setup.sh <kommando> [args...] (z. B. Build/Run).

run_baseline.sh — Baseline-Matrix (Ubuntu)

Baut stm_abort mit --features memtrace, führt die Matrix Konflikt∈{low,med,high} × Threads∈{2,4,8} aus (mit taskset und chrt -f 50), verschiebt memtrace_abort.csv, memtrace_summary.txt (und optional memtrace.csv als memtrace_events.csv) in results/FGOS_ubuntu/seed1/<conflict>/t<thr>/ und erzeugt eine tabellarische Übersicht summary_matrix.tsv mit events_total und aborts. Wichtige ENV-Variablen: OUTROOT (Standard results/FGOS_ubuntu/seed1), OPS, SEED, PKG.
Aufruf: ./run_baseline.sh (ENV nach Bedarf setzen).

time_measure.sh — Ops-basierte Zeitmessung (Ubuntu)

Misst die Wall-Zeit (/usr/bin/time -f '%e') für stm_abort mehrfach und schreibt eine Liste von Messwerten nach results1/ubuntu_timing_ops_<conflict>_t<threads>.txt. Nutzt taskset und chrt -f <prio>. Wichtige ENV-Variablen: REPEATS (Standard 10), THREADS (4), CONFLICT (low|med|high, Standard med), OPS (1_000_000), SEED (1), CPUS (z. B. 0-3), PRIO (50), OUTDIR (results1), PKG (hpc-core).
Aufruf: ./time_measure.sh (ENV nach Bedarf setzen).

run_baseline.ps1 — Baseline-Matrix (Windows)

Windows-Variante zur Erstellung der Baseline-Matrix und zum Einsammeln der memtrace_*-Ausgaben in Ergebnisordnern; analog zum Bash-Skript (run_baseline.sh).
