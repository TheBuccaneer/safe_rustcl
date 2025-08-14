#!/usr/bin/env bash
# time_measure.sh — Ubuntu: ops-basierte Zeitmessung für stm_abort
set -euo pipefail

# Konfigurierbar via ENV
REPEATS=${REPEATS:-10}
THREADS=${THREADS:-4}
CONFLICT=${CONFLICT:-med}     # low|med|high
OPS=${OPS:-1000000}
SEED=${SEED:-1}
CPUS=${CPUS:-0-3}             # CPU-Affinität, z.B. "0-3" für 4 Kerne
PRIO=${PRIO:-50}              # FIFO-Priorität für chrt
OUTDIR=${OUTDIR:-results1}
PKG=${PKG:-hpc-core}

BIN=./target/release/examples/stm_abort
OUT="${OUTDIR}/ubuntu_timing_ops_${CONFLICT}_t${THREADS}.txt"

mkdir -p "$OUTDIR"

# Build (mit memtrace-Feature)
cargo build -p "$PKG" --release --features memtrace --example stm_abort

# sudo einmalig vor-authentifizieren
sudo -v

# Messen
: > "$OUT"
for i in $(seq 1 "$REPEATS"); do
  /usr/bin/time -f '%e' \
    taskset -c "$CPUS" sudo chrt -f "$PRIO" \
    "$BIN" --threads "$THREADS" --conflict "$CONFLICT" --ops "$OPS" --seed "$SEED" \
    1>/dev/null 2>>"$OUT"
done

echo "geschrieben: $OUT"
