#!/usr/bin/env bash
set -euo pipefail
OUTROOT=${OUTROOT:-results/FGOS_ubuntu/seed1}
OPS=${OPS:-1000000}
SEED=${SEED:-1}
PKG=${PKG:-hpc-core}
BIN=./target/release/examples/stm_abort
mkdir -p "$OUTROOT"
cargo build -p "$PKG" --release --features memtrace --example stm_abort
for c in low med high; do
  for t in 2 4 8; do
    dest="$OUTROOT/$c/t$t"; mkdir -p "$dest"
    taskset -c 0-$((t-1)) sudo chrt -f 50 \
      "$BIN" --threads "$t" --conflict "$c" --ops "$OPS" --seed "$SEED"
    mv memtrace_abort.csv "$dest/"; mv memtrace_summary.txt "$dest/"
    [ -f memtrace.csv ] && mv memtrace.csv "$dest/memtrace_events.csv" || true
  done
done
# Summary
find "$OUTROOT" -name memtrace_summary.txt | while read -r f; do
  ev=$(grep -oP 'events_total:\s*\K\d+' "$f"); ab=$(grep -oP 'aborts:\s*\K\d+' "$f")
  echo "$(dirname "$f")"$'\t'"$ev"$'\t'"$ab"
done > "$OUTROOT/summary_matrix.tsv"
