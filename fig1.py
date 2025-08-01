#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_box.pdf        – Box-Plot       Raw vs. SafeRustCL
fig_overhead.pdf   – Bar Plot       Overhead no feature / metrics / MemTrace
"""

import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

root = pathlib.Path("target/criterion")

# ─────────────────────────── Helper Functions ────────────────────────────
def load_times(dirpath: pathlib.Path) -> pd.Series:
    """Reads sample.json and returns per-iteration times in ms."""
    js = json.loads((dirpath / "new" / "sample.json").read_text())

    if "iteration_times" in js:
        times = js["iteration_times"]
        iters = js.get("iters", [1] * len(times))
    elif "measured" in js:
        times = js["measured"]["iter_times"]
        iters = js["measured"].get("iters", [1] * len(times))
    elif "times" in js:
        times = js["times"]
        iters = js.get("iters", [1] * len(times))
    else:
        raise RuntimeError(f"Unknown sample.json schema: {dirpath}")

    # Convert ns → ms per iteration
    return pd.Series([t / i / 1e6 for t, i in zip(times, iters)])

def median_ms(dirpath: pathlib.Path) -> float:
    est = json.loads((dirpath / "new" / "estimates.json").read_text())
    return est["median"]["point_estimate"] / 1e6  # ns → ms

# ───────────────────────────── Box Plot ──────────────────────────────────
baseline = root / "jacobi4 no features"

raw_samples  = load_times(baseline / "raw_jacobi_1024x1024_10iter")
wrap_samples = load_times(baseline / "jacobi_1024x1024_10iter")

# Color definitions
COL_BOX = "#336699"
COL_MEDIAN = "#CC3300"
COL_OUTLIER = "#CC0033"

plt.figure(figsize=(6, 4))
plt.boxplot(
    [raw_samples, wrap_samples],
    labels=["Raw OpenCL", "SafeRustCL"],
    showfliers=True,
    patch_artist=True,
    widths=[0.4, 0.4],
    boxprops=dict(facecolor=COL_BOX, edgecolor="black"),
    medianprops=dict(color=COL_MEDIAN, linewidth=2),
    flierprops=dict(marker="o", markerfacecolor=COL_OUTLIER,
                    markeredgecolor=COL_OUTLIER, markersize=5, alpha=0.8),
)
plt.ylabel("Execution Time [ms]")
plt.title("Jacobi $1024^2$, 10 Iterations")
plt.tight_layout()
plt.savefig("fig_box.pdf")
plt.close()
print("✔ fig_box.pdf written")

# ─────────────────────── Overhead Bar Plot ────────────────────────────
rows = []
for tag, fld in [("no feature", "jacobi4 no features"),
                 ("metrics",   "jacobi4 metrics"),
                 ("MemTrace",  "jacobi4 memtrace")]:
    base = root / fld
    raw_med = median_ms(base / "raw_jacobi_1024x1024_10iter")
    wrp_med = median_ms(base / "jacobi_1024x1024_10iter")
    rows.append({
        "Feature": tag,
        "Overhead (%)": (wrp_med - raw_med) / raw_med * 100
    })

df = pd.DataFrame(rows).set_index("Feature")

# Bar colors
colors = {"no feature": "#336699", "metrics": "#CC3300", "MemTrace": "#339933"}

ax = df["Overhead (%)"].plot(
    kind="bar",
    color=[colors[idx] for idx in df.index],
    edgecolor="black",
    linewidth=1,
    width=0.6,
)
ax.set_ylabel("Overhead (%)")
ax.set_title("SafeRustCL Overhead by Feature")
ax.set_xticklabels(df.index, rotation=0)  # horizontal labels
plt.tight_layout()
plt.savefig("fig_overhead.pdf")
print("✔ fig_overhead.pdf written\n")
print(df.round(2))
