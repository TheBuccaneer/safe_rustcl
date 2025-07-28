#!/usr/bin/env python3
"""
add_vec_plot.py – Corrected vector-add benchmark plots
Inputs:
  - sample.json    # {"times": [ns, ns, ...]}
  - estimates.json # {"median": {"point_estimate": ...}, ...}
  - memtrace.csv   # t_start_us,t_end_us,bytes,dir,idle_us
Outputs:
  - runtime_boxplot.png
  - memtrace_timeline.png
  - busy_vs_idle.png
"""
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Global style
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
})
sns.set_style("whitegrid")

# ─── 1. Load data ─────────────────────────────────────────────────────
# 1.1 Load raw samples in ns
with open("sample.json", encoding="utf-8") as f:
    samples_ns = json.load(f)["times"]
times_us = np.array(samples_ns) / 1e3  # ns → µs

# 1.2 Load estimates.json (assumed in ns)
with open("estimates.json", encoding="utf-8") as f:
    est = json.load(f)
median_est_us = est["median"]["point_estimate"] / 1e3  # convert ns → µs

# 1.3 Load memtrace.csv
df = pd.read_csv("memtrace.csv")

# ─── 2. Boxplot of kernel runtimes ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 5))
sns.boxplot(y=times_us,
            color="#6897bb",
            medianprops={"color":"#ff6600","linewidth":2},
            flierprops={"markerfacecolor":"#e04b52","markeredgecolor":"#912f34","markersize":4},
            ax=ax)
ax.set_ylabel("Kernel runtime [µs]")
ax.set_xticks([])
ax.set_title("Distribution of 30 kernel runtimes")

# Annotate median from sample data instead of estimates
ymin, ymax = ax.get_ylim()
sample_median = np.median(times_us)
ax.text(0, ymin + 0.05*(ymax-ymin),
        f"Median: {sample_median:,.0f} µs",
        color="#ff6600", ha="center", va="bottom", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#ff6600", alpha=0.85, pad=2))

# Optionally annotate estimate median smaller below
ax.text(0, ymin + 0.01*(ymax-ymin),
        f"est. median: {median_est_us:,.0f} µs",
        color="#444444", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("runtime_boxplot.png", bbox_inches="tight", facecolor='white')
plt.close(fig)

# ─── 3. Timeline & throughput ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
colors = {"H2D":"#4b8bbe","D2H":"#4b8bbe","Kernel":"#696969"}
levels = {"H2D":1,"Kernel":0,"D2H":-1}

for _, row in df.iterrows():
    y = levels[row.dir]
    ax.hlines(y, row.t_start_us, row.t_end_us,
              color=colors[row.dir], linewidth=8,
              linestyles="solid" if row.dir!="Kernel" else "dashed", alpha=0.8)
    if row.dir in ("H2D","D2H") and row.bytes>0:
        dt_s = (row.t_end_us - row.t_start_us) * 1e-6  # µs → s
        thr = row.bytes / dt_s / (1<<30)
        xpos = (row.t_start_us + row.t_end_us) / 2
        label_y = y + 0.25 if y>0 else y-0.25
        ax.text(xpos, label_y, f"{thr:.2f} GiB/s",
                ha="center", va="bottom" if y>0 else "top",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=1))

ax.set_yticks([-1,0,1]); ax.set_yticklabels(["Download","Kernel","Upload"])
ax.set_xlabel("Time [µs]"); ax.set_title("Host–Device Timeline & Throughput")
ax.set_ylim(-1.6,1.6)
fig.tight_layout()
fig.savefig("memtrace_timeline.png", bbox_inches="tight", facecolor='white')
plt.close(fig)

# ─── 4. Busy vs Idle ─────────────────────────────────────────────────
t_max = df.t_end_us.max()
idle = df.idle_us.sum()
busy = t_max - idle

fig, ax = plt.subplots(figsize=(4,5))
bars = ax.bar(["Busy","Idle"], [busy,idle], color=["#007acc","#aaaaaa"], width=0.5)
ax.set_ylabel("Time [µs]"); ax.set_title("GPU utilization: busy vs. idle")
maxv = max(busy,idle)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + maxv*0.05,
            f"{h:,.0f} µs", ha="center", va="bottom", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=1))
ax.set_ylim(0, maxv*1.19)
fig.tight_layout()
fig.savefig("busy_vs_idle.png", bbox_inches="tight", facecolor='white')
plt.close(fig)

# ─── 5. Reference summary ──────────────────────────────────────────────
util = 100 * busy / t_max if t_max else 0
print(f"Reference: total={t_max:.0f} µs, busy={busy:.0f} µs, idle={idle:.0f} µs ({util:.0f}% utilization)")
print(f"Boxplot median sample: {sample_median:,.0f} µs")
print(f"Estimates.json median: {median_est_us:,.0f} µs")
print("Plots: runtime_boxplot.png, memtrace_timeline.png, busy_vs_idle.png")
