#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_timeline_multisize.pdf – Host–Device Timeline für verschiedene Buffer-Größen.
Zeigt jeweils Raw (oben) und SafeRustCL (unten) für jede Größe.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# Liste der Buffer-Größen und zugehörige Dateien
sizes = [1026, 8194, 16386, 32770]
files = {
    sz: {
        "Raw": Path(f"memtrace_raw_{sz}.csv"),
        "SafeRustCL": Path(f"memtrace{sz}.csv")
    }
    for sz in sizes
}

# Farben und Y-Level pro Phase
colors = {"H2D": "#007acc", "Kernel": "#555555", "D2H": "#007acc"}
y_base = {"Raw": 1, "SafeRustCL": -1}
line_styles = {"H2D": "-", "D2H": "-", "Kernel": "--"}
line_width = {"H2D": 6, "D2H": 6, "Kernel": 2}

n = len(sizes)
# Höhe pro Timeline-Zeile reduziert: 1.2 statt 2
fig, axes = plt.subplots(nrows=n, figsize=(8, 1.2 * n), sharex=True)

for ax, sz in zip(axes, sizes):
    for variant, path in files[sz].items():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            ax.hlines(
                y_base[variant],
                row.t_start_us,
                row.t_end_us,
                color=colors[row.dir],
                linewidth=line_width[row.dir],
                linestyles=line_styles[row.dir],
            )
    ax.set_yticks([y_base["Raw"], y_base["SafeRustCL"]])
    ax.set_yticklabels(["Raw", "SafeRustCL"], fontsize=9)
    ax.set_ylabel(f"{sz}", rotation=0, va="center", labelpad=15)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    # Vertikalen Abstand zwischen Subplots verringern
    ax.margins(y=0.4)

axes[-1].set_xlabel("Time [µs]", labelpad=8)

# Gemeinsame Legende
legend_elements = [
    Patch(color=colors["H2D"], label="H2D"),
    Patch(color=colors["Kernel"], label="Kernel"),
    Patch(color=colors["D2H"], label="D2H"),
]
axes[0].legend(handles=legend_elements, loc="upper right", frameon=False)

plt.suptitle("Host–Device Timeline for Various Buffer Sizes", y=1.02)
plt.tight_layout(h_pad=0.5)  # weniger horizontalen Padding
plt.savefig("fig_timeline_multisize.pdf", bbox_inches="tight")
plt.show()
