#!/usr/bin/env python3
"""
bench_viz.py – Erstellt drei Publikations-Plots für GPU/CPU-Benchmarkdaten.
Alle Eingabedateien (sample.json, estimates.json, stencil_memtrace.csv) müssen
im selben Verzeichnis liegen wie dieses Skript. Die PNG-Grafiken werden dort
abgelegt: runtime_boxplot.png, memtrace_timeline.png, busy_vs_idle.png
"""
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# 1) Daten einlesen
with open("sample.json", encoding="utf-8") as fp:
    times_ns = json.load(fp)["times"]              # Roh-Laufzeiten

with open("estimates.json", encoding="utf-8") as fp:
    estimates = json.load(fp)                     # Mittel/Median

mem = pd.read_csv("stencil_memtrace.csv")         # Transfer- & Kernel-Events

# ------------------------------------------------------------------ #
# 2) Stilparameter
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")

# ------------------------------------------------------------------ #
# 3) Boxplot der Laufzeiten
def plot_boxplot():
    times_us = np.array(times_ns) / 1_000.0
    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    sns.boxplot(y=times_us, color="#6897bb",
                medianprops={"color": "#ff6600"},
                flierprops={"markerfacecolor": "#e04b52",
                            "markeredgecolor": "#912f34"}, ax=ax)
    ax.set_ylabel("Kernel-Laufzeit [µs]")
    ax.set_title("Verteilung von 30 Laufzeit-Samples\n"
                 f"Median = {estimates['median']['point_estimate'] / 1_000:.1f} µs")
    fig.tight_layout()
    fig.savefig("runtime_boxplot.png", bbox_inches="tight")

# ------------------------------------------------------------------ #
# 4) Timeline-Plot
def plot_timeline():
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    colors = {"H2D": "#4b8bbe", "D2H": "#4b8bbe", "Kernel": "#666666"}
    y_level = {"H2D": 1, "Kernel": 0, "D2H": -1}
    for _, row in mem.iterrows():
        ax.hlines(y=y_level[row.dir], xmin=row.t_start_us, xmax=row.t_end_us,
                   color=colors[row.dir], linewidth=6,
                   linestyles="solid" if row.dir != "Kernel" else "dashed")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Download", "Kernel", "Upload"])
    ax.set_xlabel("Zeit [µs]")
    ax.set_title("Host↔Device-Timeline & Kernel-Slices")
    fig.tight_layout()
    fig.savefig("memtrace_timeline.png", bbox_inches="tight")

# ------------------------------------------------------------------ #
# 5) Busy-vs-Idle Balken
def plot_busy_idle():
    busy_time = (mem.t_end_us - mem.t_start_us).sum()
    idle_time = mem.idle_us.sum()
    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    ax.bar(["Busy", "Idle"], [busy_time, idle_time],
           color=["#007acc", "#a0a0a0"])
    ax.set_ylabel("Zeit [µs]")
    ax.set_title("GPU Auslastung: Busy vs. Idle")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 50,
                f"{bar.get_height():,.0f} µs",
                ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig("busy_vs_idle.png", bbox_inches="tight")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    plot_boxplot()
    plot_timeline()
    plot_busy_idle()
    print("Plots erfolgreich erstellt.")
