#!/usr/bin/env python3
"""
CSE220 Lab 1: parse Scarab CSV stats (interval _count rows, not *_total_*),
produce four grouped bar charts matching the assignment.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless Docker / no DISPLAY
matplotlib.rc("font", size=14)
import matplotlib.pyplot as plt
import numpy as np


def read_descriptor(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_row_value(line: str) -> Optional[Tuple[str, float]]:
    """Return (stat_key, value) for lines like 'NAME_count, group, value'."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    key = parts[0]
    if "_total_" in key:
        return None
    if not key.endswith("_count"):
        return None
    try:
        return key, float(parts[2])
    except ValueError:
        return None


def load_interval_counts(csv_path: str) -> Dict[str, float]:
    """Last occurrence wins (single dump is typical)."""
    out: Dict[str, float] = {}
    if not os.path.isfile(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_csv_row_value(line)
            if parsed:
                out[parsed[0]] = parsed[1]
    return out


def periodic_ipc_from_memory(memory_csv: str) -> float:
    """IPC for the last printed interval: prefer explicit Periodic IPC line, else instr/cycles."""
    if not os.path.isfile(memory_csv):
        return 0.0
    ipc = 0.0
    p_cycles = 0.0
    p_insts = 0.0
    with open(memory_csv, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Periodic IPC" in line:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2:
                    try:
                        ipc = float(parts[1])
                    except ValueError:
                        pass
            if line.startswith("Periodic_Cycles,"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    try:
                        p_cycles = float(parts[2])
                    except ValueError:
                        pass
            if line.startswith("Periodic_Instructions,"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    try:
                        p_insts = float(parts[2])
                    except ValueError:
                        pass
    if ipc > 0.0:
        return ipc
    if p_cycles > 0.0 and p_insts > 0.0:
        return p_insts / p_cycles
    return 0.0


def safe_ratio(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return num / den


def icache_miss_ratio(mem: Dict[str, float]) -> float:
    miss = mem.get("ICACHE_MISS_ONPATH_count", 0.0)
    hit = mem.get("ICACHE_HIT_ONPATH_count", 0.0)
    return safe_ratio(miss, miss + hit)


def dcache_miss_ratio(mem: Dict[str, float]) -> float:
    miss = mem.get("DCACHE_MISS_ONPATH_count", 0.0)
    hit = mem.get("DCACHE_HIT_ONPATH_count", 0.0)
    stb = mem.get("DCACHE_ST_BUFFER_HIT_ONPATH_count", 0.0)
    den = miss + hit + stb
    return safe_ratio(miss, den)


def branch_mispred_ratio_bp(bp: Dict[str, float]) -> float:
    """
    PDF text: mispredicted branches / executed branches (on-path only).

    Scarab does not emit that exact ratio as one line; we use the on-path
    confidence pair from bp.stat.def (BP_ON_PATH_CONF_MISPRED and
    BP_ON_PATH_CONF_CORRECT), i.e. mispredictions / (mispredictions + correct
    on-path predictions). State this definition in your write-up; if your TA
    wants a different denominator (e.g. all executed CF ops), swap counters
    after inspecting inst.stat.0.csv / course forum guidance.
    """
    mis = bp.get("BP_ON_PATH_CONF_MISPRED_count", 0.0)
    cor = bp.get("BP_ON_PATH_CONF_CORRECT_count", 0.0)
    return safe_ratio(mis, mis + cor)


def collect_metrics(
    sim_path: str, experiment: str, workload: str, config_key: str
) -> Tuple[float, float, float, float]:
    base = os.path.join(sim_path, workload, experiment, config_key)
    mem_csv = os.path.join(base, "memory.stat.0.csv")
    bp_csv = os.path.join(base, "bp.stat.0.csv")

    ipc = periodic_ipc_from_memory(mem_csv)
    mem = load_interval_counts(mem_csv)
    bp = load_interval_counts(bp_csv)

    return (
        ipc,
        branch_mispred_ratio_bp(bp),
        dcache_miss_ratio(mem),
        icache_miss_ratio(mem),
    )


def plot_grouped(
    benchmarks: List[str],
    series: Dict[str, List[float]],
    ylabel: str,
    title: str,
    out_path: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    colors = [
        "#800000",
        "#4363d8",
        "#3cb44b",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#000075",
    ]
    ind = np.arange(len(benchmarks))
    n = len(series)
    width = min(0.22, 0.9 / max(n + 1, 1))
    fig, ax = plt.subplots(figsize=(16, 4.8), dpi=100)
    start = -int(n / 2)
    for idx, key in enumerate(series.keys()):
        hatch = "\\\\" if idx % 2 else "///"
        ax.bar(
            ind + (start + idx) * width,
            series[key],
            width=width,
            fill=False,
            hatch=hatch,
            color=colors[idx % len(colors)],
            edgecolor=colors[idx % len(colors)],
            label=key,
        )
    ax.set_title(title)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind)
    ax.set_xticklabels(benchmarks, rotation=28, ha="right")
    ax.grid(axis="x")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory for PNG outputs (e.g. /home/$USER/plot/lab1)",
    )
    parser.add_argument(
        "-d",
        "--descriptor_name",
        required=True,
        help="Path to lab1.json",
    )
    parser.add_argument(
        "-s",
        "--simulation_path",
        required=True,
        help="Path to .../exp/simulations",
    )
    args = parser.parse_args()

    desc = read_descriptor(args.descriptor_name)
    workloads: List[str] = desc["workloads_list"]
    experiment: str = desc["experiment"]
    configs = list(desc["configurations"].keys())

    bench_labels = [w.split("/")[-1] if "/" in w else w for w in workloads]

    ipc_data: Dict[str, List[float]] = {c: [] for c in configs}
    br_data: Dict[str, List[float]] = {c: [] for c in configs}
    dc_data: Dict[str, List[float]] = {c: [] for c in configs}
    ic_data: Dict[str, List[float]] = {c: [] for c in configs}

    for cfg in configs:
        for wl in workloads:
            ipc, br, dc, ic = collect_metrics(args.simulation_path, experiment, wl, cfg)
            ipc_data[cfg].append(ipc)
            br_data[cfg].append(br)
            dc_data[cfg].append(dc)
            ic_data[cfg].append(ic)

    n = len(workloads)
    for cfg in configs:
        ipc_data[cfg].append(sum(ipc_data[cfg]) / n)
        br_data[cfg].append(sum(br_data[cfg]) / n)
        dc_data[cfg].append(sum(dc_data[cfg]) / n)
        ic_data[cfg].append(sum(ic_data[cfg]) / n)

    labels = bench_labels + ["Avg"]
    os.makedirs(args.output_dir, exist_ok=True)

    plot_grouped(
        labels,
        ipc_data,
        "IPC (dimensionless: instr. / cycle)",
        "Instructions per cycle (from Periodic region; not *_total_* counters)",
        os.path.join(args.output_dir, "Figure_ipc.png"),
    )
    plot_grouped(
        labels,
        br_data,
        "Branch mispred. ratio (dimensionless)",
        "On-path branch predictor (mispred / (mispred + correct); see plot_lab1.py docstring)",
        os.path.join(args.output_dir, "Figure_branch_mispred_ratio.png"),
        ylim=(0.0, None),
    )
    plot_grouped(
        labels,
        dc_data,
        "D-cache miss ratio (dimensionless: misses / accesses)",
        "On-path: DCACHE_MISS_ONPATH / (miss + hit + ST buffer hit on-path); interval _count only",
        os.path.join(args.output_dir, "Figure_dcache_miss_ratio.png"),
        ylim=(0.0, None),
    )
    plot_grouped(
        labels,
        ic_data,
        "I-cache miss ratio (dimensionless: misses / accesses)",
        "On-path: ICACHE_MISS_ONPATH / (miss + hit on-path); interval _count only",
        os.path.join(args.output_dir, "Figure_icache_miss_ratio.png"),
        ylim=(0.0, None),
    )

    print("Wrote:", os.path.join(args.output_dir, "Figure_*.png"))


if __name__ == "__main__":
    main()
