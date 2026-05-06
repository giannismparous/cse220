#!/usr/bin/env python3
"""
CSE220 Lab 2 plotting:
- IPC grouped bars (7 dcache configs)
- D-cache miss ratio grouped bars (7 dcache configs)

Uses interval (non-total) counters from memory.stat.0.csv.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("font", size=12)
import matplotlib.pyplot as plt
import numpy as np


def read_descriptor(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_csv_row_value(line: str) -> Optional[Tuple[str, float]]:
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


def dcache_miss_ratio(mem: Dict[str, float]) -> float:
    # In Scarab memory.stat.def these are dcache (on-path) events.
    miss = mem.get("DCACHE_MISS_count", 0.0)
    hit = mem.get("DCACHE_HIT_count", 0.0)
    stb = mem.get("DCACHE_ST_BUFFER_HIT_count", 0.0)
    den = miss + hit + stb
    return safe_ratio(miss, den)


def dcache_3c_counts(mem: Dict[str, float]) -> Tuple[float, float, float]:
    compulsory = mem.get("DCACHE_MISS_COMPULSORY_count", 0.0)
    capacity = mem.get("DCACHE_MISS_CAPACITY_count", 0.0)
    conflict = mem.get("DCACHE_MISS_CONFLICT_count", 0.0)
    return compulsory, capacity, conflict


def pretty_benchmark_label(name: str) -> str:
    if name == "Avg":
        return name
    short = name
    if "." in short:
        short = short.split(".", 1)[1]
    if short.endswith("_r"):
        short = short[:-2]
    return short


def plot_grouped(
    labels: List[str],
    series: Dict[str, List[float]],
    ylabel: str,
    title: str,
    out_path: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#64B5CD",
    ]
    ind = np.arange(len(labels))
    n = len(series)
    width = min(0.12, 0.86 / max(n + 1, 1))
    fig, ax = plt.subplots(figsize=(18, 5.6), dpi=110)
    start = -int(n / 2)
    for idx, key in enumerate(series.keys()):
        ax.bar(
            ind + (start + idx) * width,
            series[key],
            width=width,
            alpha=0.92,
            color=colors[idx % len(colors)],
            edgecolor="#333333",
            linewidth=0.5,
            label=key,
        )
    ax.set_title(title)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ind)
    ax.set_xticklabels([pretty_benchmark_label(x) for x in labels], rotation=38, ha="right")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)


def plot_stacked_3c_by_config(
    configs: List[str],
    compulsory: List[float],
    capacity: List[float],
    conflict: List[float],
    out_path: str,
) -> None:
    ind = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=110)

    ax.bar(ind, compulsory, color="#4C72B0", edgecolor="#333333", linewidth=0.5, label="Compulsory")
    ax.bar(ind, capacity, bottom=compulsory, color="#DD8452", edgecolor="#333333", linewidth=0.5, label="Capacity")
    bottom_conflict = [c + k for c, k in zip(compulsory, capacity)]
    ax.bar(ind, conflict, bottom=bottom_conflict, color="#55A868", edgecolor="#333333", linewidth=0.5, label="Conflict")

    ax.set_title("Lab2 3C D-cache Miss Breakdown (aggregated across benchmarks)")
    ax.set_xlabel("D-cache configuration")
    ax.set_ylabel("Miss share")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(ind)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)


def slice_series(
    labels_full: List[str], series_full: Dict[str, List[float]], start: int, end: int
) -> Tuple[List[str], Dict[str, List[float]]]:
    labels = labels_full[start:end] + ["Avg"]
    series: Dict[str, List[float]] = {}
    for k, vals in series_full.items():
        series[k] = vals[start:end] + [vals[-1]]
    return labels, series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-d", "--descriptor_name", required=True)
    parser.add_argument("-s", "--simulation_path", required=True)
    args = parser.parse_args()

    desc = read_descriptor(args.descriptor_name)
    workloads: List[str] = desc["workloads_list"]
    experiment: str = desc["experiment"]
    configs = list(desc["configurations"].keys())

    ipc_data: Dict[str, List[float]] = {c: [] for c in configs}
    dcmr_data: Dict[str, List[float]] = {c: [] for c in configs}
    c3_comp_counts: Dict[str, float] = {c: 0.0 for c in configs}
    c3_cap_counts: Dict[str, float] = {c: 0.0 for c in configs}
    c3_conf_counts: Dict[str, float] = {c: 0.0 for c in configs}

    for cfg in configs:
        for wl in workloads:
            base = os.path.join(args.simulation_path, wl, experiment, cfg)
            mem_csv = os.path.join(base, "memory.stat.0.csv")
            mem = load_interval_counts(mem_csv)
            ipc_data[cfg].append(periodic_ipc_from_memory(mem_csv))
            dcmr_data[cfg].append(dcache_miss_ratio(mem))
            comp, cap, conf = dcache_3c_counts(mem)
            c3_comp_counts[cfg] += comp
            c3_cap_counts[cfg] += cap
            c3_conf_counts[cfg] += conf

    n = len(workloads)
    for cfg in configs:
        ipc_data[cfg].append(sum(ipc_data[cfg]) / n)
        dcmr_data[cfg].append(sum(dcmr_data[cfg]) / n)

    labels_full = [w.split("/")[-1] if "/" in w else w for w in workloads]
    labels_with_avg = labels_full + ["Avg"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Full plots (all 23 + Avg)
    plot_grouped(
        labels_with_avg,
        ipc_data,
        "IPC (instr./cycle)",
        "Lab2 IPC by D-cache Configuration",
        os.path.join(args.output_dir, "Figure_lab2_ipc.png"),
    )
    plot_grouped(
        labels_with_avg,
        dcmr_data,
        "D-cache miss ratio (miss/access)",
        "Lab2 D-cache Miss Ratio by D-cache Configuration",
        os.path.join(args.output_dir, "Figure_lab2_dcache_miss_ratio.png"),
        ylim=(0.0, None),
    )

    # Optional readability variants: split into three plots (as allowed by assignment).
    chunks = [(0, 8), (8, 16), (16, len(labels_full))]
    for i, (st, en) in enumerate(chunks, start=1):
        lbl, ser = slice_series(labels_full, ipc_data, st, en)
        plot_grouped(
            lbl,
            ser,
            "IPC (instr./cycle)",
            f"Lab2 IPC (Part {i})",
            os.path.join(args.output_dir, f"Figure_lab2_ipc_part{i}.png"),
        )
        lbl, ser = slice_series(labels_full, dcmr_data, st, en)
        plot_grouped(
            lbl,
            ser,
            "D-cache miss ratio (miss/access)",
            f"Lab2 D-cache Miss Ratio (Part {i})",
            os.path.join(args.output_dir, f"Figure_lab2_dcache_miss_ratio_part{i}.png"),
            ylim=(0.0, None),
        )

    # 3C stacked breakdown by cache configuration.
    comp_share: List[float] = []
    cap_share: List[float] = []
    conf_share: List[float] = []
    for cfg in configs:
        total = c3_comp_counts[cfg] + c3_cap_counts[cfg] + c3_conf_counts[cfg]
        if total <= 0.0:
            comp_share.append(0.0)
            cap_share.append(0.0)
            conf_share.append(0.0)
        else:
            comp_share.append(c3_comp_counts[cfg] / total)
            cap_share.append(c3_cap_counts[cfg] / total)
            conf_share.append(c3_conf_counts[cfg] / total)

    plot_stacked_3c_by_config(
        configs,
        comp_share,
        cap_share,
        conf_share,
        os.path.join(args.output_dir, "Figure_lab2_3c_miss_breakdown.png"),
    )

    print("Wrote lab2 figures to:", args.output_dir)


if __name__ == "__main__":
    main()

