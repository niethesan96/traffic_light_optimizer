#!/usr/bin/env python3
"""
Plot Fixed-Time Baseline Results
==================================

Generates basic plots from the baseline simulation results.

Usage:
    python scripts/plot_baseline.py --results-dir results/baseline
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_step_metrics(results_dir: Path, output_dir: Path) -> None:
    """Plot time-series of waiting time and queue length from step_metrics.csv."""
    csv_path = results_dir / "step_metrics.csv"
    if not csv_path.exists():
        print(f"  Skipping step metrics plot: {csv_path} not found")
        return

    times, waits, queues = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            waits.append(float(row["avg_waiting_time"]))
            queues.append(float(row["total_queue_length"]))

    times = np.array(times)
    waits = np.array(waits)
    queues = np.array(queues)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Average waiting time
    ax1.plot(times / 60, waits, color="#2196F3", linewidth=1.5, alpha=0.8)
    ax1.fill_between(times / 60, waits, alpha=0.15, color="#2196F3")
    ax1.set_ylabel("Avg Waiting Time (s)", fontsize=12)
    ax1.set_title("Fixed-Time Baseline — Network Metrics Over Time", fontsize=14)
    ax1.grid(True, alpha=0.3)
    if len(waits) > 0:
        ax1.axhline(np.mean(waits), color="#F44336", linestyle="--", alpha=0.7,
                     label=f"Mean: {np.mean(waits):.1f}s")
        ax1.legend(fontsize=10)

    # Total queue length
    ax2.plot(times / 60, queues, color="#FF9800", linewidth=1.5, alpha=0.8)
    ax2.fill_between(times / 60, queues, alpha=0.15, color="#FF9800")
    ax2.set_xlabel("Simulation Time (min)", fontsize=12)
    ax2.set_ylabel("Total Queue Length", fontsize=12)
    ax2.grid(True, alpha=0.3)
    if len(queues) > 0:
        ax2.axhline(np.mean(queues), color="#F44336", linestyle="--", alpha=0.7,
                     label=f"Mean: {np.mean(queues):.1f}")
        ax2.legend(fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "step_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_intersection(results_dir: Path, output_dir: Path) -> None:
    """Plot per-intersection bar chart from per_intersection.csv."""
    csv_path = results_dir / "per_intersection.csv"
    if not csv_path.exists():
        print(f"  Skipping per-intersection plot: {csv_path} not found")
        return

    ids, avg_ql, max_ql, avg_spd = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["intersection_id"])
            avg_ql.append(float(row["avg_queue_length"]))
            max_ql.append(int(row["max_queue_length"]))
            avg_spd.append(float(row["avg_speed"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50",
              "#4CAF50", "#4CAF50", "#4CAF50", "#4CAF50"]

    # Avg queue length
    axes[0].bar(range(len(ids)), avg_ql, color="#2196F3", alpha=0.8)
    axes[0].set_xticks(range(len(ids)))
    axes[0].set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Avg Queue Length")
    axes[0].set_title("Avg Queue Length")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Max queue length
    axes[1].bar(range(len(ids)), max_ql, color="#FF9800", alpha=0.8)
    axes[1].set_xticks(range(len(ids)))
    axes[1].set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("Max Queue Length")
    axes[1].set_title("Max Queue Length")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Avg speed
    axes[2].bar(range(len(ids)), avg_spd, color="#4CAF50", alpha=0.8)
    axes[2].set_xticks(range(len(ids)))
    axes[2].set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
    axes[2].set_ylabel("Avg Speed (m/s)")
    axes[2].set_title("Avg Speed")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Fixed-Time Baseline — Per-Intersection Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "per_intersection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_network_summary(results_dir: Path, output_dir: Path) -> None:
    """Plot a summary bar chart from baseline_results.json."""
    json_path = results_dir / "baseline_results.json"
    if not json_path.exists():
        print(f"  Skipping summary plot: {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    net = data["network_level"]

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Avg Wait\n(s)", "Avg Travel\nTime (s)", "Avg Queue\nLength", "Max Queue\nLength"]
    values = [net["avg_waiting_time"], net["avg_travel_time"],
              net["avg_queue_length"], net["max_queue_length"]]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    bars = ax.bar(metrics, values, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Fixed-Time Baseline — Key Metrics", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = output_dir / "network_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot baseline results")
    parser.add_argument("--results-dir", type=str, default="results/baseline",
                        help="Directory containing baseline results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: results-dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating baseline plots...")
    plot_step_metrics(results_dir, output_dir)
    plot_per_intersection(results_dir, output_dir)
    plot_network_summary(results_dir, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
