#!/usr/bin/env python3
"""
plot_scaling.py — Distri-Cluster HPC Scaling Study Visualiser
==============================================================
Reads benchmark_results.csv (produced by benchmark.sh) and generates:
  1. speedup_efficiency.png  — 2-panel: Speedup & Efficiency vs total_procs
  2. time_comparison.png     — Bar chart comparing all configurations
  3. cluster_scatter.png     — 2D PCA scatter coloured by cluster assignment
                               (requires vectors.bin + cluster_labels.bin)

Usage:
    python3 plot_scaling.py                          # all three plots
    python3 plot_scaling.py --no-scatter             # skip scatter (slow)
    python3 plot_scaling.py --csv my_results.csv
"""

import argparse
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "serial":  "#6c757d",
    "omp":     "#0077b6",
    "mpi":     "#e85d04",
    "mpi_omp": "#7209b7",
}

LABELS = {
    "serial":  "Serial",
    "omp":     "OpenMP",
    "mpi":     "MPI",
    "mpi_omp": "MPI+OMP",
}


# ── Binary readers ─────────────────────────────────────────────────────────────

def read_vectors(path: str):
    with open(path, "rb") as f:
        N, D = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(N * D * 4), dtype=np.float32).reshape(N, D)
    return data


def read_labels(path: str):
    with open(path, "rb") as f:
        (N,) = struct.unpack("i", f.read(4))
        labels = np.frombuffer(f.read(N * 4), dtype=np.int32).copy()
    return labels


# ── Plot 1: Speedup & Efficiency ───────────────────────────────────────────────

def plot_speedup_efficiency(df: pd.DataFrame, serial_time: float, outpath: str):
    df = df[df["variant"] != "serial"].copy()
    df["speedup"]    = serial_time / df["time_seconds"]
    df["efficiency"] = df["speedup"] / df["total_procs"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HPC Scaling Study — Distri-Cluster K-Means", fontweight="bold")

    for variant, group in df.groupby("variant"):
        color = COLORS.get(variant, "#333")
        label = LABELS.get(variant, variant)
        g = group.sort_values("total_procs")

        axes[0].plot(g["total_procs"], g["speedup"], "o-",
                     color=color, label=label, linewidth=2, markersize=7)
        axes[1].plot(g["total_procs"], g["efficiency"], "s--",
                     color=color, label=label, linewidth=2, markersize=7)

    # Ideal speedup reference
    max_procs = df["total_procs"].max()
    x_ideal = np.arange(1, max_procs + 1)
    axes[0].plot(x_ideal, x_ideal, "k--", alpha=0.3, linewidth=1.5, label="Ideal (linear)")
    axes[1].axhline(1.0, color="k", linestyle="--", alpha=0.3, linewidth=1.5, label="Ideal (100%)")

    axes[0].set_title("Speedup")
    axes[0].set_xlabel("Total Processors (Ranks × Threads)")
    axes[0].set_ylabel("Speedup  (T_serial / T_parallel)")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(sorted(df["total_procs"].unique()))

    axes[1].set_title("Parallel Efficiency")
    axes[1].set_xlabel("Total Processors (Ranks × Threads)")
    axes[1].set_ylabel("Efficiency  (Speedup / Processors)")
    axes[1].set_ylim(0, 1.3)
    axes[1].legend(frameon=False)
    axes[1].set_xticks(sorted(df["total_procs"].unique()))

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# ── Plot 2: Time bar chart ─────────────────────────────────────────────────────

def plot_time_comparison(df: pd.DataFrame, outpath: str):
    df = df.copy().sort_values(["variant", "total_procs"])

    # Build display labels
    def make_label(row):
        if row["variant"] == "serial":
            return "Serial\n(1 proc)"
        elif row["variant"] == "omp":
            return f"OMP\n{int(row['threads_per_rank'])}T"
        elif row["variant"] == "mpi":
            return f"MPI\n{int(row['ranks'])}R×1T"
        else:
            return f"MPI+OMP\n{int(row['ranks'])}R×{int(row['threads_per_rank'])}T"

    df["label"] = df.apply(make_label, axis=1)
    bar_colors  = [COLORS.get(v, "#333") for v in df["variant"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(df)), df["time_seconds"], color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.7)

    # Annotate bars with time values
    for bar, t in zip(bars, df["time_seconds"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title("K-Means Execution Time — All Configurations", fontweight="bold")

    # Legend patches
    patches = [mpatches.Patch(color=COLORS[v], label=LABELS[v])
               for v in COLORS if v in df["variant"].values]
    ax.legend(handles=patches, frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# ── Plot 3: PCA cluster scatter ────────────────────────────────────────────────

def plot_cluster_scatter(vectors_path: str, labels_path: str, outpath: str):
    print("  Running PCA (this takes ~10s for 10k vectors)...")
    from sklearn.decomposition import PCA

    vectors = read_vectors(vectors_path)
    labels  = read_labels(labels_path)
    K       = len(np.unique(labels))

    # PCA to 2D
    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap    = plt.cm.get_cmap("tab10", K)

    for k in range(K):
        mask = labels == k
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[cmap(k)], s=4, alpha=0.5, linewidths=0,
                   label=f"Cluster {k}  (n={mask.sum()})")

    ax.set_title("K-Means Clusters — PCA Projection (2D)", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(fontsize=8, markerscale=3, frameon=False,
              loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="benchmark_results.csv")
    parser.add_argument("--vectors",    default="vectors.bin")
    parser.add_argument("--labels",     default="cluster_labels.bin")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Skip the PCA scatter plot (slow on large datasets)")
    args = parser.parse_args()

    # ── Load CSV ────────────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {args.csv} not found. Run benchmark.sh first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} benchmark rows from {args.csv}")
    print(df.to_string(index=False))

    serial_rows = df[df["variant"] == "serial"]
    if serial_rows.empty:
        print("ERROR: No serial baseline row in CSV.")
        return
    serial_time = serial_rows["time_seconds"].iloc[0]
    print(f"\nSerial baseline: {serial_time:.4f}s\n")

    # ── Plot 1 ──────────────────────────────────────────────────────────────────
    print("Generating speedup & efficiency plot...")
    if len(df[df["variant"] != "serial"]) >= 2:
        plot_speedup_efficiency(df, serial_time, "speedup_efficiency.png")
    else:
        print("  Skipped — need at least 2 parallel rows")

    # ── Plot 2 ──────────────────────────────────────────────────────────────────
    print("Generating time comparison bar chart...")
    plot_time_comparison(df, "time_comparison.png")

    # ── Plot 3 ──────────────────────────────────────────────────────────────────
    if not args.no_scatter:
        vec_path = Path(args.vectors)
        lbl_path = Path(args.labels)
        if vec_path.exists() and lbl_path.exists():
            print("Generating PCA cluster scatter plot...")
            plot_cluster_scatter(str(vec_path), str(lbl_path), "cluster_scatter.png")
        else:
            missing = [str(p) for p in [vec_path, lbl_path] if not p.exists()]
            print(f"  Skipped scatter — missing: {', '.join(missing)}")
    else:
        print("Scatter plot skipped (--no-scatter)")

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
