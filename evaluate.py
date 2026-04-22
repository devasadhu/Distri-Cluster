#!/usr/bin/env python3
"""
evaluate.py — Distri-Cluster Clustering Quality Evaluator
==========================================================
Computes NMI, Purity, and Adjusted Rand Index between K-Means cluster
assignments and ground-truth class labels.

Usage:
    python3 evaluate.py                                    # all defaults
    python3 evaluate.py --pred cluster_labels_mpi.bin     # specific prediction
    python3 evaluate.py --vectors text_vectors.bin \
                        --labels  text_labels.bin \
                        --pred    cluster_labels_omp.bin

Binary format (same for labels and predictions):
    [int32: N][int32 × N]
"""

import argparse
import struct
import numpy as np
from pathlib import Path


# ── Binary readers ─────────────────────────────────────────────────────────────

def read_labels_bin(path: str) -> np.ndarray:
    """Read [int32 N][int32 × N] binary file → numpy int32 array."""
    with open(path, "rb") as f:
        (N,) = struct.unpack("i", f.read(4))
        labels = np.frombuffer(f.read(N * 4), dtype=np.int32).copy()
    print(f"  Loaded {N} labels from {path}")
    return labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_k  max_c |cluster_k ∩ class_c|
    Intuition: for each cluster, what fraction of points belong to its
    majority class? Average that across all clusters.
    """
    N = len(true_labels)
    clusters = np.unique(pred_labels)
    total = 0
    for k in clusters:
        mask = pred_labels == k
        if mask.sum() == 0:
            continue
        counts = np.bincount(true_labels[mask])
        total += counts.max()
    return total / N


def nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Normalized Mutual Information = I(Y; C) / sqrt(H(Y) * H(C))
    where Y = true classes, C = predicted clusters.
    
    NMI = 1  → perfect alignment between clusters and classes
    NMI = 0  → clusters are independent of true labels
    """
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(true_labels, pred_labels, average_method="arithmetic")


def ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Adjusted Rand Index: measures pairwise agreement, adjusted for chance.
    ARI = 1 → perfect, ARI = 0 → random, ARI < 0 → worse than random
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(true_labels, pred_labels)


# ── Cluster size distribution ──────────────────────────────────────────────────

def cluster_stats(pred_labels: np.ndarray, K: int) -> dict:
    counts = np.bincount(pred_labels, minlength=K)
    return {
        "min_size":  int(counts.min()),
        "max_size":  int(counts.max()),
        "mean_size": float(counts.mean()),
        "std_size":  float(counts.std()),
        "empty_clusters": int((counts == 0).sum()),
    }


# ── Per-class cluster analysis ─────────────────────────────────────────────────

def class_cluster_alignment(true_labels: np.ndarray, pred_labels: np.ndarray,
                             class_names: list = None) -> None:
    """Print which cluster each true class maps to most strongly."""
    classes = np.unique(true_labels)
    K = len(np.unique(pred_labels))
    print(f"\n  {'Class':<20} {'Dominant Cluster':>16} {'Purity%':>10} {'Count':>8}")
    print(f"  {'-'*20} {'-'*16} {'-'*10} {'-'*8}")
    for c in classes:
        mask = true_labels == c
        cluster_counts = np.bincount(pred_labels[mask], minlength=K)
        dom_cluster = cluster_counts.argmax()
        dom_purity  = cluster_counts.max() / mask.sum() * 100
        name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        print(f"  {name:<20} {dom_cluster:>16}    {dom_purity:>8.1f}%  {mask.sum():>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def main():
    parser = argparse.ArgumentParser(description="Clustering quality evaluator")
    parser.add_argument("--labels",  default="labels.bin",
                        help="Ground truth labels binary (default: labels.bin)")
    parser.add_argument("--pred",    default="cluster_labels.bin",
                        help="Predicted cluster labels binary (default: cluster_labels.bin)")
    parser.add_argument("--K",       type=int, default=10,
                        help="Number of clusters (default: 10)")
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "newsgroups", "custom"],
                        help="Dataset name for class label display")
    parser.add_argument("--all",     action="store_true",
                        help="Evaluate all available cluster_labels_*.bin files")
    args = parser.parse_args()

    class_names = CIFAR10_CLASSES if args.dataset == "cifar10" else None

    # Load ground truth
    print(f"\nLoading ground truth labels...")
    true_labels = read_labels_bin(args.labels)

    pred_files = []
    if args.all:
        pred_files = sorted(Path(".").glob("cluster_labels*.bin"))
        if not pred_files:
            print("No cluster_labels*.bin files found. Run a K-Means variant first.")
            return
    else:
        pred_files = [Path(args.pred)]

    print(f"\n{'='*60}")
    print(f"  CLUSTERING QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Ground truth : {args.labels}  (N={len(true_labels)})")
    print(f"  K            : {args.K}")
    print(f"  Dataset      : {args.dataset}")

    results = []

    for pred_path in pred_files:
        if not pred_path.exists():
            print(f"\n  SKIP: {pred_path} not found")
            continue

        print(f"\n{'─'*60}")
        print(f"  Evaluating: {pred_path.name}")

        pred_labels = read_labels_bin(str(pred_path))

        if len(pred_labels) != len(true_labels):
            print(f"  ERROR: length mismatch — pred={len(pred_labels)}, true={len(true_labels)}")
            continue

        p   = purity(true_labels, pred_labels)
        n   = nmi(true_labels, pred_labels)
        a   = ari(true_labels, pred_labels)
        stats = cluster_stats(pred_labels, args.K)

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  Purity                   : {p:.4f}  │")
        print(f"  │  NMI (normalized mut. inf): {n:.4f}  │")
        print(f"  │  Adjusted Rand Index      : {a:.4f}  │")
        print(f"  └─────────────────────────────────────┘")
        print(f"\n  Cluster size stats:")
        print(f"    Min={stats['min_size']}  Max={stats['max_size']}  "
              f"Mean={stats['mean_size']:.1f}  Std={stats['std_size']:.1f}  "
              f"Empty={stats['empty_clusters']}")

        if class_names:
            class_cluster_alignment(true_labels, pred_labels, class_names)

        results.append({
            "file": pred_path.name,
            "purity": p, "nmi": n, "ari": a
        })

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'File':<35} {'Purity':>8} {'NMI':>8} {'ARI':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        for r in results:
            print(f"  {r['file']:<35} {r['purity']:>8.4f} {r['nmi']:>8.4f} {r['ari']:>8.4f}")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
