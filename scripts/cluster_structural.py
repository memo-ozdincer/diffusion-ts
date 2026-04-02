#!/usr/bin/env python
"""Structural RMSD clustering of 87 TS from adaptive run.

Computes full aligned RMSD matrix, analyzes distance distribution,
tries multiple clustering scales, visualizes with t-SNE/MDS/dendrograms.
Focus: find natural structural TS groupings (not energy-based).
"""

import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.dependencies.alignment import (
    aligned_rmsd, pairwise_rmsd_matrix, kabsch_align
)

# ============================================================================
# Config
# ============================================================================
DATA_PATH = "/scratch/memoozd/diffusion-ts/adaptive_results.json"
OUT_DIR = "/scratch/memoozd/diffusion-ts/structural_clustering"
os.makedirs(OUT_DIR, exist_ok=True)

# Isopropanol equivalence classes
EQUIV_CLASSES = {
    "central_C": [0],
    "O": [3],
    "H_on_C": [4],
    "H_on_O": [5],
    "methyl_C": [1, 2],
    "H_methyl": [6, 7, 8, 9, 10, 11],
}
METHYL_CARBONS = [1, 2]
METHYL_HYDROGENS = [[6, 7, 8], [9, 10, 11]]

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
NOISE_COLOR = [0.5, 0.5, 0.5, 1.0]


def load_ts_data():
    """Load TS geometries from adaptive results."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    ts_list = [r for r in data if r["status"] == "ts" and "ts_coords" in r]
    print(f"Loaded {len(ts_list)} TS from {DATA_PATH}")
    return ts_list


def compute_rmsd_matrix(ts_list):
    """Compute or load cached RMSD matrix."""
    cache = os.path.join(OUT_DIR, "rmsd_matrix_87.npy")
    seeds_cache = os.path.join(OUT_DIR, "seeds_87.json")

    if os.path.exists(cache) and os.path.exists(seeds_cache):
        D = np.load(cache)
        with open(seeds_cache) as f:
            cached_seeds = json.load(f)
        current_seeds = [r["seed"] for r in ts_list]
        if cached_seeds == current_seeds and D.shape[0] == len(ts_list):
            print(f"Using cached RMSD matrix ({D.shape[0]}x{D.shape[0]})")
            return D

    geometries = [np.array(r["ts_coords"]) for r in ts_list]
    n = len(geometries)
    print(f"Computing {n}x{n} RMSD matrix ({n*(n-1)//2} pairs)...")
    t0 = time.time()

    D = pairwise_rmsd_matrix(
        geometries, EQUIV_CLASSES,
        methyl_carbons=METHYL_CARBONS,
        methyl_hydrogens=METHYL_HYDROGENS,
    )

    print(f"Done in {time.time()-t0:.1f}s")
    np.save(cache, D)
    with open(seeds_cache, "w") as f:
        json.dump([r["seed"] for r in ts_list], f)

    return D


def analyze_rmsd_distribution(D, ts_list):
    """Analyze the RMSD distance distribution."""
    n = D.shape[0]
    triu = D[np.triu_indices(n, k=1)]

    print(f"\nRMSD distribution ({len(triu)} pairs):")
    print(f"  min={triu.min():.4f}, max={triu.max():.4f}")
    print(f"  mean={triu.mean():.4f}, median={np.median(triu):.4f}")
    print(f"  std={triu.std():.4f}")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {pct}th percentile: {np.percentile(triu, pct):.4f}")

    # Plot RMSD histogram with KDE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full histogram
    ax = axes[0, 0]
    ax.hist(triu, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.set_xlabel("Aligned RMSD (A)")
    ax.set_ylabel("Density")
    ax.set_title(f"RMSD Distribution ({len(triu)} pairs)")
    ax.axvline(np.median(triu), color="red", ls="--", label=f"median={np.median(triu):.3f}")
    ax.legend()

    # 2. Zoomed low-RMSD region
    ax = axes[0, 1]
    low = triu[triu < 0.5]
    if len(low) > 0:
        ax.hist(low, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.set_xlabel("Aligned RMSD (A)")
        ax.set_ylabel("Density")
        ax.set_title(f"Low-RMSD region (<0.5 A, {len(low)} pairs)")
    else:
        ax.text(0.5, 0.5, "No pairs < 0.5 A", ha="center", va="center", transform=ax.transAxes)

    # 3. Sorted nearest-neighbor distances
    ax = axes[1, 0]
    nn_dists = []
    for i in range(n):
        dists = sorted(D[i, :])
        nn_dists.append(dists[1])  # nearest neighbor (skip self)
    nn_dists.sort()
    ax.plot(nn_dists, "o-", markersize=3, color="steelblue")
    ax.set_xlabel("Point index (sorted)")
    ax.set_ylabel("Nearest-neighbor RMSD (A)")
    ax.set_title("k-NN distance plot (k=1)")
    ax.grid(True, alpha=0.3)

    # 4. k-distance plot for DBSCAN eps selection
    ax = axes[1, 1]
    for k in [2, 3, 5]:
        k_dists = []
        for i in range(n):
            dists = sorted(D[i, :])
            if k < len(dists):
                k_dists.append(dists[k])
        k_dists.sort()
        ax.plot(k_dists, "o-", markersize=2, label=f"k={k}")
    ax.set_xlabel("Point index (sorted)")
    ax.set_ylabel(f"k-th NN RMSD (A)")
    ax.set_title("k-distance plot (for DBSCAN eps selection)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_rmsd_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 01_rmsd_distribution.png")

    return triu


def compute_embeddings(D):
    """Compute MDS and t-SNE embeddings."""
    n = D.shape[0]
    print("\nComputing MDS embedding...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", max_iter=1000)
    mds_coords = mds.fit_transform(D)

    print("Computing t-SNE embeddings...")
    tsne_results = {}
    for perp in [5, 10, 15, 20, 30]:
        if perp >= n:
            continue
        tsne = TSNE(n_components=2, metric="precomputed", perplexity=perp,
                    random_state=42, init="random", max_iter=3000)
        tsne_results[perp] = tsne.fit_transform(D)

    return mds_coords, tsne_results


def plot_embeddings_colored_by_features(mds_coords, tsne_results, ts_list, D):
    """Plot embeddings colored by energy, eig0, force, nearest-neighbor dist."""
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    forces = np.array([r["final_force"] for r in ts_list])
    n = len(ts_list)
    nn_dists = np.array([sorted(D[i, :])[1] for i in range(n)])

    features = [
        ("Energy (eV)", energies, "RdYlBu_r"),
        ("eig0", eig0s, "RdYlBu_r"),
        ("RMS force", forces, "YlOrRd"),
        ("NN RMSD (A)", nn_dists, "viridis"),
    ]

    # Use best t-SNE
    best_perp = 15 if 15 in tsne_results else min(tsne_results.keys())
    tsne_xy = tsne_results[best_perp]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (label, vals, cmap) in enumerate(features):
        for row, (coords, embed_name) in enumerate([(mds_coords, "MDS"), (tsne_xy, f"t-SNE (p={best_perp})")]):
            ax = axes[row, col]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap,
                           s=40, edgecolors="k", linewidths=0.3, alpha=0.85)
            plt.colorbar(sc, ax=ax, shrink=0.8)
            ax.set_title(f"{embed_name} — {label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_embeddings_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 02_embeddings_features.png")


def plot_dendrogram_analysis(D, ts_list):
    """Multi-linkage dendrogram + threshold scan."""
    energies = np.array([r["final_energy"] for r in ts_list])
    seeds = [r["seed"] for r in ts_list]
    D_condensed = squareform(D, checks=False)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    for idx, method in enumerate(["average", "complete", "single", "ward"]):
        ax = axes[idx // 2, idx % 2]
        if method == "ward":
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                      normalized_stress="auto")
            embed = mds.fit_transform(D)
            Z = linkage(embed, method="ward")
        else:
            Z = linkage(D_condensed, method=method)

        # Color by energy
        energy_norm = (energies - energies.min()) / (energies.max() - energies.min() + 1e-10)
        cmap = plt.cm.RdYlBu_r

        dendrogram(Z, ax=ax, truncate_mode="lastp", p=40,
                   leaf_rotation=90, leaf_font_size=6, color_threshold=0)
        ax.set_title(f"{method.capitalize()} linkage", fontsize=11)
        ax.set_ylabel("RMSD (A)" if method != "ward" else "Ward distance")

        # Add horizontal lines at common thresholds
        for t in [0.1, 0.2, 0.3, 0.5]:
            ax.axhline(y=t, color="red", ls=":", alpha=0.5, lw=0.8)
            ax.text(ax.get_xlim()[1] * 0.95, t, f"{t}", fontsize=7, color="red",
                   va="bottom", ha="right")

    plt.suptitle("Dendrogram Comparison (4 linkage methods)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_dendrograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 03_dendrograms.png")

    return linkage(D_condensed, method="average")


def threshold_scan(D, Z_avg, ts_list, mds_coords, tsne_results):
    """Scan clustering threshold and plot results at multiple scales."""
    energies = np.array([r["final_energy"] for r in ts_list])

    thresholds = np.arange(0.05, 1.01, 0.025)
    n_clusters_list = []
    sil_scores = []
    largest_frac = []
    singleton_frac = []

    for t in thresholds:
        labels = fcluster(Z_avg, t=t, criterion="distance")
        nc = len(set(labels))
        n_clusters_list.append(nc)

        if nc >= 2:
            sil = silhouette_score(D, labels, metric="precomputed")
        else:
            sil = 0
        sil_scores.append(sil)

        counts = np.bincount(labels)
        largest_frac.append(counts.max() / len(labels))
        singleton_frac.append(np.sum(counts == 1) / nc)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(thresholds, n_clusters_list, "o-", markersize=3, color="steelblue")
    ax.set_xlabel("RMSD threshold (A)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster count vs threshold")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(thresholds, sil_scores, "o-", markersize=3, color="darkorange")
    ax.set_xlabel("RMSD threshold (A)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette vs threshold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(thresholds, largest_frac, "o-", markersize=3, color="green", label="Largest cluster frac")
    ax.plot(thresholds, singleton_frac, "s-", markersize=3, color="red", label="Singleton frac")
    ax.set_xlabel("RMSD threshold (A)")
    ax.set_ylabel("Fraction")
    ax.set_title("Cluster size distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Silhouette at key thresholds
    ax = axes[1, 1]
    key_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    for t in key_thresholds:
        labels = fcluster(Z_avg, t=t, criterion="distance")
        nc = len(set(labels))
        ax.bar(t, nc, width=0.04, color="steelblue", alpha=0.7, edgecolor="k")
        ax.text(t, nc + 0.5, str(nc), ha="center", fontsize=8)
    ax.set_xlabel("RMSD threshold (A)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster count at key thresholds")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_threshold_scan.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 04_threshold_scan.png")

    # Print summary table
    print("\nThreshold scan (average linkage):")
    print(f"{'Thresh':>7} {'Ncl':>5} {'Sil':>7} {'Largest':>8} {'Singletons':>11}")
    for t in key_thresholds:
        labels = fcluster(Z_avg, t=t, criterion="distance")
        nc = len(set(labels))
        sil = silhouette_score(D, labels, metric="precomputed") if nc >= 2 else 0
        counts = np.bincount(labels)
        lf = counts.max() / len(labels)
        sf = np.sum(counts == 1) / nc
        print(f"{t:7.2f} {nc:5d} {sil:7.3f} {lf:8.1%} {sf:11.1%}")


def plot_clustering_at_scales(D, Z_avg, ts_list, mds_coords, tsne_results):
    """Plot clustering at 3 different scales on MDS and t-SNE."""
    energies = np.array([r["final_energy"] for r in ts_list])
    best_perp = 15 if 15 in tsne_results else min(tsne_results.keys())
    tsne_xy = tsne_results[best_perp]

    # Choose 3 representative thresholds
    scales = [
        (0.15, "Fine (0.15 A)"),
        (0.25, "Medium (0.25 A)"),
        (0.40, "Coarse (0.40 A)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row, (thresh, title) in enumerate(scales):
        labels = fcluster(Z_avg, t=thresh, criterion="distance") - 1
        nc = len(set(labels))

        # Sort labels by cluster size (largest first)
        from collections import Counter
        counts = Counter(labels)
        rank = {cl: i for i, (cl, _) in enumerate(counts.most_common())}
        labels_ranked = np.array([rank[l] for l in labels])

        for col, (coords, embed_name) in enumerate([
            (mds_coords, "MDS"),
            (tsne_xy, f"t-SNE (p={best_perp})"),
        ]):
            ax = axes[row, col]
            for cl in range(nc):
                mask = labels_ranked == cl
                c = COLORS[cl % 20] if cl < 20 else [0.3, 0.3, 0.3, 1]
                ax.scatter(coords[mask, 0], coords[mask, 1], c=[c], s=40,
                          edgecolors="k", linewidths=0.3, alpha=0.8,
                          label=f"C{cl} ({mask.sum()})" if cl < 10 else None)
            ax.set_title(f"{embed_name} — {title} ({nc} clusters)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if nc <= 15:
                ax.legend(fontsize=5, loc="best", ncol=2, framealpha=0.7)

        # Energy by cluster
        ax = axes[row, 2]
        cluster_ids = sorted(set(labels_ranked))
        positions = []
        data_list = []
        for cl in cluster_ids[:20]:
            mask = labels_ranked == cl
            data_list.append(energies[mask])
            positions.append(cl)
        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % 20])
            patch.set_alpha(0.6)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"{title} — Energy by cluster", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Hierarchical Clustering at Three Scales (Average Linkage)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_clustering_scales.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 05_clustering_scales.png")


def dbscan_analysis(D, ts_list, mds_coords, tsne_results):
    """DBSCAN analysis with eps scan."""
    n = D.shape[0]
    energies = np.array([r["final_energy"] for r in ts_list])
    best_perp = 15 if 15 in tsne_results else min(tsne_results.keys())
    tsne_xy = tsne_results[best_perp]

    eps_vals = np.arange(0.05, 0.60, 0.025)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # eps scan
    ax = axes[0, 0]
    for ms in [2, 3]:
        ncs = []
        noise_fracs = []
        for eps in eps_vals:
            db = DBSCAN(eps=eps, min_samples=ms, metric="precomputed")
            labels = db.fit_predict(D)
            nc = len(set(labels)) - (1 if -1 in labels else 0)
            nf = np.mean(labels == -1)
            ncs.append(nc)
            noise_fracs.append(nf)
        ax.plot(eps_vals, ncs, "o-", markersize=3, label=f"ms={ms}")
    ax.set_xlabel("eps (A)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("DBSCAN: clusters vs eps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Noise fraction
    ax = axes[0, 1]
    for ms in [2, 3]:
        noise_fracs = []
        for eps in eps_vals:
            db = DBSCAN(eps=eps, min_samples=ms, metric="precomputed")
            labels = db.fit_predict(D)
            noise_fracs.append(np.mean(labels == -1))
        ax.plot(eps_vals, noise_fracs, "o-", markersize=3, label=f"ms={ms}")
    ax.set_xlabel("eps (A)")
    ax.set_ylabel("Noise fraction")
    ax.set_title("DBSCAN: noise fraction vs eps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Best DBSCAN results at a few eps values
    good_eps = [0.10, 0.15, 0.20, 0.30]
    for idx, eps in enumerate(good_eps):
        if idx + 2 >= 6:
            break
        ax = axes[(idx + 2) // 3, (idx + 2) % 3]
        db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
        labels = db.fit_predict(D)
        nc = len(set(labels)) - (1 if -1 in labels else 0)
        noise = np.sum(labels == -1)

        unique_labels = sorted(set(labels))
        for cl in unique_labels:
            mask = labels == cl
            if cl == -1:
                ax.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=[NOISE_COLOR],
                          s=20, alpha=0.5, marker="x", label=f"noise ({mask.sum()})")
            else:
                c = COLORS[cl % 20]
                ax.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=[c], s=40,
                          edgecolors="k", linewidths=0.3, alpha=0.8,
                          label=f"C{cl} ({mask.sum()})" if cl < 12 else None)
        ax.set_title(f"DBSCAN eps={eps}, ms=2: {nc} cl, {noise} noise", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if nc <= 12:
            ax.legend(fontsize=5, loc="best", ncol=2)

    plt.suptitle("DBSCAN Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_dbscan_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 06_dbscan_analysis.png")


def plot_tsne_perplexity_comparison(tsne_results, ts_list, D, Z_avg):
    """Compare t-SNE at all perplexities, colored by a good clustering."""
    energies = np.array([r["final_energy"] for r in ts_list])
    labels = fcluster(Z_avg, t=0.25, criterion="distance") - 1

    from collections import Counter
    counts = Counter(labels)
    rank = {cl: i for i, (cl, _) in enumerate(counts.most_common())}
    labels_ranked = np.array([rank[l] for l in labels])

    n_perp = len(tsne_results)
    fig, axes = plt.subplots(2, n_perp, figsize=(5 * n_perp, 9))
    if n_perp == 1:
        axes = axes.reshape(2, 1)

    for col, (perp, coords) in enumerate(sorted(tsne_results.items())):
        # Colored by cluster
        ax = axes[0, col]
        nc = len(set(labels_ranked))
        for cl in range(min(nc, 20)):
            mask = labels_ranked == cl
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[COLORS[cl % 20]],
                      s=40, edgecolors="k", linewidths=0.3, alpha=0.8,
                      label=f"C{cl} ({mask.sum()})" if cl < 8 else None)
        ax.set_title(f"t-SNE perp={perp} — clusters", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if nc <= 10:
            ax.legend(fontsize=5, loc="best", ncol=2)

        # Colored by energy
        ax = axes[1, col]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=energies, cmap="RdYlBu_r",
                       s=40, edgecolors="k", linewidths=0.3, alpha=0.85)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f"t-SNE perp={perp} — energy", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("t-SNE Perplexity Comparison (avg linkage t=0.25)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_tsne_perplexity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 07_tsne_perplexity.png")


def analyze_clusters_detail(D, Z_avg, ts_list, thresh=0.25):
    """Detailed analysis of clusters at chosen threshold."""
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    seeds = [r["seed"] for r in ts_list]
    labels = fcluster(Z_avg, t=thresh, criterion="distance")
    n = len(ts_list)

    from collections import Counter
    counts = Counter(labels)
    print(f"\n{'='*70}")
    print(f"Cluster analysis at threshold={thresh} A (average linkage)")
    print(f"{'='*70}")
    print(f"Total TS: {n}, Clusters: {len(counts)}")
    print(f"Cluster sizes: {sorted(counts.values(), reverse=True)}")
    print()

    # Detailed per-cluster stats
    print(f"{'Cl':>3} {'Size':>5} {'E_mean':>10} {'E_std':>8} {'E_range':>8} "
          f"{'eig0_mean':>10} {'RMSD_intra':>11} {'Seeds':>30}")
    cluster_info = []
    for cl in sorted(counts.keys()):
        mask = labels == cl
        cl_energies = energies[mask]
        cl_eig0s = eig0s[mask]
        cl_seeds = [seeds[i] for i in range(n) if mask[i]]

        # Intra-cluster RMSD
        cl_idx = np.where(mask)[0]
        if len(cl_idx) > 1:
            intra_rmsds = []
            for ii in range(len(cl_idx)):
                for jj in range(ii + 1, len(cl_idx)):
                    intra_rmsds.append(D[cl_idx[ii], cl_idx[jj]])
            mean_intra = np.mean(intra_rmsds)
        else:
            mean_intra = 0

        info = {
            "cluster": int(cl),
            "size": int(counts[cl]),
            "energy_mean": float(cl_energies.mean()),
            "energy_std": float(cl_energies.std()) if len(cl_energies) > 1 else 0,
            "energy_range": float(cl_energies.max() - cl_energies.min()) if len(cl_energies) > 1 else 0,
            "eig0_mean": float(cl_eig0s.mean()),
            "rmsd_intra_mean": float(mean_intra),
            "seeds": cl_seeds,
        }
        cluster_info.append(info)

        seeds_str = str(cl_seeds[:6]) + ("..." if len(cl_seeds) > 6 else "")
        print(f"{cl:3d} {counts[cl]:5d} {cl_energies.mean():10.4f} {info['energy_std']:8.4f} "
              f"{info['energy_range']:8.4f} {cl_eig0s.mean():10.4f} {mean_intra:11.4f} {seeds_str}")

    # Save detailed results
    results = {
        "threshold": thresh,
        "n_clusters": len(counts),
        "n_ts": n,
        "clusters": cluster_info,
        "rmsd_matrix": D.tolist(),
        "seeds": seeds,
        "energies": energies.tolist(),
        "labels": labels.tolist(),
    }
    out_path = os.path.join(OUT_DIR, f"clusters_t{thresh}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    return labels, cluster_info


def plot_rmsd_vs_features(D, ts_list):
    """Scatter: pairwise RMSD vs energy difference, eig0 difference."""
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    n = len(ts_list)

    rmsd_flat, de_flat, deig_flat = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            rmsd_flat.append(D[i, j])
            de_flat.append(abs(energies[i] - energies[j]))
            deig_flat.append(abs(eig0s[i] - eig0s[j]))

    rmsd_flat = np.array(rmsd_flat)
    de_flat = np.array(de_flat)
    deig_flat = np.array(deig_flat)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(rmsd_flat, de_flat, s=3, alpha=0.3, color="steelblue")
    ax.set_xlabel("Aligned RMSD (A)")
    ax.set_ylabel("|dE| (eV)")
    ax.set_title("RMSD vs Energy Difference")
    ax.grid(True, alpha=0.3)
    # Correlation
    corr = np.corrcoef(rmsd_flat, de_flat)[0, 1]
    ax.text(0.95, 0.95, f"r={corr:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)

    ax = axes[1]
    ax.scatter(rmsd_flat, deig_flat, s=3, alpha=0.3, color="darkorange")
    ax.set_xlabel("Aligned RMSD (A)")
    ax.set_ylabel("|d(eig0)|")
    ax.set_title("RMSD vs eig0 Difference")
    ax.grid(True, alpha=0.3)
    corr2 = np.corrcoef(rmsd_flat, deig_flat)[0, 1]
    ax.text(0.95, 0.95, f"r={corr2:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)

    # 2D histogram for RMSD vs dE
    ax = axes[2]
    h = ax.hist2d(rmsd_flat, de_flat, bins=50, cmap="Blues", cmin=1)
    plt.colorbar(h[3], ax=ax, shrink=0.8)
    ax.set_xlabel("Aligned RMSD (A)")
    ax.set_ylabel("|dE| (eV)")
    ax.set_title("RMSD vs dE (2D histogram)")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/08_rmsd_vs_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 08_rmsd_vs_features.png")

    print(f"\nCorrelations:")
    print(f"  RMSD vs |dE|: r = {corr:.4f}")
    print(f"  RMSD vs |d(eig0)|: r = {corr2:.4f}")


def plot_cluster_representatives(D, Z_avg, ts_list, mds_coords, thresh=0.25):
    """Show representative TS for each cluster — the medoid."""
    labels = fcluster(Z_avg, t=thresh, criterion="distance")
    energies = np.array([r["final_energy"] for r in ts_list])
    seeds = [r["seed"] for r in ts_list]

    from collections import Counter
    counts = Counter(labels)
    # Only show clusters with >= 2 members, sorted by size
    big_clusters = [cl for cl, cnt in counts.most_common() if cnt >= 2][:12]

    n_cl = len(big_clusters)
    if n_cl == 0:
        return

    # Find medoid for each cluster
    medoids = []
    for cl in big_clusters:
        mask = labels == cl
        idx = np.where(mask)[0]
        sub_D = D[np.ix_(idx, idx)]
        medoid_local = sub_D.sum(axis=1).argmin()
        medoid_global = idx[medoid_local]
        medoids.append(medoid_global)

    # Plot medoids on MDS
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Background: all points in gray
    ax.scatter(mds_coords[:, 0], mds_coords[:, 1], c="lightgray", s=20, alpha=0.4)

    # Highlight clusters and medoids
    for i, cl in enumerate(big_clusters):
        mask = labels == cl
        c = COLORS[i % 20]
        ax.scatter(mds_coords[mask, 0], mds_coords[mask, 1], c=[c], s=50,
                  edgecolors="k", linewidths=0.5, alpha=0.8)
        # Medoid
        mi = medoids[i]
        ax.scatter(mds_coords[mi, 0], mds_coords[mi, 1], c=[c], s=200,
                  edgecolors="black", linewidths=2, marker="*", zorder=10)
        ax.annotate(f"C{cl}\n(n={counts[cl]}, E={energies[mi]:.2f})",
                   xy=(mds_coords[mi, 0], mds_coords[mi, 1]),
                   fontsize=6, ha="center", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    ax.set_title(f"Cluster Representatives (medoids) — threshold={thresh} A", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/09_cluster_representatives.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 09_cluster_representatives.png")


def intercluster_distances(D, Z_avg, ts_list, thresh=0.25):
    """Compute and plot inter-cluster distance matrix."""
    labels = fcluster(Z_avg, t=thresh, criterion="distance")
    from collections import Counter
    counts = Counter(labels)
    clusters = sorted(counts.keys())
    nc = len(clusters)

    # Inter-cluster distances (centroid linkage: mean RMSD between all pairs)
    inter_D = np.zeros((nc, nc))
    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            if i == j:
                # Intra-cluster mean
                idx_i = np.where(labels == ci)[0]
                if len(idx_i) > 1:
                    rmsds = [D[a, b] for a in idx_i for b in idx_i if a < b]
                    inter_D[i, j] = np.mean(rmsds) if rmsds else 0
            else:
                idx_i = np.where(labels == ci)[0]
                idx_j = np.where(labels == cj)[0]
                rmsds = [D[a, b] for a in idx_i for b in idx_j]
                inter_D[i, j] = np.mean(rmsds)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, nc * 0.5), max(6, nc * 0.4)))
    im = ax.imshow(inter_D, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean RMSD (A)")

    tick_labels = [f"C{cl} ({counts[cl]})" for cl in clusters]
    ax.set_xticks(range(nc))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(nc))
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_title(f"Inter-cluster Mean RMSD (threshold={thresh} A)", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/10_intercluster_distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 10_intercluster_distances.png")


def main():
    print("=" * 70)
    print("Structural RMSD Clustering — 87 TS from adaptive run")
    print("=" * 70)

    ts_list = load_ts_data()
    D = compute_rmsd_matrix(ts_list)
    triu = analyze_rmsd_distribution(D, ts_list)

    mds_coords, tsne_results = compute_embeddings(D)
    plot_embeddings_colored_by_features(mds_coords, tsne_results, ts_list, D)

    Z_avg = plot_dendrogram_analysis(D, ts_list)
    threshold_scan(D, Z_avg, ts_list, mds_coords, tsne_results)
    plot_clustering_at_scales(D, Z_avg, ts_list, mds_coords, tsne_results)
    dbscan_analysis(D, ts_list, mds_coords, tsne_results)
    plot_tsne_perplexity_comparison(tsne_results, ts_list, D, Z_avg)
    plot_rmsd_vs_features(D, ts_list)

    # Detailed cluster analysis at 0.25 A
    labels, cluster_info = analyze_clusters_detail(D, Z_avg, ts_list, thresh=0.25)

    plot_cluster_representatives(D, Z_avg, ts_list, mds_coords, thresh=0.25)
    intercluster_distances(D, Z_avg, ts_list, thresh=0.25)

    print(f"\n{'='*70}")
    print(f"All outputs in {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
