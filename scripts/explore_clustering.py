#!/usr/bin/env python
"""Comprehensive clustering exploration for TS geometries.

Tries multiple clustering algorithms, distance metrics, heuristics, and
generates t-SNE/MDS visualizations to compare which methods best correlate
with energy groupings.

Algorithms tested:
  - Hierarchical: complete, average, single, ward linkage
  - DBSCAN with various eps
  - OPTICS (automatic eps)
  - KMeans on RMSD embeddings
  - Spectral clustering on RMSD similarity
  - Energy-based clustering (as ground truth reference)

Evaluation:
  - Adjusted Rand Index vs energy clusters
  - Silhouette score on RMSD matrix
  - Calinski-Harabasz on embeddings
  - Visual comparison via t-SNE and MDS

Output: figures in docs/figures/clustering_exploration/
"""

import json
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================

DATA_PATH = "/scratch/memoozd/diffusion-ts/clustered_results_67ts.json"
OUT_DIR = "/scratch/memoozd/diffusion-ts/clustering_exploration"
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette for clusters (up to 20)
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
NOISE_COLOR = [0.5, 0.5, 0.5, 1.0]  # gray for DBSCAN noise


def load_data():
    """Load RMSD matrix, energies, eig0s from clustered results."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    D = np.array(data["rmsd_matrix"])
    n = D.shape[0]

    energies, eig0s, seeds = [], [], []
    for cl in data["clusters"]:
        for m in cl["members"]:
            energies.append(m["energy"])
            eig0s.append(m["eig0"])
            seeds.append(m["seed"])

    return D, np.array(energies), np.array(eig0s), np.array(seeds), n


def make_energy_clusters(energies, tol=0.1):
    """Create ground-truth clusters by energy proximity.

    Groups points whose energies are within `tol` eV of each other.
    """
    order = np.argsort(energies)
    labels = np.zeros(len(energies), dtype=int)
    current_label = 0
    labels[order[0]] = current_label

    for i in range(1, len(order)):
        if abs(energies[order[i]] - energies[order[i-1]]) > tol:
            current_label += 1
        labels[order[i]] = current_label

    return labels


def compute_embeddings(D, n):
    """Compute MDS and t-SNE 2D embeddings from distance matrix."""
    print("Computing MDS embedding...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", max_iter=1000)
    mds_coords = mds.fit_transform(D)

    print("Computing t-SNE embeddings (multiple perplexities)...")
    tsne_results = {}
    for perp in [5, 10, 15, 30]:
        if perp >= n:
            continue
        tsne = TSNE(n_components=2, metric="precomputed", perplexity=perp,
                    random_state=42, init="random", max_iter=2000)
        tsne_results[perp] = tsne.fit_transform(D)

    return mds_coords, tsne_results


# ============================================================================
# Clustering methods
# ============================================================================

def run_all_clusterings(D, mds_coords, n):
    """Run all clustering algorithms and return {name: labels} dict."""
    results = {}
    D_condensed = squareform(D, checks=False)

    # --- Hierarchical with different linkages and thresholds ---
    for method in ["complete", "average", "single", "ward"]:
        if method == "ward":
            # Ward needs raw features, not distances. Use MDS embedding.
            Z = linkage(mds_coords, method="ward")
        else:
            Z = linkage(D_condensed, method=method)

        for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            labels = fcluster(Z, t=thresh, criterion="distance")
            n_clusters = len(set(labels))
            if 2 <= n_clusters <= n - 1:
                name = f"hier_{method}_t{thresh}"
                results[name] = labels

        # Also try cutting by number of clusters
        for k in [3, 4, 5, 6, 7, 8, 10, 15, 20]:
            if k >= n:
                continue
            labels = fcluster(Z, t=k, criterion="maxclust")
            name = f"hier_{method}_k{k}"
            results[name] = labels

    # --- DBSCAN with various eps ---
    for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for min_samples in [2, 3, 5]:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = db.fit_predict(D)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                name = f"dbscan_eps{eps}_ms{min_samples}"
                results[name] = labels

    # --- OPTICS ---
    for min_samples in [2, 3, 5]:
        try:
            opt = OPTICS(min_samples=min_samples, metric="precomputed")
            labels = opt.fit_predict(D)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                name = f"optics_ms{min_samples}"
                results[name] = labels
        except Exception:
            pass

    # --- KMeans on MDS embedding ---
    for k in [3, 4, 5, 6, 7, 8, 10, 15, 20]:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(mds_coords)
        name = f"kmeans_k{k}"
        results[name] = labels

    # --- Spectral clustering on similarity matrix ---
    sigma_vals = [np.median(D[D > 0]), np.percentile(D[D > 0], 25)]
    for sigma in sigma_vals:
        similarity = np.exp(-D**2 / (2 * sigma**2))
        np.fill_diagonal(similarity, 0)
        for k in [3, 4, 5, 6, 7, 8, 10]:
            try:
                sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                       random_state=42, assign_labels="kmeans")
                labels = sc.fit_predict(similarity)
                name = f"spectral_s{sigma:.2f}_k{k}"
                results[name] = labels
            except Exception:
                pass

    # --- AgglomerativeClustering (sklearn, for connectivity options) ---
    for k in [3, 4, 5, 6, 7, 8, 10]:
        for link in ["complete", "average", "single"]:
            ac = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                        linkage=link)
            labels = ac.fit_predict(D)
            name = f"agglo_{link}_k{k}"
            results[name] = labels

    return results


def evaluate_clusterings(results, D, mds_coords, energy_labels):
    """Evaluate all clusterings against energy clusters and internal metrics."""
    metrics = []

    for name, labels in results.items():
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1) if -1 in labels else 0

        # Adjusted Rand Index vs energy clusters
        ari = adjusted_rand_score(energy_labels, labels)

        # Silhouette score (skip if only 1 cluster or too many noise points)
        valid = labels != -1
        n_valid = valid.sum()
        n_valid_clusters = len(set(labels[valid]))
        if n_valid_clusters >= 2 and n_valid >= n_valid_clusters + 1:
            try:
                sil = silhouette_score(D[np.ix_(valid, valid)], labels[valid],
                                       metric="precomputed")
            except Exception:
                sil = float("nan")
        else:
            sil = float("nan")

        # Calinski-Harabasz on MDS embedding
        if n_valid_clusters >= 2 and n_valid >= n_valid_clusters + 1:
            try:
                ch = calinski_harabasz_score(mds_coords[valid], labels[valid])
            except Exception:
                ch = float("nan")
        else:
            ch = float("nan")

        # Energy coherence: mean energy std within clusters (lower = better)
        energy_stds = []
        for cl in set(labels):
            if cl == -1:
                continue
            mask = labels == cl
            if mask.sum() > 1:
                energy_stds.append(np.std(energies[mask]))
        mean_energy_std = np.mean(energy_stds) if energy_stds else float("nan")

        metrics.append({
            "name": name,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "ari": ari,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "mean_energy_std": mean_energy_std,
        })

    return sorted(metrics, key=lambda x: -x["ari"])


# ============================================================================
# Visualization
# ============================================================================

def plot_embedding_comparison(mds_coords, tsne_results, labels, title, filename):
    """Plot MDS + t-SNE side by side with given cluster labels."""
    n_plots = 1 + len(tsne_results)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    def scatter_labeled(ax, coords, labels, title_str):
        unique = sorted(set(labels))
        for cl in unique:
            mask = labels == cl
            color = NOISE_COLOR if cl == -1 else COLORS[cl % len(COLORS)]
            label_str = "noise" if cl == -1 else f"C{cl}"
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=40,
                      alpha=0.8, edgecolors="k", linewidths=0.3, label=label_str)
        ax.set_title(title_str, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if len(unique) <= 12:
            ax.legend(fontsize=6, loc="best", ncol=2)

    scatter_labeled(axes[0], mds_coords, labels, "MDS")

    for i, (perp, coords) in enumerate(sorted(tsne_results.items())):
        scatter_labeled(axes[i + 1], coords, labels, f"t-SNE (perp={perp})")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", bbox_inches="tight", dpi=150)
    plt.close()


def plot_top_methods_grid(mds_coords, tsne_best, results, metrics_sorted,
                          energy_labels, energies):
    """Grid: top N methods on MDS and best t-SNE, plus energy coloring."""
    top_n = 6
    top = metrics_sorted[:top_n]

    fig = plt.figure(figsize=(20, 5 * ((top_n + 2) // 3)))
    ncols = 3
    nrows = max((top_n + 2 + ncols - 1) // ncols, 1)

    # First two panels: energy-colored
    ax1 = fig.add_subplot(nrows, ncols, 1)
    sc = ax1.scatter(mds_coords[:, 0], mds_coords[:, 1], c=energies, cmap="viridis",
                     s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax1, label="Energy (eV)", shrink=0.8)
    ax1.set_title("MDS — colored by energy", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(nrows, ncols, 2)
    sc2 = ax2.scatter(mds_coords[:, 0], mds_coords[:, 1], c=energy_labels, cmap="tab10",
                      s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
    ax2.set_title("MDS — energy clusters (ground truth)", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Top methods
    for i, m in enumerate(top):
        ax = fig.add_subplot(nrows, ncols, i + 3)
        labels = results[m["name"]]
        unique = sorted(set(labels))
        for cl in unique:
            mask = labels == cl
            color = NOISE_COLOR if cl == -1 else COLORS[cl % len(COLORS)]
            ax.scatter(mds_coords[mask, 0], mds_coords[mask, 1], c=[color], s=40,
                      alpha=0.8, edgecolors="k", linewidths=0.3)
        title = (f"{m['name']}\nARI={m['ari']:.3f} Sil={m['silhouette']:.3f} "
                f"k={m['n_clusters']}")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Top Clustering Methods (MDS embedding)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top_methods_mds_grid.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_tsne_perplexity_comparison(tsne_results, energy_labels, energies):
    """Compare t-SNE perplexities with energy coloring."""
    perps = sorted(tsne_results.keys())
    fig, axes = plt.subplots(2, len(perps), figsize=(5 * len(perps), 9))

    for i, perp in enumerate(perps):
        coords = tsne_results[perp]
        # Energy coloring
        sc = axes[0, i].scatter(coords[:, 0], coords[:, 1], c=energies, cmap="viridis",
                                s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
        axes[0, i].set_title(f"t-SNE perp={perp}\ncolored by energy", fontsize=10)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Energy cluster coloring
        axes[1, i].scatter(coords[:, 0], coords[:, 1], c=energy_labels, cmap="tab10",
                          s=40, alpha=0.8, edgecolors="k", linewidths=0.3)
        axes[1, i].set_title(f"t-SNE perp={perp}\nenergy clusters", fontsize=10)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.colorbar(sc, ax=axes[0, -1], label="Energy (eV)", shrink=0.8)
    plt.suptitle("t-SNE Perplexity Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/tsne_perplexity_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_metrics_comparison(metrics_sorted):
    """Bar chart of top methods by ARI, silhouette, energy coherence."""
    top = [m for m in metrics_sorted if not np.isnan(m["silhouette"])][:20]
    if not top:
        return

    names = [m["name"] for m in top]
    aris = [m["ari"] for m in top]
    sils = [m["silhouette"] for m in top]
    estds = [m["mean_energy_std"] for m in top]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ARI
    colors_ari = plt.cm.RdYlGn(np.array(aris) / max(max(aris), 0.01))
    axes[0].barh(range(len(names)), aris, color=colors_ari, edgecolor="k", linewidth=0.3)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlabel("Adjusted Rand Index (vs energy clusters)")
    axes[0].set_title("ARI: Agreement with Energy-Based Clustering")
    axes[0].invert_yaxis()

    # Silhouette
    sils_arr = np.array(sils)
    colors_sil = plt.cm.RdYlGn((sils_arr - sils_arr.min()) / max(sils_arr.max() - sils_arr.min(), 0.01))
    axes[1].barh(range(len(names)), sils, color=colors_sil, edgecolor="k", linewidth=0.3)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].set_xlabel("Silhouette Score")
    axes[1].set_title("Silhouette: Internal Cluster Quality")
    axes[1].invert_yaxis()

    # Energy coherence
    estds_arr = np.array(estds)
    valid_estds = estds_arr[~np.isnan(estds_arr)]
    if len(valid_estds) > 0:
        colors_est = plt.cm.RdYlGn_r((estds_arr - np.nanmin(estds_arr)) /
                                       max(np.nanmax(estds_arr) - np.nanmin(estds_arr), 0.01))
        axes[2].barh(range(len(names)), estds, color=colors_est, edgecolor="k", linewidth=0.3)
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels(names, fontsize=7)
    axes[2].set_xlabel("Mean within-cluster energy std (eV) — lower is better")
    axes[2].set_title("Energy Coherence: How well clusters group same-energy TS")
    axes[2].invert_yaxis()

    plt.suptitle("Clustering Method Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/metrics_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_best_method_detail(mds_coords, tsne_results, results, best_name,
                            energies, eig0s, energy_labels):
    """Detailed view of the best clustering method."""
    labels = results[best_name]
    best_perp = max(tsne_results.keys()) if tsne_results else None

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)

    # MDS colored by best method
    ax1 = fig.add_subplot(gs[0, 0])
    unique = sorted(set(labels))
    for cl in unique:
        mask = labels == cl
        color = NOISE_COLOR if cl == -1 else COLORS[cl % len(COLORS)]
        n_in = mask.sum()
        ax1.scatter(mds_coords[mask, 0], mds_coords[mask, 1], c=[color], s=50,
                   alpha=0.8, edgecolors="k", linewidths=0.3, label=f"C{cl} (n={n_in})")
    ax1.set_title(f"MDS — {best_name}", fontsize=10)
    ax1.legend(fontsize=6, loc="best", ncol=2)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # MDS colored by energy
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(mds_coords[:, 0], mds_coords[:, 1], c=energies, cmap="viridis",
                     s=50, alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax2, label="Energy (eV)", shrink=0.8)
    ax2.set_title("MDS — Energy", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # MDS colored by eig0
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter(mds_coords[:, 0], mds_coords[:, 1], c=eig0s, cmap="coolwarm",
                      s=50, alpha=0.8, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc3, ax=ax3, label="eig0 (eV/A^2)", shrink=0.8)
    ax3.set_title("MDS — Lowest eigenvalue", fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # t-SNE colored by best method
    if best_perp and best_perp in tsne_results:
        tsne_coords = tsne_results[best_perp]
        ax4 = fig.add_subplot(gs[1, 0])
        for cl in unique:
            mask = labels == cl
            color = NOISE_COLOR if cl == -1 else COLORS[cl % len(COLORS)]
            ax4.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], c=[color], s=50,
                       alpha=0.8, edgecolors="k", linewidths=0.3)
        ax4.set_title(f"t-SNE (perp={best_perp}) — {best_name}", fontsize=10)
        ax4.set_xticks([])
        ax4.set_yticks([])

        # t-SNE colored by energy
        ax5 = fig.add_subplot(gs[1, 1])
        sc5 = ax5.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=energies, cmap="viridis",
                          s=50, alpha=0.8, edgecolors="k", linewidths=0.3)
        plt.colorbar(sc5, ax=ax5, label="Energy (eV)", shrink=0.8)
        ax5.set_title(f"t-SNE (perp={best_perp}) — Energy", fontsize=10)
        ax5.set_xticks([])
        ax5.set_yticks([])

    # Energy vs eig0 colored by cluster
    ax6 = fig.add_subplot(gs[1, 2])
    for cl in unique:
        mask = labels == cl
        color = NOISE_COLOR if cl == -1 else COLORS[cl % len(COLORS)]
        ax6.scatter(energies[mask], eig0s[mask], c=[color], s=50,
                   alpha=0.8, edgecolors="k", linewidths=0.3, label=f"C{cl}")
    ax6.set_xlabel("Energy (eV)")
    ax6.set_ylabel("Lowest eigenvalue (eV/A^2)")
    ax6.set_title("Energy vs Curvature — cluster colors")
    if len(unique) <= 12:
        ax6.legend(fontsize=6, loc="best", ncol=2)

    plt.suptitle(f"Best Method: {best_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/best_method_detail.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_energy_vs_structure_scatter(D, energies, eig0s):
    """Scatter: pairwise RMSD vs pairwise energy difference."""
    n = len(energies)
    rmsds, de, deig = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            rmsds.append(D[i, j])
            de.append(abs(energies[i] - energies[j]))
            deig.append(abs(eig0s[i] - eig0s[j]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(rmsds, de, s=5, alpha=0.3, c="steelblue")
    axes[0].set_xlabel("Pairwise aligned RMSD (A)")
    axes[0].set_ylabel("|Energy difference| (eV)")
    axes[0].set_title("Structure distance vs Energy distance")

    axes[1].scatter(rmsds, deig, s=5, alpha=0.3, c="coral")
    axes[1].set_xlabel("Pairwise aligned RMSD (A)")
    axes[1].set_ylabel("|eig0 difference| (eV/A^2)")
    axes[1].set_title("Structure distance vs Curvature distance")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rmsd_vs_energy_scatter.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_rmsd_band_analysis(D, energies):
    """Analyze RMSD distribution within and between energy groups."""
    energy_labels = make_energy_clusters(energies, tol=0.1)

    intra_rmsds = []
    inter_rmsds = []
    n = len(energies)
    for i in range(n):
        for j in range(i+1, n):
            if energy_labels[i] == energy_labels[j]:
                intra_rmsds.append(D[i, j])
            else:
                inter_rmsds.append(D[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(intra_rmsds, bins=40, alpha=0.7, color="steelblue", edgecolor="k",
                linewidth=0.3, label=f"Intra-group (n={len(intra_rmsds)})")
    axes[0].hist(inter_rmsds, bins=40, alpha=0.5, color="coral", edgecolor="k",
                linewidth=0.3, label=f"Inter-group (n={len(inter_rmsds)})")
    axes[0].set_xlabel("Aligned RMSD (A)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Intra- vs Inter-Energy-Group RMSD")
    axes[0].legend()

    # Box plot per energy group pair
    unique_labels = sorted(set(energy_labels))
    group_data = {}
    for li in unique_labels:
        for lj in unique_labels:
            if li > lj:
                continue
            key = f"{li}-{lj}" if li != lj else f"{li} (intra)"
            vals = []
            mask_i = energy_labels == li
            mask_j = energy_labels == lj
            idx_i = np.where(mask_i)[0]
            idx_j = np.where(mask_j)[0]
            for ii in idx_i:
                for jj in idx_j:
                    if ii < jj:
                        vals.append(D[ii, jj])
            if vals:
                group_data[key] = vals

    # Show only intra groups
    intra_keys = [k for k in group_data if "(intra)" in k]
    if intra_keys:
        bp_data = [group_data[k] for k in intra_keys]
        bp = axes[1].boxplot(bp_data, labels=[k.replace(" (intra)", "") for k in intra_keys],
                            patch_artist=True)
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
        axes[1].set_xlabel("Energy group")
        axes[1].set_ylabel("Intra-group RMSD (A)")
        axes[1].set_title("Within-group RMSD spread")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rmsd_band_analysis.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cluster_count_histogram(results, metrics_sorted):
    """Histogram of cluster counts across all methods — is there a natural k?"""
    all_k = [m["n_clusters"] for m in metrics_sorted]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Raw histogram
    axes[0].hist(all_k, bins=range(1, max(all_k) + 2), color="steelblue",
                edgecolor="k", linewidth=0.3, alpha=0.8)
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Number of methods producing k clusters")
    axes[0].set_title("Cluster count distribution across all methods")

    # Silhouette vs k
    k_vals = sorted(set(all_k))
    sil_by_k = {k: [] for k in k_vals}
    for m in metrics_sorted:
        if not np.isnan(m["silhouette"]):
            sil_by_k[m["n_clusters"]].append(m["silhouette"])

    k_plot = [k for k in k_vals if sil_by_k[k]]
    sil_means = [np.mean(sil_by_k[k]) for k in k_plot]
    sil_stds = [np.std(sil_by_k[k]) for k in k_plot]

    axes[1].errorbar(k_plot, sil_means, yerr=sil_stds, fmt="o-", capsize=3,
                    color="steelblue", markersize=5)
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Mean silhouette score")
    axes[1].set_title("Silhouette score vs cluster count")
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Good clustering threshold")
    axes[1].legend(fontsize=8)

    plt.suptitle("Is there a natural cluster count?", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cluster_count_analysis.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_cross_method_agreement(results, D, mds_coords):
    """Pairwise ARI between all methods — do methods agree with each other?"""
    # Pick a diverse subset of methods
    method_families = {}
    for name in results:
        if name.startswith("hier_complete_k"):
            method_families.setdefault("hier_complete", []).append(name)
        elif name.startswith("hier_average_k"):
            method_families.setdefault("hier_average", []).append(name)
        elif name.startswith("dbscan"):
            method_families.setdefault("dbscan", []).append(name)
        elif name.startswith("kmeans"):
            method_families.setdefault("kmeans", []).append(name)
        elif name.startswith("spectral"):
            method_families.setdefault("spectral", []).append(name)
        elif name.startswith("optics"):
            method_families.setdefault("optics", []).append(name)

    # Select representative from each family at k~7-8
    selected = []
    for fam, names in method_families.items():
        # Prefer k=7 or k=8
        for target_k in ["k7", "k8", "k5", "k10"]:
            matches = [n for n in names if target_k in n]
            if matches:
                selected.append(matches[0])
                break
        else:
            selected.append(names[0])

    # Add a few specific ones
    for extra in ["hier_complete_t0.2", "hier_average_t0.2", "hier_complete_t0.3"]:
        if extra in results and extra not in selected:
            selected.append(extra)

    selected = selected[:12]  # limit

    # Compute pairwise ARI
    n_sel = len(selected)
    ari_matrix = np.zeros((n_sel, n_sel))
    for i in range(n_sel):
        for j in range(n_sel):
            ari_matrix[i, j] = adjusted_rand_score(results[selected[i]], results[selected[j]])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ari_matrix, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Adjusted Rand Index", shrink=0.8)
    ax.set_xticks(range(n_sel))
    ax.set_xticklabels([s.replace("hier_", "h_").replace("spectral_", "sp_") for s in selected],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_sel))
    ax.set_yticklabels([s.replace("hier_", "h_").replace("spectral_", "sp_") for s in selected],
                       fontsize=7)
    ax.set_title("Cross-Method Agreement (Pairwise ARI)", fontsize=12)

    # Annotate
    for i in range(n_sel):
        for j in range(n_sel):
            ax.text(j, i, f"{ari_matrix[i, j]:.2f}", ha="center", va="center",
                   fontsize=6, color="black" if ari_matrix[i, j] > 0.3 else "gray")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cross_method_agreement.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Print agreement summary
    off_diag = ari_matrix[np.triu_indices(n_sel, k=1)]
    print(f"\n  Cross-method ARI: mean={off_diag.mean():.3f}, "
          f"max={off_diag.max():.3f}, min={off_diag.min():.3f}")
    # Find most agreeing pair
    idx = np.unravel_index(np.argmax(ari_matrix - np.eye(n_sel) * 2), ari_matrix.shape)
    print(f"  Most agreeing: {selected[idx[0]]} ↔ {selected[idx[1]]} (ARI={ari_matrix[idx]:.3f})")


def plot_dendrogram_with_features(D, energies, eig0s):
    """Dendrogram with energy and eig0 color bars to show structure."""
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax_dend = fig.add_subplot(gs[0])
    dn = dendrogram(Z, ax=ax_dend, color_threshold=0.2,
                    above_threshold_color="#999", leaf_rotation=90, leaf_font_size=6)
    ax_dend.axhline(y=0.2, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_dend.set_ylabel("RMSD (A)")
    ax_dend.set_title("Hierarchical Clustering (average linkage)")

    # Feature bars below dendrogram
    leaf_order = dn["leaves"]
    ax_feat = fig.add_subplot(gs[1])
    e_ordered = energies[leaf_order]
    eig_ordered = eig0s[leaf_order]

    # Normalize for color mapping
    ax_feat.scatter(range(len(leaf_order)), [0.7] * len(leaf_order),
                   c=e_ordered, cmap="viridis", s=30, marker="s")
    ax_feat.scatter(range(len(leaf_order)), [0.3] * len(leaf_order),
                   c=eig_ordered, cmap="coolwarm", s=30, marker="s")
    ax_feat.set_xlim(-0.5, len(leaf_order) - 0.5)
    ax_feat.set_ylim(0, 1)
    ax_feat.set_yticks([0.3, 0.7])
    ax_feat.set_yticklabels(["eig0", "Energy"])
    ax_feat.set_xlabel("TS (ordered by dendrogram)")

    plt.savefig(f"{OUT_DIR}/dendrogram_with_features.png", bbox_inches="tight", dpi=150)
    plt.close()


def analyze_natural_clusters(D, energies, eig0s):
    """Try to find natural number of clusters via multiple heuristics."""
    D_condensed = squareform(D, checks=False)

    print("\n  === Natural cluster count analysis ===")

    # Gap statistic approach: silhouette vs k for hierarchical average
    Z = linkage(D_condensed, method="average")
    sil_scores = []
    for k in range(2, 30):
        labels = fcluster(Z, t=k, criterion="maxclust")
        try:
            s = silhouette_score(D, labels, metric="precomputed")
            sil_scores.append((k, s))
        except Exception:
            pass

    if sil_scores:
        best_k, best_s = max(sil_scores, key=lambda x: x[1])
        print(f"  Hierarchical avg: best silhouette k={best_k} (sil={best_s:.3f})")
        for k, s in sil_scores[:15]:
            bar = "#" * int(s * 40)
            print(f"    k={k:2d}: sil={s:.3f} {bar}")

    # DBSCAN natural clustering
    for eps in [0.05, 0.1, 0.15, 0.2, 0.3]:
        db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
        labels = db.fit_predict(D)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        if n_clusters >= 2:
            s = silhouette_score(D[labels != -1][:, labels != -1],
                                labels[labels != -1], metric="precomputed")
            print(f"  DBSCAN eps={eps}: k={n_clusters}, noise={n_noise}, sil={s:.3f}")

    return sil_scores


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Clustering Exploration for TS Geometries")
    print("=" * 70)

    D, energies, eig0s, seeds, n = load_data()
    print(f"Loaded {n} TS geometries")
    print(f"Energy range: {energies.min():.2f} to {energies.max():.2f} eV")
    print(f"eig0 range: {eig0s.min():.2f} to {eig0s.max():.2f} eV/A^2")

    # Ground truth: energy clusters
    energy_labels = make_energy_clusters(energies, tol=0.1)
    n_energy_clusters = len(set(energy_labels))
    print(f"\nEnergy-based clusters (tol=0.1 eV): {n_energy_clusters} groups")
    for cl in sorted(set(energy_labels)):
        mask = energy_labels == cl
        e_mean = energies[mask].mean()
        print(f"  Group {cl}: n={mask.sum()}, E={e_mean:.4f} eV")

    # Compute embeddings
    mds_coords, tsne_results = compute_embeddings(D, n)

    # Run all clustering methods
    print(f"\nRunning clustering algorithms...")
    results = run_all_clusterings(D, mds_coords, n)
    print(f"  Tested {len(results)} configurations")

    # Evaluate
    print(f"\nEvaluating...")
    metrics_sorted = evaluate_clusterings(results, D, mds_coords, energy_labels)

    # Print top 20
    print(f"\n{'='*90}")
    print(f"{'Method':<40} {'k':>3} {'ARI':>7} {'Sil':>7} {'CH':>8} {'E_std':>7}")
    print(f"{'='*90}")
    for m in metrics_sorted[:30]:
        print(f"{m['name']:<40} {m['n_clusters']:>3} {m['ari']:>7.3f} "
              f"{m['silhouette']:>7.3f} {m['calinski_harabasz']:>8.1f} "
              f"{m['mean_energy_std']:>7.4f}")

    # Identify best by different criteria
    best_ari = metrics_sorted[0]
    best_sil = max([m for m in metrics_sorted if not np.isnan(m["silhouette"])],
                   key=lambda x: x["silhouette"], default=None)
    best_estd = min([m for m in metrics_sorted if not np.isnan(m["mean_energy_std"])],
                    key=lambda x: x["mean_energy_std"], default=None)

    print(f"\n--- Best by ARI: {best_ari['name']} (ARI={best_ari['ari']:.3f})")
    if best_sil:
        print(f"--- Best by Silhouette: {best_sil['name']} (Sil={best_sil['silhouette']:.3f})")
    if best_estd:
        print(f"--- Best by Energy Coherence: {best_estd['name']} (E_std={best_estd['mean_energy_std']:.4f})")

    # Natural cluster analysis
    print(f"\n--- Natural cluster count analysis ---")
    sil_scores = analyze_natural_clusters(D, energies, eig0s)

    # Generate all figures
    print(f"\nGenerating figures...")

    # 1. Energy-structure relationship
    print("  1/10 RMSD vs energy scatter...")
    plot_energy_vs_structure_scatter(D, energies, eig0s)

    # 2. RMSD band analysis
    print("  2/10 RMSD band analysis...")
    plot_rmsd_band_analysis(D, energies)

    # 3. t-SNE perplexity comparison
    print("  3/10 t-SNE perplexity comparison...")
    if tsne_results:
        plot_tsne_perplexity_comparison(tsne_results, energy_labels, energies)

    # 4. Top methods grid
    print("  4/10 Top methods grid...")
    plot_top_methods_grid(mds_coords,
                         tsne_results.get(max(tsne_results.keys())) if tsne_results else None,
                         results, metrics_sorted, energy_labels, energies)

    # 5. Metrics comparison bars
    print("  5/10 Metrics comparison...")
    plot_metrics_comparison(metrics_sorted)

    # 6. Best method detail
    print("  6/10 Best method detail...")
    plot_best_method_detail(mds_coords, tsne_results, results, best_ari["name"],
                           energies, eig0s, energy_labels)

    # 7. Energy clusters reference
    print("  7/10 Energy cluster reference plots...")
    for perp, coords in tsne_results.items():
        plot_embedding_comparison(mds_coords, {perp: coords}, energy_labels,
                                f"Energy Clusters (perp={perp})",
                                f"energy_clusters_perp{perp}.png")

    # 8. Cluster count analysis
    print("  8/10 Cluster count histogram...")
    plot_cluster_count_histogram(results, metrics_sorted)

    # 9. Cross-method agreement
    print("  9/10 Cross-method agreement...")
    plot_cross_method_agreement(results, D, mds_coords)

    # 10. Dendrogram with feature bars
    print("  10/10 Dendrogram with features...")
    plot_dendrogram_with_features(D, energies, eig0s)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print(f"\nDone! {len(results)} methods tested, figures generated.")

    # Save metrics to JSON for reference
    # Convert numpy types to native Python for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_native = [
        {k: to_native(v) for k, v in m.items()} for m in metrics_sorted
    ]
    with open(f"{OUT_DIR}/clustering_metrics.json", "w") as f:
        json.dump(metrics_native, f, indent=2)
    print(f"Metrics saved to {OUT_DIR}/clustering_metrics.json")
