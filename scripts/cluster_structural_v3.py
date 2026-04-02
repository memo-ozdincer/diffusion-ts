#!/usr/bin/env python
"""Structural clustering v3: refined two-level + structural characterization.

Improvements over v2:
- L2=0.4 A (avoids mega-cluster from 0.5 A)
- Characterize each family by bond distances/angles
- Assign singletons to nearest family (soft assignment)
- Better labeling and visualization
"""

import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE, MDS
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.dependencies.alignment import aligned_rmsd

DATA_PATH = "/scratch/memoozd/diffusion-ts/adaptive_results.json"
RMSD_CACHE = "/scratch/memoozd/diffusion-ts/structural_clustering/rmsd_matrix_87.npy"
SEEDS_CACHE = "/scratch/memoozd/diffusion-ts/structural_clustering/seeds_87.json"
OUT_DIR = "/scratch/memoozd/diffusion-ts/structural_clustering_v3"
os.makedirs(OUT_DIR, exist_ok=True)

EQUIV_CLASSES = {
    "central_C": [0], "O": [3], "H_on_C": [4], "H_on_O": [5],
    "methyl_C": [1, 2], "H_methyl": [6, 7, 8, 9, 10, 11],
}
METHYL_CARBONS = [1, 2]
METHYL_HYDROGENS = [[6, 7, 8], [9, 10, 11]]

# Isopropanol atom labels
ATOM_LABELS = ["C0(central)", "C1(methyl)", "C2(methyl)", "O", "H(C0)", "H(O)",
               "H1a", "H1b", "H1c", "H2a", "H2b", "H2c"]
ATOM_SYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"]

# Key bond pairs for structural characterization
KEY_BONDS = [
    (0, 1, "C0-C1"), (0, 2, "C0-C2"), (0, 3, "C0-O"), (0, 4, "C0-H"),
    (3, 5, "O-H"), (1, 2, "C1-C2"),
]

# Key angles (i, j, k) = angle at j between i-j-k
KEY_ANGLES = [
    (1, 0, 2, "C1-C0-C2"), (1, 0, 3, "C1-C0-O"), (2, 0, 3, "C2-C0-O"),
    (0, 3, 5, "C0-O-H"),
]

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    ts_list = [r for r in data if r["status"] == "ts" and "ts_coords" in r]
    D = np.load(RMSD_CACHE)
    with open(SEEDS_CACHE) as f:
        cached_seeds = json.load(f)
    assert cached_seeds == [r["seed"] for r in ts_list]
    return ts_list, D


def compute_bond_distance(coords, i, j):
    return np.linalg.norm(coords[i] - coords[j])


def compute_angle(coords, i, j, k):
    """Angle at j between i-j and k-j, in degrees."""
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def structural_features(coords):
    """Extract key bond distances and angles from a geometry."""
    bonds = {name: compute_bond_distance(coords, i, j) for i, j, name in KEY_BONDS}
    angles = {name: compute_angle(coords, i, j, k) for i, j, k, name in KEY_ANGLES}
    return bonds, angles


def two_level_clustering(D, ts_list, L1_thresh=0.15, L2_thresh=0.4):
    """Two-level hierarchical clustering."""
    n = len(ts_list)
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")

    # Level 1: tight cores
    labels_L1 = fcluster(Z, t=L1_thresh, criterion="distance")
    counts_L1 = Counter(labels_L1)
    core_ids = sorted(counts_L1.keys())
    nc = len(core_ids)

    # Core-to-core distance matrix
    core_D = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(i + 1, nc):
            idx_i = np.where(labels_L1 == core_ids[i])[0]
            idx_j = np.where(labels_L1 == core_ids[j])[0]
            rmsds = [D[a, b] for a in idx_i for b in idx_j]
            core_D[i, j] = np.mean(rmsds)
            core_D[j, i] = core_D[i, j]

    # Level 2: family grouping
    core_condensed = squareform(core_D, checks=False)
    Z_core = linkage(core_condensed, method="average")
    fam_labels = fcluster(Z_core, t=L2_thresh, criterion="distance")

    # Map to individual TS
    ts_family = np.zeros(n, dtype=int)
    for i, cl in enumerate(core_ids):
        mask = labels_L1 == cl
        ts_family[mask] = fam_labels[i]

    return labels_L1, ts_family, Z, Z_core, core_ids, core_D


def assign_singletons(ts_family, D, min_family_size=2):
    """Soft-assign singletons to nearest multi-member family."""
    n = len(ts_family)
    fam_counts = Counter(ts_family)
    multi_fams = {f for f, c in fam_counts.items() if c >= min_family_size}

    soft_assignment = ts_family.copy()
    assignment_dist = np.zeros(n)
    is_soft = np.zeros(n, dtype=bool)

    for i in range(n):
        if fam_counts[ts_family[i]] >= min_family_size:
            # Already in a multi-member family
            fam_idx = np.where(ts_family == ts_family[i])[0]
            fam_idx = fam_idx[fam_idx != i]
            assignment_dist[i] = D[i, fam_idx].mean() if len(fam_idx) > 0 else 0
        else:
            # Singleton — find nearest multi-member family
            best_fam = None
            best_dist = float("inf")
            for f in multi_fams:
                fam_idx = np.where(ts_family == f)[0]
                mean_dist = D[i, fam_idx].mean()
                if mean_dist < best_dist:
                    best_dist = mean_dist
                    best_fam = f
            if best_fam is not None:
                soft_assignment[i] = best_fam
                assignment_dist[i] = best_dist
                is_soft[i] = True

    return soft_assignment, assignment_dist, is_soft


def plot_main_overview(ts_family, soft_assignment, is_soft, ts_list, D):
    """Main overview figure with hard and soft assignments."""
    n = len(ts_list)
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])

    fam_counts = Counter(ts_family)
    fam_rank = {f: i for i, (f, _) in enumerate(fam_counts.most_common())}

    # Embeddings
    tsne = TSNE(n_components=2, metric="precomputed", perplexity=10,
                random_state=42, init="random", max_iter=3000)
    tsne_xy = tsne.fit_transform(D)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", max_iter=1000)
    mds_xy = mds.fit_transform(D)

    fig = plt.figure(figsize=(24, 18))

    # Row 1: t-SNE hard clusters + soft assignment + energy
    ax = fig.add_subplot(3, 3, 1)
    _scatter_families(ax, tsne_xy, ts_family, fam_rank, fam_counts, is_soft=None)
    ax.set_title("t-SNE (p=10) — Hard families (L2=0.4 A)", fontsize=9)

    ax = fig.add_subplot(3, 3, 2)
    _scatter_families(ax, tsne_xy, soft_assignment, fam_rank, fam_counts, is_soft=is_soft)
    ax.set_title("t-SNE — With singleton reassignment", fontsize=9)

    ax = fig.add_subplot(3, 3, 3)
    sc = ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=energies, cmap="RdYlBu_r",
                   s=50, edgecolors="k", linewidths=0.3, alpha=0.85)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title("t-SNE — Energy (eV)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2: MDS versions
    ax = fig.add_subplot(3, 3, 4)
    _scatter_families(ax, mds_xy, ts_family, fam_rank, fam_counts, is_soft=None)
    ax.set_title("MDS — Hard families", fontsize=9)

    ax = fig.add_subplot(3, 3, 5)
    _scatter_families(ax, mds_xy, soft_assignment, fam_rank, fam_counts, is_soft=is_soft)
    ax.set_title("MDS — With singleton reassignment", fontsize=9)

    ax = fig.add_subplot(3, 3, 6)
    sc = ax.scatter(mds_xy[:, 0], mds_xy[:, 1], c=eig0s, cmap="RdYlBu_r",
                   s=50, edgecolors="k", linewidths=0.3, alpha=0.85)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title("MDS — eig0 (TS curvature)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 3: Energy boxplots + family size + RMSD heatmap
    ax = fig.add_subplot(3, 3, 7)
    big_fams = [(f, c) for f, c in fam_counts.most_common() if c >= 2]
    if big_fams:
        data_list = [energies[ts_family == f] for f, _ in big_fams]
        labels_list = [f"F{fam_rank[f]} (n={c})" for f, c in big_fams]
        bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[fam_rank[big_fams[i][0]] % 20])
            patch.set_alpha(0.6)
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Energy by Family (n>=2)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(3, 3, 8)
    sizes = sorted(fam_counts.values(), reverse=True)
    bars = ax.bar(range(len(sizes)), sizes,
                  color=[COLORS[i % 20] for i in range(len(sizes))],
                  edgecolor="k", linewidth=0.3)
    ax.axhline(y=2, color="red", ls="--", alpha=0.5, label="n=2 threshold")
    ax.set_xlabel("Family rank")
    ax.set_ylabel("Size")
    n_multi = sum(1 for s in sizes if s >= 2)
    n_single = sum(1 for s in sizes if s == 1)
    ax.set_title(f"{len(fam_counts)} families: {n_multi} multi + {n_single} singleton", fontsize=9)
    ax.legend(fontsize=7)

    # Sorted RMSD heatmap
    ax = fig.add_subplot(3, 3, 9)
    sort_key = [(fam_rank.get(ts_family[i], 99), energies[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: sort_key[i])
    D_sorted = D[np.ix_(order, order)]
    im = ax.imshow(D_sorted, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label="RMSD (A)")
    # Family boundaries
    current_fam = sort_key[order[0]][0]
    for i in range(1, n):
        if sort_key[order[i]][0] != current_fam:
            ax.axhline(y=i - 0.5, color="white", lw=1)
            ax.axvline(x=i - 0.5, color="white", lw=1)
            current_fam = sort_key[order[i]][0]
    ax.set_title("Pairwise RMSD (sorted by family)", fontsize=9)

    plt.suptitle("Structural Clustering v3: 87 TS → Families (L1=0.15 A, L2=0.4 A)",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_main_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 01_main_overview.png")

    return tsne_xy, mds_xy


def _scatter_families(ax, coords, labels, fam_rank, fam_counts, is_soft=None):
    """Helper: scatter plot colored by family."""
    n_fam = len(fam_rank)
    for rank in range(min(n_fam, 20)):
        fam_id = [f for f, r in fam_rank.items() if r == rank][0]
        mask = labels == fam_id
        if is_soft is not None:
            hard = mask & ~is_soft
            soft = mask & is_soft
        else:
            hard = mask
            soft = np.zeros_like(mask)

        cnt_total = mask.sum()
        c = COLORS[rank % 20]

        if hard.any():
            marker = "o" if fam_counts.get(fam_id, 0) > 1 else "x"
            size = 60 if fam_counts.get(fam_id, 0) > 1 else 20
            label = f"F{rank} ({cnt_total})" if rank < 10 and cnt_total > 1 else None
            ax.scatter(coords[hard, 0], coords[hard, 1], c=[c], s=size,
                      marker=marker, edgecolors="k", linewidths=0.3, alpha=0.85,
                      label=label)
        if soft.any():
            ax.scatter(coords[soft, 0], coords[soft, 1], c=[c], s=30,
                      marker="d", edgecolors="k", linewidths=0.5, alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    if n_fam <= 15:
        ax.legend(fontsize=5, loc="best", ncol=2, framealpha=0.8)


def plot_structural_fingerprints(ts_family, ts_list, D):
    """Characterize each family by bond distances and angles."""
    n = len(ts_list)
    energies = np.array([r["final_energy"] for r in ts_list])
    fam_counts = Counter(ts_family)
    fam_rank = {f: i for i, (f, _) in enumerate(fam_counts.most_common())}

    # Compute structural features for all TS
    all_bonds = []
    all_angles = []
    for r in ts_list:
        coords = np.array(r["ts_coords"])
        bonds, angles = structural_features(coords)
        all_bonds.append(bonds)
        all_angles.append(angles)

    bond_names = [name for _, _, name in KEY_BONDS]
    angle_names = [name for _, _, _, name in KEY_ANGLES]

    # Plot bond distance distributions per family
    big_fams = [(f, c) for f, c in fam_counts.most_common() if c >= 3]
    n_fams = len(big_fams)

    if n_fams == 0:
        print("No families with >= 3 members for fingerprints")
        return

    fig, axes = plt.subplots(len(bond_names), 1, figsize=(14, 3 * len(bond_names)))

    for b_idx, bname in enumerate(bond_names):
        ax = axes[b_idx]
        for f_idx, (fam_id, cnt) in enumerate(big_fams):
            mask = ts_family == fam_id
            vals = [all_bonds[i][bname] for i in range(n) if mask[i]]
            rank = fam_rank[fam_id]
            ax.hist(vals, bins=20, alpha=0.5, color=COLORS[rank % 20],
                   label=f"F{rank} (n={cnt})", density=True)
        ax.set_xlabel(f"{bname} distance (A)")
        ax.set_ylabel("Density")
        ax.set_title(bname, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Bond Distance Distributions by Family", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_bond_fingerprints.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 02_bond_fingerprints.png")

    # Angle distributions
    fig, axes = plt.subplots(len(angle_names), 1, figsize=(14, 3 * len(angle_names)))
    if len(angle_names) == 1:
        axes = [axes]

    for a_idx, aname in enumerate(angle_names):
        ax = axes[a_idx]
        for f_idx, (fam_id, cnt) in enumerate(big_fams):
            mask = ts_family == fam_id
            vals = [all_angles[i][aname] for i in range(n) if mask[i]]
            rank = fam_rank[fam_id]
            ax.hist(vals, bins=20, alpha=0.5, color=COLORS[rank % 20],
                   label=f"F{rank} (n={cnt})", density=True)
        ax.set_xlabel(f"{aname} angle (deg)")
        ax.set_ylabel("Density")
        ax.set_title(aname, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Bond Angle Distributions by Family", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_angle_fingerprints.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 03_angle_fingerprints.png")

    # Summary table: mean bond/angle per family
    print(f"\n{'='*100}")
    print("Structural Fingerprints by Family")
    print(f"{'='*100}")
    header = f"{'Family':>8} {'Size':>5} {'Energy':>10}"
    for bname in bond_names:
        header += f" {bname:>10}"
    for aname in angle_names:
        header += f" {aname:>12}"
    print(header)
    print("-" * len(header))

    for fam_id, cnt in fam_counts.most_common():
        if cnt < 2:
            continue
        rank = fam_rank[fam_id]
        mask = ts_family == fam_id
        row = f"F{rank:>6} {cnt:5d} {energies[mask].mean():10.3f}"
        for bname in bond_names:
            vals = [all_bonds[i][bname] for i in range(n) if mask[i]]
            row += f" {np.mean(vals):10.3f}"
        for aname in angle_names:
            vals = [all_angles[i][aname] for i in range(n) if mask[i]]
            row += f" {np.mean(vals):12.1f}"
        print(row)


def plot_family_detail_tsne(ts_family, labels_L1, ts_list, D, tsne_xy):
    """Per-family detail with seed labels."""
    n = len(ts_list)
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    seeds = [r["seed"] for r in ts_list]

    fam_counts = Counter(ts_family)
    big_fams = [(f, c) for f, c in fam_counts.most_common() if c >= 3]
    n_big = len(big_fams)
    if n_big == 0:
        return

    fam_rank = {f: i for i, (f, _) in enumerate(fam_counts.most_common())}

    ncols = min(3, n_big)
    nrows = (n_big + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (fam_id, cnt) in enumerate(big_fams):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        rank = fam_rank[fam_id]

        mask = ts_family == fam_id
        fam_idx = np.where(mask)[0]

        # Background
        ax.scatter(tsne_xy[~mask, 0], tsne_xy[~mask, 1], c="lightgray", s=8, alpha=0.2)

        # Family members colored by L1 core
        l1_labels = labels_L1[fam_idx]
        l1_counts = Counter(l1_labels)
        for ci, (core, core_cnt) in enumerate(l1_counts.most_common()):
            core_mask = mask & (labels_L1 == core)
            c = COLORS[ci % 20]
            ax.scatter(tsne_xy[core_mask, 0], tsne_xy[core_mask, 1], c=[c], s=100,
                      edgecolors="k", linewidths=1, alpha=0.9,
                      label=f"Core {core} (n={core_cnt})")

        # Seed labels
        for i in fam_idx:
            ax.annotate(f"s{seeds[i]}", xy=(tsne_xy[i, 0], tsne_xy[i, 1]),
                       fontsize=6, ha="center", va="top",
                       xytext=(0, -8), textcoords="offset points")

        fam_e = energies[fam_idx]
        fam_eig = eig0s[fam_idx]
        ax.set_title(f"F{rank} (n={cnt}): E={fam_e.mean():.2f}±{fam_e.std():.3f}, "
                    f"eig0={fam_eig.mean():.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, loc="best")

    # Hide empty
    for idx in range(n_big, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle("Family Detail (t-SNE, colored by L1 core, labeled by seed)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_family_detail.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 04_family_detail.png")


def plot_combined_dendrogram(Z, ts_family, ts_list, D):
    """Full dendrogram with colored leaves by family."""
    n = len(ts_list)
    energies = np.array([r["final_energy"] for r in ts_list])
    seeds = [r["seed"] for r in ts_list]
    fam_counts = Counter(ts_family)
    fam_rank = {f: i for i, (f, _) in enumerate(fam_counts.most_common())}

    fig, ax = plt.subplots(1, 1, figsize=(24, 8))

    # Custom leaf labels
    leaf_labels = [f"s{seeds[i]}" for i in range(n)]

    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=5,
               labels=leaf_labels, color_threshold=0.4)

    ax.axhline(y=0.15, color="blue", ls="--", lw=1, alpha=0.5, label="L1=0.15 A")
    ax.axhline(y=0.40, color="red", ls="--", lw=1.5, label="L2=0.40 A")
    ax.set_ylabel("RMSD (A)")
    ax.set_title("Full Dendrogram — Average Linkage (87 TS)", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_full_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 05_full_dendrogram.png")


def print_summary(ts_family, soft_assignment, is_soft, labels_L1, ts_list, D):
    """Full summary with family details."""
    n = len(ts_list)
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    seeds = [r["seed"] for r in ts_list]

    fam_counts = Counter(ts_family)
    fam_rank = {f: i for i, (f, _) in enumerate(fam_counts.most_common())}

    print(f"\n{'='*80}")
    print(f"FAMILY SUMMARY — Two-level (L1=0.15 A, L2=0.4 A)")
    print(f"{'='*80}")
    print(f"Total TS: {n}")
    print(f"Families: {len(fam_counts)} ({sum(1 for c in fam_counts.values() if c >= 2)} multi-member)")
    print(f"TS in multi-member families: {sum(c for c in fam_counts.values() if c >= 2)}")
    print(f"Singletons: {sum(1 for c in fam_counts.values() if c == 1)}")
    print(f"Singletons reassigned: {is_soft.sum()}")

    print(f"\n{'Rank':>4} {'Size':>5} {'Cores':>6} {'E_mean':>10} {'E_std':>8} "
          f"{'eig0_mean':>10} {'RMSD_max':>9} {'Seeds'}")
    print("-" * 110)

    results = []
    for fam_id, cnt in fam_counts.most_common():
        rank = fam_rank[fam_id]
        mask = ts_family == fam_id
        fam_idx = np.where(mask)[0]
        fam_e = energies[fam_idx]
        fam_eig = eig0s[fam_idx]
        fam_seeds = [seeds[i] for i in fam_idx]
        n_cores = len(set(labels_L1[fam_idx]))

        if len(fam_idx) > 1:
            intra = [D[a, b] for a in fam_idx for b in fam_idx if a < b]
            max_intra = max(intra)
            mean_intra = np.mean(intra)
        else:
            max_intra = mean_intra = 0

        info = {
            "rank": rank, "family_id": int(fam_id), "size": cnt,
            "n_cores": n_cores,
            "energy_mean": float(fam_e.mean()),
            "energy_std": float(fam_e.std()) if cnt > 1 else 0,
            "eig0_mean": float(fam_eig.mean()),
            "rmsd_intra_max": float(max_intra),
            "rmsd_intra_mean": float(mean_intra),
            "seeds": fam_seeds,
        }
        results.append(info)

        if cnt >= 2 or rank < 12:
            seeds_str = str(fam_seeds[:8]) + ("..." if len(fam_seeds) > 8 else "")
            print(f"F{rank:3d} {cnt:5d} {n_cores:6d} {fam_e.mean():10.4f} "
                  f"{info['energy_std']:8.4f} {fam_eig.mean():10.4f} "
                  f"{max_intra:9.4f} {seeds_str}")

    # Save
    output = {
        "method": "two_level_hierarchical",
        "L1_threshold": 0.15, "L2_threshold": 0.4,
        "n_ts": n, "n_families": len(fam_counts),
        "families": results,
        "ts_labels": ts_family.tolist(),
        "soft_labels": soft_assignment.tolist(),
        "is_soft": is_soft.tolist(),
        "seeds": seeds,
        "energies": energies.tolist(),
    }
    out_path = os.path.join(OUT_DIR, "families_v3.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")

    return results


def main():
    print("=" * 70)
    print("Structural Clustering v3: Refined Two-Level + Characterization")
    print("=" * 70)

    ts_list, D = load_data()

    labels_L1, ts_family, Z, Z_core, core_ids, core_D = \
        two_level_clustering(D, ts_list, L1_thresh=0.15, L2_thresh=0.4)

    soft_assignment, assignment_dist, is_soft = assign_singletons(ts_family, D)

    tsne_xy, mds_xy = plot_main_overview(ts_family, soft_assignment, is_soft, ts_list, D)
    plot_structural_fingerprints(ts_family, ts_list, D)
    plot_family_detail_tsne(ts_family, labels_L1, ts_list, D, tsne_xy)
    plot_combined_dendrogram(Z, ts_family, ts_list, D)

    results = print_summary(ts_family, soft_assignment, is_soft, labels_L1, ts_list, D)

    print(f"\n{'='*70}")
    print(f"All outputs in {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
