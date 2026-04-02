#!/usr/bin/env python
"""Structural clustering iteration 2: two-level + HDBSCAN + family analysis.

Builds on v1 findings:
- ~40 TS are in tight groups (NN RMSD < 0.1 A), ~47 are isolated
- DBSCAN eps=0.1 finds ~10 core groups, labels rest as noise
- Need a strategy that handles both tight cores AND singletons

Approach:
1. HDBSCAN to find variable-density clusters
2. Two-level: tight cores at 0.15A, then merge singletons at 0.5A
3. Analyze each TS family: geometry, bonding, what makes them different
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
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

OUT_DIR = "/scratch/memoozd/diffusion-ts/structural_clustering"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
NOISE_COLOR = [0.7, 0.7, 0.7, 1.0]

ATOMIC_SYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"]


def load_data():
    """Load RMSD matrix and TS data."""
    D = np.load(os.path.join(OUT_DIR, "rmsd_matrix_87.npy"))
    np.fill_diagonal(D, 0)  # ensure exact zeros on diagonal
    with open("/scratch/memoozd/diffusion-ts/adaptive_results.json") as f:
        data = json.load(f)
    ts_list = [r for r in data if r["status"] == "ts" and "ts_coords" in r]
    return D, ts_list


def hdbscan_analysis(D, ts_list):
    """Try HDBSCAN with various min_cluster_size."""
    energies = np.array([r["final_energy"] for r in ts_list])
    n = len(ts_list)

    print("=" * 70)
    print("HDBSCAN Analysis")
    print("=" * 70)

    results = []
    for mcs in [2, 3, 4, 5, 7, 10]:
        for ms in [1, 2, 3]:
            if ms > mcs:
                continue
            hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="precomputed")
            labels = hdb.fit_predict(D)
            nc = len(set(labels)) - (1 if -1 in labels else 0)
            noise = np.sum(labels == -1)
            if nc >= 2:
                valid = labels != -1
                D_valid = D[np.ix_(valid, valid)].copy()
                np.fill_diagonal(D_valid, 0)
                sil = silhouette_score(D_valid, labels[valid],
                                      metric="precomputed") if valid.sum() > nc else 0
            else:
                sil = 0
            results.append({
                "mcs": mcs, "ms": ms, "nc": nc, "noise": noise,
                "sil": sil, "labels": labels.copy(),
            })
            print(f"  mcs={mcs}, ms={ms}: {nc} clusters, {noise} noise, sil={sil:.3f}")

    return results


def two_level_clustering(D, ts_list):
    """Two-level: tight cores at 0.15A, merge singletons at coarser scale."""
    energies = np.array([r["final_energy"] for r in ts_list])
    n = len(ts_list)
    D_condensed = squareform(D, checks=False)

    # Level 1: tight cores
    Z = linkage(D_condensed, method="average")
    labels_fine = fcluster(Z, t=0.15, criterion="distance")

    counts_fine = Counter(labels_fine)
    core_clusters = {cl for cl, cnt in counts_fine.items() if cnt >= 2}
    singletons = {cl for cl, cnt in counts_fine.items() if cnt == 1}

    print(f"\n{'='*70}")
    print(f"Two-level clustering")
    print(f"{'='*70}")
    print(f"Level 1 (t=0.15): {len(counts_fine)} clusters, "
          f"{len(core_clusters)} cores (size>=2), {len(singletons)} singletons")

    # Level 2: assign singletons to nearest core, or merge nearby singletons
    labels_2level = np.zeros(n, dtype=int)
    next_label = 1

    # Map core clusters
    core_map = {}
    for cl in sorted(core_clusters):
        core_map[cl] = next_label
        mask = labels_fine == cl
        labels_2level[mask] = next_label
        next_label += 1

    # For each singleton, find nearest core
    singleton_indices = []
    for i in range(n):
        if labels_fine[i] in singletons:
            singleton_indices.append(i)

    # Try to assign singletons to nearest core within 0.5A
    assigned = 0
    unassigned_idx = []
    for idx in singleton_indices:
        best_dist = float("inf")
        best_label = -1
        for cl in core_clusters:
            core_members = np.where(labels_fine == cl)[0]
            mean_dist = D[idx, core_members].mean()
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_label = core_map[cl]
        if best_dist < 0.5:
            labels_2level[idx] = best_label
            assigned += 1
        else:
            unassigned_idx.append(idx)

    # Remaining singletons: try to group among themselves
    if unassigned_idx:
        sub_D = D[np.ix_(unassigned_idx, unassigned_idx)]
        if len(unassigned_idx) > 1:
            sub_condensed = squareform(sub_D, checks=False)
            Z_sub = linkage(sub_condensed, method="average")
            sub_labels = fcluster(Z_sub, t=0.5, criterion="distance")
            for i, idx in enumerate(unassigned_idx):
                labels_2level[idx] = next_label + sub_labels[i] - 1
            next_label += len(set(sub_labels))
        else:
            labels_2level[unassigned_idx[0]] = next_label
            next_label += 1

    counts_2level = Counter(labels_2level)
    print(f"Level 2: assigned {assigned} singletons to cores, "
          f"{len(unassigned_idx)} remain isolated")
    print(f"Final: {len(counts_2level)} clusters")
    print(f"Sizes: {sorted(counts_2level.values(), reverse=True)}")

    return labels_2level, Z


def find_best_grouping(D, ts_list):
    """Try multiple strategies and pick the one with best balance of
    cluster cohesion vs coverage (not too many singletons)."""
    energies = np.array([r["final_energy"] for r in ts_list])
    n = len(ts_list)
    D_condensed = squareform(D, checks=False)

    print(f"\n{'='*70}")
    print("Strategy comparison")
    print(f"{'='*70}")

    strategies = {}

    # 1. Hierarchical at various thresholds
    Z = linkage(D_condensed, method="average")
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        labels = fcluster(Z, t=t, criterion="distance")
        strategies[f"hier_avg_t{t}"] = labels

    # 2. Complete linkage
    Z_comp = linkage(D_condensed, method="complete")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        labels = fcluster(Z_comp, t=t, criterion="distance")
        strategies[f"hier_comp_t{t}"] = labels

    # 3. DBSCAN
    for eps in [0.1, 0.15, 0.2, 0.25, 0.3]:
        db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
        labels = db.fit_predict(D)
        # Convert noise to singleton clusters
        noise_idx = np.where(labels == -1)[0]
        max_label = labels.max() + 1
        for idx in noise_idx:
            labels[idx] = max_label
            max_label += 1
        strategies[f"dbscan_eps{eps}"] = labels

    # 4. HDBSCAN (best from analysis)
    for mcs in [3, 4, 5]:
        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=2, metric="precomputed")
        labels = hdb.fit_predict(D)
        noise_idx = np.where(labels == -1)[0]
        max_label = labels.max() + 1
        for idx in noise_idx:
            labels[idx] = max_label
            max_label += 1
        strategies[f"hdbscan_mcs{mcs}_ms2"] = labels

    # Evaluate
    print(f"\n{'Method':<25} {'NCl':>4} {'Singletons':>11} {'MaxCl':>6} "
          f"{'Coverage':>9} {'MeanIntra':>10} {'MeanInter':>10}")
    best_score = -1
    best_method = None

    for name, labels in sorted(strategies.items()):
        counts = Counter(labels)
        nc = len(counts)
        n_sing = sum(1 for v in counts.values() if v == 1)
        max_cl = max(counts.values())
        coverage = 1 - n_sing / n  # fraction in multi-member clusters

        # Mean intra-cluster RMSD
        intra_rmsds = []
        for cl in counts:
            idx = np.where(labels == cl)[0]
            if len(idx) >= 2:
                for ii in range(len(idx)):
                    for jj in range(ii+1, len(idx)):
                        intra_rmsds.append(D[idx[ii], idx[jj]])
        mean_intra = np.mean(intra_rmsds) if intra_rmsds else 0

        # Mean inter-cluster RMSD (between cluster centroids)
        cluster_ids = [cl for cl in counts if counts[cl] >= 2]
        inter_rmsds = []
        for i, ci in enumerate(cluster_ids):
            idx_i = np.where(labels == ci)[0]
            for j in range(i+1, len(cluster_ids)):
                cj = cluster_ids[j]
                idx_j = np.where(labels == cj)[0]
                inter_rmsds.append(D[np.ix_(idx_i, idx_j)].mean())
        mean_inter = np.mean(inter_rmsds) if inter_rmsds else 0

        # Score: balance coverage, low intra, high inter
        separation = mean_inter / (mean_intra + 0.01) if mean_intra > 0 else 0
        score = coverage * separation

        print(f"{name:<25} {nc:4d} {n_sing:11d} {max_cl:6d} "
              f"{coverage:9.1%} {mean_intra:10.4f} {mean_inter:10.4f}")

        if score > best_score:
            best_score = score
            best_method = name

    print(f"\nBest method (coverage * separation): {best_method}")
    return strategies, best_method


def plot_best_methods(D, ts_list, strategies, top_methods):
    """Visualize the top clustering methods on t-SNE."""
    energies = np.array([r["final_energy"] for r in ts_list])
    n = len(ts_list)

    # t-SNE at perp=15
    tsne = TSNE(n_components=2, metric="precomputed", perplexity=15,
                random_state=42, init="random", max_iter=3000)
    tsne_xy = tsne.fit_transform(D)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", max_iter=1000)
    mds_xy = mds.fit_transform(D)

    n_methods = len(top_methods)
    fig, axes = plt.subplots(n_methods, 3, figsize=(18, 5 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for row, method in enumerate(top_methods):
        labels = strategies[method]
        counts = Counter(labels)

        # Rank by size
        rank = {cl: i for i, (cl, _) in enumerate(counts.most_common())}
        labels_ranked = np.array([rank[l] for l in labels])
        nc = len(counts)

        # t-SNE
        ax = axes[row, 0]
        for cl in range(min(nc, 20)):
            mask = labels_ranked == cl
            sz = counts[list(counts.keys())[list(rank.values()).index(cl)]]
            if sz == 1:
                ax.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=[NOISE_COLOR],
                          s=15, alpha=0.5, marker="x")
            else:
                c = COLORS[cl % 20]
                ax.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=[c], s=40,
                          edgecolors="k", linewidths=0.3, alpha=0.8,
                          label=f"C{cl}({sz})" if cl < 10 else None)
        ax.set_title(f"{method} — t-SNE ({nc} cl)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if nc <= 15:
            ax.legend(fontsize=5, loc="best", ncol=2)

        # MDS
        ax = axes[row, 1]
        for cl in range(min(nc, 20)):
            mask = labels_ranked == cl
            sz = counts[list(counts.keys())[list(rank.values()).index(cl)]]
            if sz == 1:
                ax.scatter(mds_xy[mask, 0], mds_xy[mask, 1], c=[NOISE_COLOR],
                          s=15, alpha=0.5, marker="x")
            else:
                c = COLORS[cl % 20]
                ax.scatter(mds_xy[mask, 0], mds_xy[mask, 1], c=[c], s=40,
                          edgecolors="k", linewidths=0.3, alpha=0.8)
        ax.set_title(f"{method} — MDS", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Energy by cluster (only multi-member clusters)
        ax = axes[row, 2]
        multi_clusters = [(cl, sz) for cl, sz in counts.most_common() if sz >= 2]
        if multi_clusters:
            data_list = []
            tick_labels = []
            for cl, sz in multi_clusters[:15]:
                mask = labels == cl
                data_list.append(energies[mask])
                tick_labels.append(f"C{rank[cl]}({sz})")
            bp = ax.boxplot(data_list, widths=0.6, patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(COLORS[i % 20])
                patch.set_alpha(0.6)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=6)
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"{method} — Energy by cluster", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Top Clustering Methods Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/11_top_methods.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 11_top_methods.png")

    return tsne_xy, mds_xy


def analyze_ts_families(D, ts_list, labels, tsne_xy, mds_xy, method_name):
    """Deep analysis of each TS family."""
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    n = len(ts_list)
    counts = Counter(labels)

    print(f"\n{'='*70}")
    print(f"TS Family Analysis — {method_name}")
    print(f"{'='*70}")

    # Sort clusters by size
    sorted_clusters = counts.most_common()
    multi_member = [(cl, sz) for cl, sz in sorted_clusters if sz >= 2]

    print(f"\nMulti-member families ({len(multi_member)}):")
    print(f"{'Fam':>4} {'Size':>5} {'E_mean':>10} {'E_std':>8} {'eig0_mean':>10} "
          f"{'RMSD_intra':>11} {'RMSD_nn':>8}")

    families = []
    for cl, sz in multi_member:
        idx = np.where(labels == cl)[0]
        cl_e = energies[idx]
        cl_eig = eig0s[idx]

        # Intra RMSD
        intra = []
        for ii in range(len(idx)):
            for jj in range(ii+1, len(idx)):
                intra.append(D[idx[ii], idx[jj]])

        # Nearest-neighbor outside cluster
        nn_out = []
        for i in idx:
            ext_dists = [D[i, j] for j in range(n) if labels[j] != cl]
            nn_out.append(min(ext_dists) if ext_dists else 0)

        # Medoid
        sub_D = D[np.ix_(idx, idx)]
        medoid_local = sub_D.sum(axis=1).argmin()
        medoid_global = idx[medoid_local]

        family = {
            "cluster_id": int(cl),
            "size": int(sz),
            "member_indices": idx.tolist(),
            "member_seeds": [ts_list[i]["seed"] for i in idx],
            "energy_mean": float(cl_e.mean()),
            "energy_std": float(cl_e.std()),
            "energy_range": float(cl_e.max() - cl_e.min()),
            "eig0_mean": float(cl_eig.mean()),
            "eig0_std": float(cl_eig.std()),
            "rmsd_intra_mean": float(np.mean(intra)),
            "rmsd_intra_max": float(np.max(intra)) if intra else 0,
            "rmsd_nn_external": float(np.mean(nn_out)),
            "medoid_seed": ts_list[medoid_global]["seed"],
            "medoid_coords": ts_list[medoid_global]["ts_coords"],
        }
        families.append(family)

        print(f"{cl:4d} {sz:5d} {cl_e.mean():10.4f} {cl_e.std():8.4f} {cl_eig.mean():10.4f} "
              f"{np.mean(intra):11.4f} {np.mean(nn_out):8.4f}")

    # Singletons
    singletons = [(cl, 1) for cl, sz in sorted_clusters if sz == 1]
    print(f"\nSingletons: {len(singletons)}")
    for cl, _ in singletons[:10]:
        idx = np.where(labels == cl)[0][0]
        nn_dist = sorted(D[idx, :])[1]
        print(f"  seed={ts_list[idx]['seed']:3d}  E={energies[idx]:.4f}  "
              f"eig0={eig0s[idx]:.4f}  NN_RMSD={nn_dist:.4f}")
    if len(singletons) > 10:
        print(f"  ... and {len(singletons)-10} more")

    # Save results
    results = {
        "method": method_name,
        "n_ts": n,
        "n_families": len(multi_member),
        "n_singletons": len(singletons),
        "families": families,
        "all_labels": labels.tolist(),
        "seeds": [r["seed"] for r in ts_list],
    }
    out_path = os.path.join(OUT_DIR, "ts_families.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    return families


def plot_families_detail(D, ts_list, labels, families, tsne_xy, mds_xy):
    """Detailed visualization of each TS family."""
    energies = np.array([r["final_energy"] for r in ts_list])
    eig0s = np.array([r.get("ts_eig0", 0) for r in ts_list])
    n = len(ts_list)
    counts = Counter(labels)

    # Sort by family size
    families_sorted = sorted(families, key=lambda f: -f["size"])

    # Plot 1: t-SNE with family labels and annotations
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax_idx, (coords, name) in enumerate([(tsne_xy, "t-SNE (p=15)"), (mds_xy, "MDS")]):
        ax = axes[ax_idx]
        # Background: singletons in gray
        for i in range(n):
            if counts[labels[i]] == 1:
                ax.scatter(coords[i, 0], coords[i, 1], c=[NOISE_COLOR],
                          s=15, alpha=0.4, marker="x", zorder=1)

        # Families colored
        for fi, fam in enumerate(families_sorted[:15]):
            idx = np.array(fam["member_indices"])
            c = COLORS[fi % 20]
            ax.scatter(coords[idx, 0], coords[idx, 1], c=[c], s=50,
                      edgecolors="k", linewidths=0.5, alpha=0.85, zorder=3,
                      label=f"F{fi+1} (n={fam['size']}, E={fam['energy_mean']:.1f})")

            # Annotate medoid
            sub_D = D[np.ix_(idx, idx)]
            medoid_local = sub_D.sum(axis=1).argmin()
            mi = idx[medoid_local]
            ax.annotate(f"F{fi+1}", xy=(coords[mi, 0], coords[mi, 1]),
                       fontsize=7, fontweight="bold", ha="center", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec=c),
                       zorder=5)

        ax.set_title(f"{name}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.9)

    plt.suptitle(f"TS Families ({len(families_sorted)} families, "
                f"{sum(f['size'] for f in families_sorted)} of {n} TS)",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/12_families_labeled.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 12_families_labeled.png")

    # Plot 2: Family properties summary
    n_fam = min(len(families_sorted), 15)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Energy by family
    ax = axes[0, 0]
    data = [energies[np.array(f["member_indices"])] for f in families_sorted[:n_fam]]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % 20])
        patch.set_alpha(0.6)
    ax.set_xticklabels([f"F{i+1}\n({f['size']})" for i, f in enumerate(families_sorted[:n_fam])],
                       fontsize=7)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Energy by family")
    ax.grid(True, alpha=0.3, axis="y")

    # eig0 by family
    ax = axes[0, 1]
    data = [eig0s[np.array(f["member_indices"])] for f in families_sorted[:n_fam]]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % 20])
        patch.set_alpha(0.6)
    ax.set_xticklabels([f"F{i+1}" for i in range(n_fam)], fontsize=7)
    ax.set_ylabel("eig0 (TS eigenvalue)")
    ax.set_title("eig0 by family")
    ax.grid(True, alpha=0.3, axis="y")

    # Intra-cluster RMSD vs inter-cluster distance
    ax = axes[1, 0]
    for fi, fam in enumerate(families_sorted[:n_fam]):
        ax.bar(fi, fam["rmsd_intra_mean"], width=0.4, color=COLORS[fi % 20],
              edgecolor="k", alpha=0.7, label=f"F{fi+1}")
        ax.bar(fi + 0.4, fam["rmsd_nn_external"], width=0.4, color=COLORS[fi % 20],
              edgecolor="k", alpha=0.3)
    ax.set_xticks(range(n_fam))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_fam)], fontsize=7)
    ax.set_ylabel("RMSD (A)")
    ax.set_title("Intra-cluster (solid) vs nearest external (light)")
    ax.grid(True, alpha=0.3, axis="y")

    # Family size distribution
    ax = axes[1, 1]
    sizes = sorted([f["size"] for f in families_sorted], reverse=True)
    ax.bar(range(len(sizes)), sizes, color="steelblue", edgecolor="k", alpha=0.7)
    ax.set_xlabel("Family rank")
    ax.set_ylabel("Number of TS")
    ax.set_title(f"Family size distribution ({len(families_sorted)} families)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/13_family_properties.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 13_family_properties.png")


def plot_distance_heatmap_ordered(D, ts_list, labels):
    """RMSD heatmap with rows/cols ordered by cluster."""
    n = len(ts_list)
    counts = Counter(labels)
    sorted_clusters = counts.most_common()

    # Build ordering: by cluster (large first), within cluster by energy
    energies = np.array([r["final_energy"] for r in ts_list])
    order = []
    cluster_boundaries = []
    for cl, sz in sorted_clusters:
        idx = np.where(labels == cl)[0]
        idx_sorted = idx[np.argsort(energies[idx])]
        cluster_boundaries.append(len(order))
        order.extend(idx_sorted)

    order = np.array(order)
    D_ordered = D[np.ix_(order, order)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(D_ordered, cmap="YlOrRd_r", aspect="auto", vmin=0, vmax=1.5)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Aligned RMSD (A)")

    # Draw cluster boundaries
    for b in cluster_boundaries[1:]:
        ax.axhline(y=b - 0.5, color="blue", lw=0.5, alpha=0.5)
        ax.axvline(x=b - 0.5, color="blue", lw=0.5, alpha=0.5)

    # Label major clusters
    for i, (cl, sz) in enumerate(sorted_clusters[:10]):
        if sz >= 2:
            start = cluster_boundaries[i]
            mid = start + sz / 2
            ax.text(-2, mid, f"F{i+1}({sz})", fontsize=6, va="center", ha="right",
                   color=COLORS[i % 20], fontweight="bold")

    ax.set_title(f"RMSD Distance Matrix (ordered by cluster)", fontsize=12)
    ax.set_xlabel("TS index (cluster-ordered)")
    ax.set_ylabel("TS index (cluster-ordered)")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/14_rmsd_heatmap_ordered.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 14_rmsd_heatmap_ordered.png")


def analyze_bonding_patterns(ts_list, families):
    """Analyze what makes each family structurally distinct.
    Look at interatomic distances, especially bonds that break/form."""
    print(f"\n{'='*70}")
    print("Bonding pattern analysis")
    print(f"{'='*70}")

    # Key atom pairs to watch in isopropanol
    bond_pairs = [
        (0, 1, "C0-C1"), (0, 2, "C0-C2"), (0, 3, "C0-O"), (3, 5, "O-H5"),
        (0, 4, "C0-H4"), (1, 6, "C1-H6"), (1, 7, "C1-H7"), (1, 8, "C1-H8"),
        (2, 9, "C2-H9"), (2, 10, "C2-H10"), (2, 11, "C2-H11"),
        (1, 3, "C1-O"), (2, 3, "C2-O"), (0, 5, "C0-H5"),
        (1, 2, "C1-C2"),
    ]

    equil_coords = np.array([
        [0.000, 0.000, 0.000], [-1.270, 0.760, 0.000], [1.270, 0.760, 0.000],
        [0.000, -0.930, 1.100], [0.000, -0.670, -0.870], [0.000, -0.400, 1.920],
        [-1.270, 1.410, 0.870], [-1.270, 1.410, -0.870], [-2.160, 0.120, 0.000],
        [1.270, 1.410, 0.870], [1.270, 1.410, -0.870], [2.160, 0.120, 0.000],
    ])

    for fi, fam in enumerate(families[:10]):
        medoid_coords = np.array(fam["medoid_coords"])

        print(f"\nFamily {fi+1} (n={fam['size']}, E={fam['energy_mean']:.3f}, "
              f"eig0={fam['eig0_mean']:.3f}):")

        changed = []
        for a, b, name in bond_pairs:
            d_eq = np.linalg.norm(equil_coords[a] - equil_coords[b])
            d_ts = np.linalg.norm(medoid_coords[a] - medoid_coords[b])
            change = d_ts - d_eq
            if abs(change) > 0.2:
                changed.append((name, d_eq, d_ts, change))

        if changed:
            for name, d_eq, d_ts, change in sorted(changed, key=lambda x: -abs(x[3])):
                direction = "stretched" if change > 0 else "compressed"
                print(f"  {name}: {d_eq:.3f} -> {d_ts:.3f} ({change:+.3f}, {direction})")
        else:
            print("  No bonds changed > 0.2 A from equilibrium")


def main():
    D, ts_list = load_data()
    n = len(ts_list)
    print(f"Loaded {n} TS, RMSD matrix {D.shape}")

    # 1. HDBSCAN
    hdb_results = hdbscan_analysis(D, ts_list)

    # 2. Two-level clustering
    labels_2level, Z = two_level_clustering(D, ts_list)

    # 3. Strategy comparison
    strategies, best_method = find_best_grouping(D, ts_list)

    # Add two-level to strategies
    strategies["two_level"] = labels_2level

    # Pick top methods to visualize
    top_methods = [best_method, "hier_avg_t0.4", "hier_avg_t0.5", "two_level"]
    # Deduplicate
    seen = set()
    top_unique = []
    for m in top_methods:
        if m not in seen and m in strategies:
            top_unique.append(m)
            seen.add(m)

    tsne_xy, mds_xy = plot_best_methods(D, ts_list, strategies, top_unique)

    # 4. Deep analysis of best method
    primary_method = "hier_avg_t0.5"
    primary_labels = strategies[primary_method]
    families = analyze_ts_families(D, ts_list, primary_labels, tsne_xy, mds_xy, primary_method)

    # 5. Detailed visualizations
    plot_families_detail(D, ts_list, primary_labels, families, tsne_xy, mds_xy)
    plot_distance_heatmap_ordered(D, ts_list, primary_labels)
    analyze_bonding_patterns(ts_list, families)

    print(f"\n{'='*70}")
    print(f"All outputs in {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
