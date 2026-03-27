"""Cluster collected TS geometries using Kabsch alignment + Hungarian matching.

Loads TS results from parallel_results.json, computes pairwise aligned RMSD,
and performs hierarchical clustering to identify distinct transition states.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.dependencies.alignment import pairwise_rmsd_matrix

# Isopropanol equivalence classes
EQUIV_CLASSES = {
    "central_C": [0],
    "methyl_C": [1, 2],
    "O": [3],
    "H_central": [4],
    "H_hydroxyl": [5],
    "H_methyl": [6, 7, 8, 9, 10, 11],
}

# Methyl carbon swap info: carbons [1,2] with H groups [6,7,8] and [9,10,11]
METHYL_CARBONS = [1, 2]
METHYL_HYDROGENS = [[6, 7, 8], [9, 10, 11]]

INPUT_PATH = "/scratch/memoozd/diffusion-ts/parallel_results.json"
OUTPUT_PATH = "/scratch/memoozd/diffusion-ts/clustered_results.json"
RMSD_THRESHOLD = 0.3  # Angstrom, for cluster cutoff


@dataclass
class ClusterInfo:
    """Precomputed per-cluster data used for both printing and serialization."""

    cluster_id: int
    indices: np.ndarray
    energies: list[float]
    eig0s: list[float]
    seeds: list[int]
    representative_idx: int
    mean_intra_rmsd: float
    max_intra_rmsd: float


def load_ts_records(path: str) -> list[dict]:
    """Load and filter TS records from parallel results."""
    with open(path) as f:
        data = json.load(f)
    ts_records = [r for r in data if r.get("status") == "ts" and "ts_coords" in r]
    print(f"Loaded {len(data)} records, {len(ts_records)} are converged TS")
    return ts_records


def cluster_geometries(
    geometries: list[np.ndarray], threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise RMSD and cluster with complete linkage.

    Returns:
        labels: (N,) cluster labels (1-indexed).
        D: (N, N) distance matrix.
    """
    print(f"Computing pairwise RMSD matrix for {len(geometries)} geometries...")
    D = pairwise_rmsd_matrix(
        geometries, EQUIV_CLASSES, METHYL_CARBONS, METHYL_HYDROGENS
    )

    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="complete")
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels, D


def build_cluster_infos(
    ts_records: list[dict], labels: np.ndarray, D: np.ndarray
) -> list[ClusterInfo]:
    """Extract per-cluster statistics once for reuse."""
    infos = []
    for cl in sorted(set(labels)):
        indices = np.where(labels == cl)[0]
        energies = [ts_records[i]["final_energy"] for i in indices]
        eig0s = [ts_records[i]["ts_eig0"] for i in indices]
        seeds = [ts_records[i]["seed"] for i in indices]
        representative_idx = indices[int(np.argmin(energies))]

        if len(indices) > 1:
            sub_D = D[np.ix_(indices, indices)]
            max_intra = float(sub_D.max())
            mean_intra = float(sub_D[np.triu_indices(len(indices), k=1)].mean())
        else:
            max_intra = 0.0
            mean_intra = 0.0

        infos.append(
            ClusterInfo(
                cluster_id=int(cl),
                indices=indices,
                energies=energies,
                eig0s=eig0s,
                seeds=seeds,
                representative_idx=representative_idx,
                mean_intra_rmsd=mean_intra,
                max_intra_rmsd=max_intra,
            )
        )
    return infos


def print_summary(ts_records: list[dict], cluster_infos: list[ClusterInfo]) -> None:
    """Print clustering summary with energy and structural info."""
    print(f"\n{'='*60}")
    print(
        f"Clustering results: {len(cluster_infos)} clusters "
        f"(threshold={RMSD_THRESHOLD} A)"
    )
    print(f"{'='*60}")

    for ci in cluster_infos:
        rep = ts_records[ci.representative_idx]
        print(f"\nCluster {ci.cluster_id}: {len(ci.indices)} members")
        print(f"  Seeds: {ci.seeds}")
        print(
            f"  Energy: {min(ci.energies):.4f} to {max(ci.energies):.4f} eV "
            f"(mean {np.mean(ci.energies):.4f})"
        )
        print(
            f"  eig0:   {min(ci.eig0s):.4f} to {max(ci.eig0s):.4f} "
            f"(mean {np.mean(ci.eig0s):.4f})"
        )
        print(
            f"  Intra-cluster RMSD: mean={ci.mean_intra_rmsd:.4f}, "
            f"max={ci.max_intra_rmsd:.4f} A"
        )
        print(
            f"  Representative: seed {rep['seed']} "
            f"(E={rep['final_energy']:.4f})"
        )


def save_results(
    ts_records: list[dict],
    cluster_infos: list[ClusterInfo],
    D: np.ndarray,
    output_path: str,
) -> None:
    """Save clustering results to JSON."""
    clusters = []
    for ci in cluster_infos:
        rep = ts_records[ci.representative_idx]
        members = [
            {
                "seed": ts_records[i]["seed"],
                "energy": ts_records[i]["final_energy"],
                "eig0": ts_records[i]["ts_eig0"],
                "ts_coords": ts_records[i]["ts_coords"],
            }
            for i in ci.indices
        ]
        clusters.append({
            "cluster_id": ci.cluster_id,
            "size": len(ci.indices),
            "energy_mean": float(np.mean(ci.energies)),
            "energy_std": float(np.std(ci.energies)),
            "representative_seed": int(rep["seed"]),
            "representative_coords": rep["ts_coords"],
            "members": members,
        })

    result = {
        "n_clusters": len(cluster_infos),
        "n_ts": len(ts_records),
        "rmsd_threshold": RMSD_THRESHOLD,
        "clusters": clusters,
        "rmsd_matrix": D.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main() -> None:
    ts_records = load_ts_records(INPUT_PATH)
    if not ts_records:
        print("No TS records found.")
        sys.exit(1)

    geometries = [np.array(r["ts_coords"]) for r in ts_records]
    labels, D = cluster_geometries(geometries, RMSD_THRESHOLD)
    cluster_infos = build_cluster_infos(ts_records, labels, D)
    print_summary(ts_records, cluster_infos)
    save_results(ts_records, cluster_infos, D, OUTPUT_PATH)


if __name__ == "__main__":
    main()
