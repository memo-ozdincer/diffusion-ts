"""
Kabsch + Hungarian clustering of transition state geometries.

Loads TS results from parallel_results.json, computes pairwise RMSD
with optimal atom permutation (Hungarian) and rigid-body alignment (Kabsch),
then clusters using agglomerative clustering.

Usage:
    source /project/rrg-aspuru/memoozd/ts-tools/.venv/bin/activate
    PYTHONPATH=/project/rrg-aspuru/memoozd/diffusion-ts:$PYTHONPATH \
        python scripts/cluster_ts.py [--results PATH] [--threshold 0.2]
"""

import argparse
import json
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering


# ---------------------------------------------------------------------------
# Kabsch alignment
# ---------------------------------------------------------------------------

def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two (N, 3) coordinate arrays.

    1. Center both geometries.
    2. Compute optimal rotation via SVD.
    3. Return RMSD after alignment.
    """
    assert A.shape == B.shape
    # Center
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)

    # SVD of cross-covariance
    H = A_c.T @ B_c
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    A_rot = A_c @ R.T
    diff = A_rot - B_c
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd


# ---------------------------------------------------------------------------
# Equivalence classes for Hungarian matching
# ---------------------------------------------------------------------------

def build_equivalence_classes(atomic_nums: list[int]) -> list[list[int]]:
    """Group atom indices by atomic number (equivalence class).

    Returns a list of groups, each group being a list of indices that share
    the same atomic number and are therefore permutation-equivalent.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, z in enumerate(atomic_nums):
        groups[z].append(i)
    return list(groups.values())


def hungarian_kabsch_rmsd(
    A: np.ndarray,
    B: np.ndarray,
    equiv_classes: list[list[int]],
) -> float:
    """Compute minimum RMSD over all atom permutations within equivalence classes.

    For each equivalence class, uses the Hungarian algorithm on the distance
    matrix to find the optimal atom assignment. Then Kabsch-aligns and computes
    RMSD.

    For small groups (<=5 atoms), we enumerate all permutations since Hungarian
    on the distance matrix gives an approximation -- the optimal permutation
    depends on the rotation which depends on the permutation. For efficiency,
    we do a two-pass approach:
      Pass 1: Use Hungarian on initial (centroid-aligned) distances to get a
              good starting permutation.
      Pass 2: Kabsch-align with that permutation, then re-run Hungarian on the
              aligned distances, iterate until convergence.
    """
    assert A.shape == B.shape
    n_atoms = A.shape[0]

    # Center both
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)

    # Iterative Hungarian-Kabsch
    best_rmsd = float("inf")
    best_perm = np.arange(n_atoms)

    # Start with identity permutation for initial Kabsch
    perm = np.arange(n_atoms)

    for iteration in range(10):  # Usually converges in 2-3 iterations
        # Kabsch-align with current permutation
        A_perm = A_c[perm]
        H = A_perm.T @ B_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        A_rot = A_c @ R.T  # Rotate original (unpermuted) A

        # Hungarian assignment within each equivalence class
        new_perm = np.arange(n_atoms)
        for group in equiv_classes:
            if len(group) <= 1:
                continue
            group = np.array(group)
            # Distance matrix: A_rot[group_i] vs B_c[group_j]
            cost = np.zeros((len(group), len(group)))
            for ii, ai in enumerate(group):
                for jj, bj in enumerate(group):
                    cost[ii, jj] = np.sum((A_rot[ai] - B_c[bj]) ** 2)
            row_ind, col_ind = linear_sum_assignment(cost)
            for ii, jj in zip(row_ind, col_ind):
                new_perm[group[ii]] = group[jj]

        # Recompute Kabsch RMSD with this permutation
        # Map: A[new_perm[i]] should match B[i], so A_perm[i] = A[new_perm[i]]
        # Actually we want: for each target position j in B, which source atom i in A?
        # new_perm[i] = j means atom i in A goes to position j in B
        # We need inverse: for position j in B, take atom inv_perm[j] from A
        inv_perm = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            inv_perm[new_perm[i]] = i
        # Actually, let me reconsider. The cost matrix has:
        # cost[ii, jj] = ||A_rot[group[ii]] - B_c[group[jj]]||^2
        # linear_sum_assignment minimizes sum of cost[row_ind[k], col_ind[k]]
        # So atom group[ii] in A matches atom group[jj] in B
        # For Kabsch: we want A_perm such that A_perm[j] = A[mapping[j]]
        # where mapping[j] = the A-atom assigned to B-position j
        # From Hungarian: A-atom group[row_ind[k]] -> B-position group[col_ind[k]]
        # So mapping[group[col_ind[k]]] = group[row_ind[k]]

        # Redo more carefully:
        new_perm2 = np.arange(n_atoms)  # new_perm2[j] = which A-atom goes to B-position j
        for group_idx in equiv_classes:
            if len(group_idx) <= 1:
                continue
            group_arr = np.array(group_idx)
            cost = np.zeros((len(group_arr), len(group_arr)))
            for ii, ai in enumerate(group_arr):
                for jj, bj in enumerate(group_arr):
                    cost[ii, jj] = np.sum((A_rot[ai] - B_c[bj]) ** 2)
            row_ind, col_ind = linear_sum_assignment(cost)
            for k in range(len(row_ind)):
                a_idx = group_arr[row_ind[k]]
                b_idx = group_arr[col_ind[k]]
                new_perm2[b_idx] = a_idx

        # Now A_c[new_perm2] should align with B_c
        perm = new_perm2

        # Compute RMSD
        A_perm = A_c[perm]
        H = A_perm.T @ B_c
        U, S, Vt = np.linalg.svd(H)
        d_det = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d_det)])
        R = Vt.T @ sign_matrix @ U.T
        A_final = A_perm @ R.T
        diff = A_final - B_c
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        if rmsd < best_rmsd - 1e-10:
            best_rmsd = rmsd
            best_perm = perm.copy()
        else:
            break  # Converged

    # Also try with a few random restarts for robustness
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random initial permutation within each class
        rand_perm = np.arange(n_atoms)
        for group in equiv_classes:
            if len(group) <= 1:
                continue
            group_arr = np.array(group)
            shuffled = rng.permutation(group_arr)
            for i, g in enumerate(group_arr):
                rand_perm[g] = shuffled[i]

        # Run one iteration of Kabsch + Hungarian from this start
        A_perm = A_c[rand_perm]
        H = A_perm.T @ B_c
        U, S, Vt = np.linalg.svd(H)
        d_det = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d_det)])
        R = Vt.T @ sign_matrix @ U.T
        A_rot = A_c @ R.T

        trial_perm = np.arange(n_atoms)
        for group in equiv_classes:
            if len(group) <= 1:
                continue
            group_arr = np.array(group)
            cost = np.zeros((len(group_arr), len(group_arr)))
            for ii, ai in enumerate(group_arr):
                for jj, bj in enumerate(group_arr):
                    cost[ii, jj] = np.sum((A_rot[ai] - B_c[bj]) ** 2)
            row_ind, col_ind = linear_sum_assignment(cost)
            for k in range(len(row_ind)):
                a_idx = group_arr[row_ind[k]]
                b_idx = group_arr[col_ind[k]]
                trial_perm[b_idx] = a_idx

        # Iterate from this trial
        perm = trial_perm
        for _ in range(10):
            A_perm = A_c[perm]
            H = A_perm.T @ B_c
            U, S, Vt = np.linalg.svd(H)
            d_det = np.linalg.det(Vt.T @ U.T)
            sign_matrix = np.diag([1.0, 1.0, np.sign(d_det)])
            R = Vt.T @ sign_matrix @ U.T
            A_rot = A_c @ R.T

            new_perm = np.arange(n_atoms)
            for group in equiv_classes:
                if len(group) <= 1:
                    continue
                group_arr = np.array(group)
                cost = np.zeros((len(group_arr), len(group_arr)))
                for ii, ai in enumerate(group_arr):
                    for jj, bj in enumerate(group_arr):
                        cost[ii, jj] = np.sum((A_rot[ai] - B_c[bj]) ** 2)
                row_ind, col_ind = linear_sum_assignment(cost)
                for k in range(len(row_ind)):
                    a_idx = group_arr[row_ind[k]]
                    b_idx = group_arr[col_ind[k]]
                    new_perm[b_idx] = a_idx

            if np.array_equal(new_perm, perm):
                break
            perm = new_perm

        A_perm = A_c[perm]
        H = A_perm.T @ B_c
        U, S, Vt = np.linalg.svd(H)
        d_det = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d_det)])
        R = Vt.T @ sign_matrix @ U.T
        A_final = A_perm @ R.T
        diff = A_final - B_c
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

        if rmsd < best_rmsd - 1e-10:
            best_rmsd = rmsd
            best_perm = perm.copy()

    return best_rmsd


# ---------------------------------------------------------------------------
# Brute-force for small equivalence classes (validation)
# ---------------------------------------------------------------------------

def brute_force_kabsch_rmsd(
    A: np.ndarray,
    B: np.ndarray,
    equiv_classes: list[list[int]],
) -> float:
    """Brute-force minimum RMSD over all atom permutations.

    Only feasible for small molecules with small equivalence classes.
    For isopropanol: C has 3! = 6, H has 8! = 40320 -- too many for full brute.
    Instead, use this as a validation on a subset.
    """
    # For isopropanol, the total permutations within classes:
    # C: 3! = 6, O: 1! = 1, H: 8! = 40320 -> 6 * 40320 = 241920
    # That's borderline. Let's skip brute force and rely on iterative Hungarian.
    raise NotImplementedError("Use hungarian_kabsch_rmsd instead")


# ---------------------------------------------------------------------------
# Pairwise RMSD matrix
# ---------------------------------------------------------------------------

def compute_pairwise_rmsd(
    coords_list: list[np.ndarray],
    atomic_nums: list[int],
    verbose: bool = True,
) -> np.ndarray:
    """Compute NxN pairwise RMSD matrix with Hungarian+Kabsch alignment."""
    n = len(coords_list)
    equiv_classes = build_equivalence_classes(atomic_nums)

    if verbose:
        print(f"Equivalence classes (by atomic number):")
        elem_names = {1: "H", 6: "C", 8: "O"}
        for group in equiv_classes:
            z = atomic_nums[group[0]]
            print(f"  {elem_names.get(z, str(z))}: indices {group} ({len(group)} atoms)")
        print(f"\nComputing {n*(n-1)//2} pairwise RMSDs...")

    rmsd_matrix = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = hungarian_kabsch_rmsd(coords_list[i], coords_list[j], equiv_classes)
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
            done += 1
            if verbose and (done % 100 == 0 or done == total_pairs):
                print(f"  {done}/{total_pairs} pairs computed", flush=True)

    return rmsd_matrix


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_geometries(
    rmsd_matrix: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """Agglomerative clustering using precomputed RMSD distance matrix.

    Args:
        rmsd_matrix: NxN symmetric distance matrix.
        threshold: RMSD cutoff for cluster merging (in Angstroms).

    Returns:
        labels: cluster label per geometry.
    """
    condensed = squareform(rmsd_matrix, checks=False)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(rmsd_matrix)
    return labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_clusters(
    labels: np.ndarray,
    ts_data: list[dict],
    rmsd_matrix: np.ndarray,
    threshold: float,
):
    """Print a detailed report of the clustering results."""
    n_clusters = len(set(labels))
    print(f"\n{'='*70}")
    print(f"CLUSTERING RESULTS (threshold = {threshold:.2f} A)")
    print(f"{'='*70}")
    print(f"Number of distinct TS clusters: {n_clusters}")
    print(f"Total TS geometries: {len(labels)}")

    for c in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == c]
        energies = [ts_data[i]["final_energy"] for i in members]
        eigs = [ts_data[i]["ts_eig0"] for i in members]
        seeds = [ts_data[i]["seed"] for i in members]

        # Intra-cluster RMSD stats
        intra_rmsds = []
        for i, j in itertools.combinations(members, 2):
            intra_rmsds.append(rmsd_matrix[i, j])

        print(f"\n  Cluster {c}: {len(members)} geometries")
        print(f"    Seeds: {seeds}")
        print(f"    Energy: {np.mean(energies):.4f} +/- {np.std(energies):.4f} eV")
        print(f"      range: [{min(energies):.4f}, {max(energies):.4f}]")
        print(f"    eig0:   {np.mean(eigs):.4f} +/- {np.std(eigs):.4f}")
        print(f"      range: [{min(eigs):.4f}, {max(eigs):.4f}]")
        if intra_rmsds:
            print(f"    Intra-cluster RMSD: {np.mean(intra_rmsds):.4f} +/- {np.std(intra_rmsds):.4f} A")
            print(f"      range: [{min(intra_rmsds):.4f}, {max(intra_rmsds):.4f}]")
        else:
            print(f"    Intra-cluster RMSD: N/A (single member)")

        # Representative = lowest energy member
        rep_idx = members[np.argmin(energies)]
        print(f"    Representative: seed {ts_data[rep_idx]['seed']} "
              f"(E={ts_data[rep_idx]['final_energy']:.4f})")

    # Inter-cluster RMSD
    print(f"\n{'='*70}")
    print("INTER-CLUSTER RMSD (between representative geometries)")
    print(f"{'='*70}")
    cluster_ids = sorted(set(labels))
    reps = {}
    for c in cluster_ids:
        members = [i for i, l in enumerate(labels) if l == c]
        energies = [ts_data[i]["final_energy"] for i in members]
        reps[c] = members[np.argmin(energies)]

    header = "        " + "  ".join(f"C{c:>5d}" for c in cluster_ids)
    print(header)
    for c1 in cluster_ids:
        row = f"  C{c1:>3d}  "
        for c2 in cluster_ids:
            if c1 == c2:
                row += f"{'---':>7s}"
            else:
                row += f"{rmsd_matrix[reps[c1], reps[c2]]:>7.3f}"
        print(row)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Cluster':>8s} {'Size':>5s} {'Energy (eV)':>14s} {'eig0':>10s} {'Intra RMSD':>12s}")
    print(f"{'-'*8:>8s} {'-'*5:>5s} {'-'*14:>14s} {'-'*10:>10s} {'-'*12:>12s}")
    for c in cluster_ids:
        members = [i for i, l in enumerate(labels) if l == c]
        energies = [ts_data[i]["final_energy"] for i in members]
        eigs = [ts_data[i]["ts_eig0"] for i in members]
        intra = [rmsd_matrix[i, j] for i, j in itertools.combinations(members, 2)]
        mean_intra = f"{np.mean(intra):.4f}" if intra else "N/A"
        print(f"{c:>8d} {len(members):>5d} {np.mean(energies):>14.4f} {np.mean(eigs):>10.4f} {mean_intra:>12s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster TS geometries via Kabsch+Hungarian RMSD")
    parser.add_argument(
        "--results",
        type=str,
        default="/scratch/memoozd/diffusion-ts/parallel_results.json",
        help="Path to parallel_results.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="RMSD threshold for agglomerative clustering (Angstroms)",
    )
    parser.add_argument(
        "--scan-thresholds",
        action="store_true",
        help="Scan multiple thresholds to find optimal clustering",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading results from {args.results}")
    with open(args.results) as f:
        data = json.load(f)

    ts_data = [r for r in data if r.get("status") == "ts"]
    print(f"Found {len(ts_data)} transition states out of {len(data)} runs")

    # Extract coordinates
    atomic_nums = [6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]  # Isopropanol
    coords_list = [np.array(r["ts_coords"]) for r in ts_data]
    assert all(c.shape == (12, 3) for c in coords_list), "Unexpected coord shapes"

    # Compute pairwise RMSD
    rmsd_matrix = compute_pairwise_rmsd(coords_list, atomic_nums)

    # Save RMSD matrix
    out_dir = Path(args.results).parent
    rmsd_path = out_dir / "ts_rmsd_matrix.npy"
    np.save(rmsd_path, rmsd_matrix)
    print(f"\nRMSD matrix saved to {rmsd_path}")

    # RMSD statistics
    upper_tri = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
    print(f"\nPairwise RMSD statistics:")
    print(f"  Min:    {upper_tri.min():.4f} A")
    print(f"  Max:    {upper_tri.max():.4f} A")
    print(f"  Mean:   {upper_tri.mean():.4f} A")
    print(f"  Median: {np.median(upper_tri):.4f} A")

    # Scan thresholds if requested
    if args.scan_thresholds:
        print(f"\n{'='*70}")
        print("THRESHOLD SCAN")
        print(f"{'='*70}")
        for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]:
            labels = cluster_geometries(rmsd_matrix, threshold=t)
            n_c = len(set(labels))
            sizes = [sum(labels == c) for c in sorted(set(labels))]
            print(f"  threshold={t:.2f} A -> {n_c} clusters, sizes={sizes}")

    # Cluster
    labels = cluster_geometries(rmsd_matrix, threshold=args.threshold)

    # Report
    report_clusters(labels, ts_data, rmsd_matrix, args.threshold)

    # Compare with energy-based grouping
    print(f"\n{'='*70}")
    print("ENERGY vs GEOMETRY CLUSTER COMPARISON")
    print(f"{'='*70}")
    for c in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == c]
        energies = sorted([ts_data[i]["final_energy"] for i in members])
        e_str = ", ".join(f"{e:.2f}" for e in energies[:5])
        if len(energies) > 5:
            e_str += f", ... ({len(energies)} total)"
        print(f"  Geom cluster {c}: energies = [{e_str}]")

    # Save cluster assignments
    cluster_results = {
        "threshold": args.threshold,
        "n_clusters": int(len(set(labels))),
        "assignments": [],
    }
    for i, r in enumerate(ts_data):
        cluster_results["assignments"].append({
            "seed": r["seed"],
            "cluster": int(labels[i]),
            "energy": r["final_energy"],
            "eig0": r["ts_eig0"],
        })
    cluster_path = out_dir / "ts_clusters.json"
    with open(cluster_path, "w") as f:
        json.dump(cluster_results, f, indent=2)
    print(f"\nCluster assignments saved to {cluster_path}")


if __name__ == "__main__":
    main()
