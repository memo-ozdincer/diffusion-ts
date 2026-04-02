#!/usr/bin/env python
"""Parallel TS search using adaptive NR/GAD optimizer.

Uses find_transition_state() which switches between NR (n_neg >= 2) and
GAD (n_neg < 2) every step. Convergence: n_neg == 1 and rms_force < threshold.
"""

import sys, os, time, json, torch
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

ISOPROPANOL_COORDS = torch.tensor([
    [ 0.000,  0.000,  0.000], [-1.270,  0.760,  0.000], [ 1.270,  0.760,  0.000],
    [ 0.000, -0.930,  1.100], [ 0.000, -0.670, -0.870], [ 0.000, -0.400,  1.920],
    [-1.270,  1.410,  0.870], [-1.270,  1.410, -0.870], [-2.160,  0.120,  0.000],
    [ 1.270,  1.410,  0.870], [ 1.270,  1.410, -0.870], [ 2.160,  0.120,  0.000],
], dtype=torch.float32)
ATOMIC_NUMS = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
ATOMSYMBOLS = ["C", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"]


def worker(seed, noise_std=0.3, max_steps=10000, force_converged=0.01):
    """Single TS search with adaptive NR/GAD."""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    from src.dependencies.scine_calculator import ScineSparrowCalculator
    from src.dependencies.calculators import make_scine_predict_fn
    from src.core_algos.saddle_optimizer import (
        find_transition_state, TSOptimizerConfig, vib_eig,
        rms_force, max_atomic_force,
        run_nr_minimization, NRConfig,
        run_gad_saddle_search, GADConfig,
    )

    functional = os.environ.get("TS_FUNCTIONAL", "DFTB0")
    calc = ScineSparrowCalculator(functional=functional)
    predict_fn = make_scine_predict_fn(calc)

    torch.manual_seed(seed)
    noisy_coords = ISOPROPANOL_COORDS + torch.randn_like(ISOPROPANOL_COORDS) * noise_std

    # Check min interatomic distance
    c = noisy_coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    min_dist = float(dist.min().item())
    if min_dist < 0.4:
        return {"seed": seed, "status": "skip", "reason": f"min_dist={min_dist:.3f}"}

    t0 = time.time()
    noise_vec = noisy_coords - ISOPROPANOL_COORDS

    # Build starting geometries: full noise + reduced noise
    starts = [("full", noisy_coords)]
    if noise_std >= 0.3:
        c60 = ISOPROPANOL_COORDS + noise_vec * 0.6
        d = c60.unsqueeze(0) - c60.unsqueeze(1)
        if float(d.norm(dim=2).add_(torch.eye(12) * 1e10).min().item()) >= 0.4:
            starts.append(("60pct", c60))

    # Phase 1: NR minimization from each start, collect handoff candidates
    nr_cfg = NRConfig(n_steps=1000, force_converged=0.01, min_interatomic_dist=0.4)
    candidates = []  # (label, coords, n_neg, force, step)

    for start_label, start_coords in starts:
        nr_result = run_nr_minimization(predict_fn, start_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
        traj = nr_result["trajectory"]

        # Collect snapshots where n_neg <= 2
        nneg_points = {0: [], 1: [], 2: []}
        for rec in traj:
            if rec["n_neg"] in nneg_points and "coords" in rec:
                nneg_points[rec["n_neg"]].append(rec)

        # Build candidate list: prefer n_neg=1, then n_neg=0, then n_neg=2
        for nn in [1, 0, 2]:
            pts = nneg_points[nn]
            if pts:
                # Take last, first, and middle
                candidates.append((f"{start_label}_last_nn{nn}", pts[-1]["coords"], nn))
                if len(pts) > 2:
                    candidates.append((f"{start_label}_mid_nn{nn}", pts[len(pts)//2]["coords"], nn))
                if len(pts) > 1:
                    candidates.append((f"{start_label}_first_nn{nn}", pts[0]["coords"], nn))

        # If NR converged, also use final coords
        if nr_result["converged"]:
            candidates.append((f"{start_label}_min", nr_result["final_coords"], nr_result["final_n_neg"]))

    if not candidates:
        wall = time.time() - t0
        return {"seed": seed, "status": "no_ts", "converged": False,
                "final_energy": 0, "final_force": 999, "final_max_force": 999,
                "final_n_neg": -1, "total_steps": 0, "n_nr_steps": 0, "n_gad_steps": 0,
                "start": "none", "wall": wall, "reason": "no candidates"}

    # Phase 2: Try GAD from each candidate until one converges
    gad_cfg = GADConfig(
        n_steps=max_steps, dt=0.01, dt_max=0.3, max_atom_disp=0.5,
        min_interatomic_dist=0.4, track_mode=True, project_gradient=True,
    )

    result = None
    start_label = None
    for label, cand_coords, cand_nneg in candidates:
        gad_result = run_gad_saddle_search(
            predict_fn, cand_coords, ATOMIC_NUMS, ATOMSYMBOLS, gad_cfg,
            force_converged=force_converged,
        )
        if gad_result["converged"]:
            result = gad_result
            start_label = label
            break
        # Keep best non-converged
        if result is None or (
            gad_result["final_morse_index"] == 1 and
            gad_result.get("final_force_norm", 999) < (result.get("final_force_norm", 999))
        ):
            result = gad_result
            start_label = label

    wall = time.time() - t0

    converged = result.get("converged", False)
    out = {
        "seed": seed,
        "status": "ts" if converged else "no_ts",
        "converged": converged,
        "start": start_label,
        "final_energy": result.get("final_energy", 0),
        "final_force": result.get("final_force_norm", 999),
        "final_max_force": 0,
        "final_n_neg": result.get("final_morse_index", result.get("final_n_neg", -1)),
        "total_steps": result.get("total_steps", 0),
        "n_nr_steps": 0,
        "n_gad_steps": result.get("total_steps", 0),
        "n_candidates": len(candidates),
        "wall": wall,
    }

    if converged:
        ts_coords = result["final_coords"]
        out_ts = predict_fn(ts_coords, ATOMIC_NUMS, do_hessian=True)
        evals_ts, _, _ = vib_eig(out_ts["hessian"], ts_coords, ATOMSYMBOLS)
        out["ts_eig0"] = float(evals_ts[0])
        out["ts_coords"] = ts_coords.tolist()
        out["final_force"] = result["final_force_norm"]
        out["final_max_force"] = max_atomic_force(out_ts["forces"].reshape(-1, 3))

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--force-converged", type=float, default=0.01)
    parser.add_argument("--functional", type=str, default="DFTB0")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    os.environ["TS_FUNCTIONAL"] = args.functional

    print(f"Adaptive NR/GAD TS search: {args.n_seeds} seeds, {args.n_workers} workers, "
          f"noise={args.noise}A, F<{args.force_converged}, max_steps={args.max_steps}")
    print(f"Seeds: {seeds}")
    print()

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(worker, s, args.noise, args.max_steps, args.force_converged): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                if r["status"] == "ts":
                    print(f"  seed={r['seed']:3d} TS! E={r['final_energy']:.4f} "
                          f"eig0={r.get('ts_eig0', 0):.4f} "
                          f"rmsF={r['final_force']:.6f} maxF={r.get('final_max_force', 0):.6f} "
                          f"NR={r['n_nr_steps']} GAD={r['n_gad_steps']} "
                          f"total={r['total_steps']} {r['wall']:.1f}s")
                elif r["status"] == "no_ts":
                    print(f"  seed={r['seed']:3d} no TS  n_neg={r.get('final_n_neg', '?')} "
                          f"rmsF={r['final_force']:.4f} "
                          f"NR={r['n_nr_steps']} GAD={r['n_gad_steps']} "
                          f"total={r['total_steps']} {r['wall']:.1f}s")
                else:
                    print(f"  seed={r['seed']:3d} {r['status']}: {r.get('reason', '')}")
            except Exception as e:
                print(f"  seed={seed} CRASH: {e}")
                import traceback; traceback.print_exc()
                results.append({"seed": seed, "status": "crash", "reason": str(e)})

    total_wall = time.time() - t0

    n_ts = sum(1 for r in results if r["status"] == "ts")
    n_no_ts = sum(1 for r in results if r["status"] == "no_ts")
    n_skip = sum(1 for r in results if r["status"] in ("skip", "crash"))

    print(f"\n{'='*70}")
    print(f"Summary: {n_ts}/{len(seeds)} TS found  ({n_no_ts} no_ts, {n_skip} skip/crash)")
    valid = n_ts + n_no_ts
    if valid > 0:
        print(f"Rate excl. skips: {n_ts}/{valid} = {100*n_ts/valid:.0f}%")
    print(f"Wall time: {total_wall:.1f}s")

    # Force stats for converged
    ts_results = [r for r in results if r["status"] == "ts"]
    if ts_results:
        forces = [r["final_force"] for r in ts_results]
        print(f"Force (rms): mean={sum(forces)/len(forces):.6f}, "
              f"min={min(forces):.6f}, max={max(forces):.6f}")
        nr_steps = [r["n_nr_steps"] for r in ts_results]
        gad_steps = [r["n_gad_steps"] for r in ts_results]
        print(f"NR steps: mean={sum(nr_steps)/len(nr_steps):.0f}")
        print(f"GAD steps: mean={sum(gad_steps)/len(gad_steps):.0f}")
    print(f"{'='*70}")

    # Failure analysis
    no_ts = [r for r in results if r["status"] == "no_ts"]
    if no_ts:
        print("\nFailure analysis:")
        for r in sorted(no_ts, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']:3d} n_neg={r.get('final_n_neg', '?')} "
                  f"rmsF={r['final_force']:.4f} steps={r['total_steps']}")

    out_dir = os.environ.get("SCRATCH_DIR", "/scratch/memoozd/diffusion-ts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adaptive_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
