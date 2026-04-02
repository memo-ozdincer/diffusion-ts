#!/usr/bin/env python
"""Parallel TS search v2: improved convergence at high noise levels.

Changes from v1 (test_parallel_quick.py):
  - Lower min_dist skip threshold (0.3A instead of 0.5A)
  - More NR steps (1000) for better trajectory sampling
  - Expanded candidate set: also try minimum (n_neg=0) handoff with eigvec perturbation
  - More GAD steps (5000) with adjusted parameters
  - Restart-from-minimum strategy when main candidates fail
  - Wider spread of handoff candidates (middle n_neg=1 points too)
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


def _min_dist(coords):
    """Minimum interatomic distance."""
    c = coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    return float(dist.min().item())


def worker(seed, noise_std=0.3):
    """Single TS search with improved strategy for high noise."""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)

    from src.dependencies.scine_calculator import ScineSparrowCalculator
    from src.dependencies.calculators import make_scine_predict_fn
    from src.core_algos.saddle_optimizer import (
        run_nr_minimization, NRConfig,
        run_gad_saddle_search, GADConfig as SOGADConfig,
        vib_eig,
    )

    functional = os.environ.get("TS_FUNCTIONAL", "DFTB0")
    calc = ScineSparrowCalculator(functional=functional)
    predict_fn = make_scine_predict_fn(calc)

    torch.manual_seed(seed)
    noisy_coords = ISOPROPANOL_COORDS + torch.randn_like(ISOPROPANOL_COORDS) * noise_std

    # Lowered min_dist threshold: 0.3A instead of 0.5A
    min_d = _min_dist(noisy_coords)
    if min_d < 0.3:
        return {"seed": seed, "status": "skip", "reason": f"min_dist={min_d:.3f}"}

    t0 = time.time()

    # ========== Phase 1: NR (more steps for better trajectory) ==========
    nr_cfg = NRConfig(n_steps=1000, force_converged=0.01, min_interatomic_dist=0.3)
    nr_result = run_nr_minimization(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)

    # Collect snapshots by n_neg value
    traj = nr_result["trajectory"]
    nneg0_points = [rec for rec in traj if rec["n_neg"] == 0 and "coords" in rec]
    nneg1_points = [rec for rec in traj if rec["n_neg"] == 1 and "coords" in rec]
    nneg2_points = [rec for rec in traj if rec["n_neg"] == 2 and "coords" in rec]
    nneg3_points = [rec for rec in traj if rec["n_neg"] == 3 and "coords" in rec]

    # ========== Build expanded candidate list ==========
    candidates = []

    # Priority 1: n_neg=1 points (best handoff)
    if nneg1_points:
        candidates.append(("last_nneg1", nneg1_points[-1]))
        # Add middle point too (not just first/last)
        if len(nneg1_points) > 2:
            mid = len(nneg1_points) // 2
            candidates.append(("mid_nneg1", nneg1_points[mid]))
        if len(nneg1_points) > 1:
            candidates.append(("first_nneg1", nneg1_points[0]))

    # Priority 2: n_neg=2 points
    if nneg2_points:
        candidates.append(("last_nneg2", nneg2_points[-1]))
        if len(nneg2_points) > 1:
            candidates.append(("first_nneg2", nneg2_points[0]))

    # Priority 3: n_neg=3 (early trajectory, high index)
    if nneg3_points:
        candidates.append(("last_nneg3", nneg3_points[-1]))

    # Priority 4: n_neg=0 minimum with perturbation along lowest eigenvector
    if nneg0_points and nr_result["converged"]:
        # Use the converged minimum -- we'll perturb it
        min_coords = nr_result["final_coords"]
        out_min = predict_fn(min_coords, ATOMIC_NUMS, do_hessian=True)
        evals_min, evecs_min, _ = vib_eig(out_min["hessian"], min_coords, ATOMSYMBOLS)

        # Perturb along lowest eigenvector (most likely TS direction)
        v0 = evecs_min[:, 0].reshape(-1, 3).to(min_coords.dtype)
        for scale in [0.3, 0.5, -0.3]:
            perturbed = min_coords + scale * v0
            if _min_dist(perturbed) >= 0.3:
                fake_rec = {"step": -1, "n_neg": 0, "coords": perturbed}
                candidates.append((f"min_perturb_{scale}", fake_rec))

    if not candidates:
        return {
            "seed": seed, "status": "no_handoff",
            "reason": "no suitable handoff point found",
            "nr_steps": nr_result["total_steps"],
            "wall": time.time() - t0,
        }

    # ========== Phase 2: GAD with expanded parameters ==========
    gad_cfg = SOGADConfig(
        n_steps=5000,
        dt=0.01,
        dt_max=0.3,
        max_atom_disp=0.5,
        min_interatomic_dist=0.3,
    )

    best_result = None
    best_label = None
    best_step = None
    best_nneg = None

    for label, rec in candidates:
        gad_result = run_gad_saddle_search(
            predict_fn, rec["coords"], ATOMIC_NUMS, ATOMSYMBOLS, gad_cfg)

        if gad_result["converged"]:
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]
            break

        # Keep track of best non-converged attempt
        if best_result is None or (
            gad_result["final_morse_index"] == 1 and
            gad_result.get("final_force_norm", 999) < best_result.get("final_force_norm", 999)
        ):
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]

    wall = time.time() - t0
    gad_result = best_result

    result = {
        "seed": seed,
        "status": "ts" if gad_result["converged"] else "no_ts",
        "nr_steps": nr_result["total_steps"],
        "handoff_step": best_step,
        "handoff_nneg": best_nneg,
        "handoff_label": best_label,
        "gad_steps": gad_result["total_steps"],
        "gad_converged": gad_result["converged"],
        "final_energy": gad_result["final_energy"],
        "final_morse_idx": gad_result["final_morse_index"],
        "final_force": gad_result.get("final_force_norm", None),
        "n_candidates_tried": len(candidates),
        "wall": wall,
    }

    if gad_result["converged"]:
        ts_coords = gad_result["final_coords"]
        out_ts = predict_fn(ts_coords, ATOMIC_NUMS, do_hessian=True)
        evals_ts, _, _ = vib_eig(out_ts["hessian"], ts_coords, ATOMSYMBOLS)
        result["ts_eig0"] = float(evals_ts[0])
        result["ts_coords"] = ts_coords.tolist()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=25)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--functional", type=str, default="DFTB0")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    os.environ["TS_FUNCTIONAL"] = args.functional

    print(f"Parallel TS search v2: {args.n_seeds} seeds, {args.n_workers} workers, "
          f"noise={args.noise}A, functional={args.functional}")
    print(f"Seeds: {seeds}")
    print()

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(worker, s, args.noise): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                if r["status"] == "ts":
                    print(f"  seed={r['seed']:3d} TS! E={r['final_energy']:.4f} "
                          f"eig0={r.get('ts_eig0', '?'):.4f} "
                          f"NR={r['nr_steps']}->GAD={r['gad_steps']} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} "
                          f"tried={r.get('n_candidates_tried', '?')} "
                          f"{r['wall']:.1f}s")
                elif r["status"] == "no_ts":
                    print(f"  seed={r['seed']:3d} no TS  morse={r['final_morse_idx']} "
                          f"F={r.get('final_force', '?'):.4f} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} "
                          f"tried={r.get('n_candidates_tried', '?')} "
                          f"NR={r['nr_steps']}->GAD={r['gad_steps']} {r['wall']:.1f}s")
                else:
                    print(f"  seed={r['seed']:3d} {r['status']}: {r.get('reason', '')}")
            except Exception as e:
                print(f"  seed={seed} CRASH: {e}")
                import traceback; traceback.print_exc()
                results.append({"seed": seed, "status": "crash", "reason": str(e)})

    total_wall = time.time() - t0

    n_ts = sum(1 for r in results if r["status"] == "ts")
    n_no_ts = sum(1 for r in results if r["status"] == "no_ts")
    n_skip = sum(1 for r in results if r["status"] in ("skip", "no_handoff", "crash"))

    print(f"\n{'='*60}")
    print(f"Summary: {n_ts}/{len(seeds)} TS found  ({n_no_ts} no_ts, {n_skip} skip/crash)")
    print(f"Convergence rate: {n_ts}/{len(seeds)} = {100*n_ts/len(seeds):.0f}%")
    print(f"Rate excl. skips: {n_ts}/{n_ts+n_no_ts} = {100*n_ts/(n_ts+n_no_ts):.0f}%" if (n_ts+n_no_ts) > 0 else "")
    print(f"Wall time: {total_wall:.1f}s  ({total_wall/len(seeds):.1f}s/sample sequential equiv)")
    print(f"{'='*60}")

    # Failure analysis
    no_ts_results = [r for r in results if r["status"] == "no_ts"]
    if no_ts_results:
        print("\nFailure analysis:")
        for r in sorted(no_ts_results, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']:3d} morse={r['final_morse_idx']} "
                  f"F={r.get('final_force', '?'):.4f} "
                  f"{r.get('handoff_label','?')} tried={r.get('n_candidates_tried', '?')}")

    out_dir = os.environ.get("SCRATCH_DIR", "/scratch/memoozd/diffusion-ts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "parallel_results_v2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
