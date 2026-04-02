#!/usr/bin/env python
"""Parallel TS search v3: aggressive convergence improvements.

Changes from v2:
  - GAD restart with perturbation when stuck at n_neg=2
  - Two-phase GAD: coarse (fast exploration) then fine (precision convergence)
  - Longer combined budget (8000 steps) split across phases
  - Better handling of near-converged cases (morse=1, force > 0.01)
  - NR with even more steps (2000) for 0.5A noise
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
    c = coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    return float(dist.min().item())


def run_gad_with_restarts(predict_fn, handoff_coords, atomic_nums, atomsymbols,
                           GADConfig, vib_eig, max_total_steps=8000):
    """Run GAD with automatic restarts when stuck.

    Strategy:
    1. Run coarse GAD (large dt, fast exploration)
    2. If stuck at n_neg=2 with low force, perturb along 2nd negative eigvec
    3. If n_neg=1 but force not converged, switch to fine dt
    4. Budget total steps across restarts
    """
    steps_used = 0
    best_result = None

    # Phase 1: Coarse exploration
    coarse_steps = min(4000, max_total_steps)
    gad_cfg = GADConfig(
        n_steps=coarse_steps,
        dt=0.01,
        dt_max=0.3,
        max_atom_disp=0.5,
        min_interatomic_dist=0.3,
    )

    from src.core_algos.saddle_optimizer import run_gad_saddle_search

    result = run_gad_saddle_search(
        predict_fn, handoff_coords, atomic_nums, atomsymbols, gad_cfg)
    steps_used += result["total_steps"]

    if result["converged"]:
        return result, steps_used

    # Check if we're stuck at n_neg=2 with low force -- perturb and retry
    remaining = max_total_steps - steps_used
    if remaining > 500 and result["final_morse_index"] == 2 and result.get("final_force_norm", 999) < 0.1:
        # Perturb along the second negative eigenvector to break out of index-2 saddle
        final_coords = result["final_coords"]
        out = predict_fn(final_coords, atomic_nums, do_hessian=True)
        evals, evecs, _ = vib_eig(out["hessian"], final_coords, atomsymbols)

        # Try perturbing along second eigenvector (to remove one negative eigenvalue)
        if evecs.shape[1] >= 2:
            v1 = evecs[:, 1].reshape(-1, 3).to(final_coords.dtype)
            for scale in [0.2, -0.2, 0.4, -0.4]:
                if remaining <= 0:
                    break
                perturbed = final_coords + scale * v1
                if _min_dist(perturbed) < 0.3:
                    continue

                retry_steps = min(2000, remaining)
                retry_cfg = GADConfig(
                    n_steps=retry_steps,
                    dt=0.005,
                    dt_max=0.2,
                    max_atom_disp=0.3,
                    min_interatomic_dist=0.3,
                )
                retry_result = run_gad_saddle_search(
                    predict_fn, perturbed, atomic_nums, atomsymbols, retry_cfg)
                steps_used += retry_result["total_steps"]
                remaining = max_total_steps - steps_used

                if retry_result["converged"]:
                    return retry_result, steps_used

                # Track best result
                if best_result is None or (
                    retry_result["final_morse_index"] == 1 and
                    retry_result.get("final_force_norm", 999) < (best_result.get("final_force_norm", 999) if best_result else 999)
                ):
                    best_result = retry_result

    # Check if n_neg=1 but force not converged -- use fine dt
    check_result = best_result if best_result is not None and best_result["final_morse_index"] == 1 else result
    remaining = max_total_steps - steps_used
    if remaining > 500 and check_result["final_morse_index"] == 1 and check_result.get("final_force_norm", 999) < 0.5:
        fine_cfg = GADConfig(
            n_steps=min(3000, remaining),
            dt=0.003,
            dt_max=0.1,
            max_atom_disp=0.2,
            min_interatomic_dist=0.3,
        )
        fine_result = run_gad_saddle_search(
            predict_fn, check_result["final_coords"], atomic_nums, atomsymbols, fine_cfg)
        steps_used += fine_result["total_steps"]

        if fine_result["converged"]:
            return fine_result, steps_used
        if fine_result.get("final_force_norm", 999) < (check_result.get("final_force_norm", 999)):
            check_result = fine_result

    # Return best result found
    final_result = best_result if best_result is not None else result
    if check_result.get("final_force_norm", 999) < final_result.get("final_force_norm", 999):
        final_result = check_result
    return final_result, steps_used


def worker(seed, noise_std=0.3):
    """Single TS search with v3 strategy."""
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

    min_d = _min_dist(noisy_coords)
    if min_d < 0.3:
        return {"seed": seed, "status": "skip", "reason": f"min_dist={min_d:.3f}"}

    t0 = time.time()

    # Phase 1: NR (2000 steps for high noise)
    nr_steps = 2000 if noise_std >= 0.4 else 1000
    nr_cfg = NRConfig(n_steps=nr_steps, force_converged=0.01, min_interatomic_dist=0.3)
    nr_result = run_nr_minimization(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)

    traj = nr_result["trajectory"]
    nneg0_points = [rec for rec in traj if rec["n_neg"] == 0 and "coords" in rec]
    nneg1_points = [rec for rec in traj if rec["n_neg"] == 1 and "coords" in rec]
    nneg2_points = [rec for rec in traj if rec["n_neg"] == 2 and "coords" in rec]
    nneg3_points = [rec for rec in traj if rec["n_neg"] == 3 and "coords" in rec]

    # Build candidates (wider variety)
    candidates = []
    if nneg1_points:
        candidates.append(("last_nneg1", nneg1_points[-1]))
        if len(nneg1_points) > 2:
            mid = len(nneg1_points) // 2
            candidates.append(("mid_nneg1", nneg1_points[mid]))
        if len(nneg1_points) > 4:
            q1 = len(nneg1_points) // 4
            q3 = 3 * len(nneg1_points) // 4
            candidates.append(("q1_nneg1", nneg1_points[q1]))
            candidates.append(("q3_nneg1", nneg1_points[q3]))
        if len(nneg1_points) > 1:
            candidates.append(("first_nneg1", nneg1_points[0]))

    if nneg2_points:
        candidates.append(("last_nneg2", nneg2_points[-1]))
        if len(nneg2_points) > 1:
            candidates.append(("first_nneg2", nneg2_points[0]))

    if nneg3_points:
        candidates.append(("last_nneg3", nneg3_points[-1]))

    # Minimum perturbation
    if nr_result["converged"] and nneg0_points:
        min_coords = nr_result["final_coords"]
        out_min = predict_fn(min_coords, ATOMIC_NUMS, do_hessian=True)
        evals_min, evecs_min, _ = vib_eig(out_min["hessian"], min_coords, ATOMSYMBOLS)
        v0 = evecs_min[:, 0].reshape(-1, 3).to(min_coords.dtype)
        for scale in [0.3, 0.5, -0.3, -0.5, 0.8]:
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

    # Phase 2: GAD with restarts for each candidate
    best_result = None
    best_label = None
    best_step = None
    best_nneg = None
    total_gad_steps = 0

    for label, rec in candidates:
        gad_result, steps = run_gad_with_restarts(
            predict_fn, rec["coords"], ATOMIC_NUMS, ATOMSYMBOLS,
            SOGADConfig, vib_eig, max_total_steps=8000)
        total_gad_steps += steps

        if gad_result["converged"]:
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]
            break

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
        "gad_steps": total_gad_steps,
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

    print(f"Parallel TS search v3: {args.n_seeds} seeds, {args.n_workers} workers, "
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
    if (n_ts + n_no_ts) > 0:
        print(f"Rate excl. skips: {n_ts}/{n_ts+n_no_ts} = {100*n_ts/(n_ts+n_no_ts):.0f}%")
    print(f"Wall time: {total_wall:.1f}s  ({total_wall/len(seeds):.1f}s/sample sequential equiv)")
    print(f"{'='*60}")

    no_ts_results = [r for r in results if r["status"] == "no_ts"]
    if no_ts_results:
        print("\nFailure analysis:")
        for r in sorted(no_ts_results, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']:3d} morse={r['final_morse_idx']} "
                  f"F={r.get('final_force', '?'):.4f} "
                  f"{r.get('handoff_label','?')} tried={r.get('n_candidates_tried', '?')}")

    out_dir = os.environ.get("SCRATCH_DIR", "/scratch/memoozd/diffusion-ts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "parallel_results_v3.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
