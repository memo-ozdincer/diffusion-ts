#!/usr/bin/env python
"""Parallel TS search: 10 random isopropanol starts, early NR→GAD handoff.

Strategy (proven on single geometry):
  1. NR from 0.2A-noisy geometry (saves coords at each step)
  2. Hand off to GAD at first n_neg=1 snapshot
  3. GAD Euler moderate (dt=0.01, dt_max=0.3, max_disp=0.5) until n_neg=1 AND force<0.01
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


def worker(seed, noise_std=0.2):
    """Single TS search: NR (early handoff at n_neg=1) → GAD moderate."""
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

    # Check min interatomic distance
    c = noisy_coords.reshape(-1, 3)
    diff = c.unsqueeze(0) - c.unsqueeze(1)
    dist = diff.norm(dim=2) + torch.eye(c.shape[0]) * 1e10
    min_dist = float(dist.min().item())
    if min_dist < 0.5:
        return {"seed": seed, "status": "skip", "reason": f"min_dist={min_dist:.3f}"}

    t0 = time.time()

    # Phase 1: NR (500 steps, will stop early if converged)
    nr_cfg = NRConfig(n_steps=500, force_converged=0.01)
    nr_result = run_nr_minimization(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)

    # Collect ALL n_neg=1 snapshots (use last one — more relaxed geometry)
    nneg1_points = [rec for rec in nr_result["trajectory"] if rec["n_neg"] == 1 and "coords" in rec]
    nneg2_points = [rec for rec in nr_result["trajectory"] if rec["n_neg"] == 2 and "coords" in rec]

    # Build candidate handoff list: last n_neg=1 first, then first n_neg=1, then last n_neg=2
    candidates = []
    if nneg1_points:
        candidates.append(("last_nneg1", nneg1_points[-1]))
        if len(nneg1_points) > 1:
            candidates.append(("first_nneg1", nneg1_points[0]))
    if nneg2_points:
        candidates.append(("last_nneg2", nneg2_points[-1]))

    if not candidates:
        return {
            "seed": seed, "status": "no_handoff",
            "reason": "n_neg never reached 1 or 2 during NR",
            "nr_steps": nr_result["total_steps"],
            "wall": time.time() - t0,
        }

    # Phase 2: Try GAD from each candidate until one works
    gad_cfg = SOGADConfig(
        n_steps=3000,
        dt=0.01,
        dt_max=0.3,
        max_atom_disp=0.5,
    )

    best_result = None
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
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-workers", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--functional", type=str, default="DFTB0")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    # Pass functional to workers via environment
    os.environ["TS_FUNCTIONAL"] = args.functional

    print(f"Parallel TS search: {args.n_seeds} seeds, {args.n_workers} workers, "
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
                # Print as they complete
                if r["status"] == "ts":
                    print(f"  seed={r['seed']:3d} TS! E={r['final_energy']:.4f} "
                          f"eig0={r.get('ts_eig0', '?'):.4f} "
                          f"NR={r['nr_steps']}→GAD={r['gad_steps']} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} "
                          f"{r['wall']:.1f}s")
                elif r["status"] == "no_ts":
                    print(f"  seed={r['seed']:3d} no TS  morse={r['final_morse_idx']} "
                          f"F={r.get('final_force', '?'):.4f} "
                          f"{r.get('handoff_label','?')}@step{r['handoff_step']} "
                          f"NR={r['nr_steps']}→GAD={r['gad_steps']} {r['wall']:.1f}s")
                else:
                    print(f"  seed={r['seed']:3d} {r['status']}: {r.get('reason', '')}")
            except Exception as e:
                print(f"  seed={seed} CRASH: {e}")
                results.append({"seed": seed, "status": "crash", "reason": str(e)})

    total_wall = time.time() - t0

    # Summary
    n_ts = sum(1 for r in results if r["status"] == "ts")
    n_no_ts = sum(1 for r in results if r["status"] == "no_ts")
    n_skip = sum(1 for r in results if r["status"] in ("skip", "no_handoff", "crash"))

    print(f"\n{'='*60}")
    print(f"Summary: {n_ts}/{len(seeds)} TS found  ({n_no_ts} no_ts, {n_skip} skip/crash)")
    print(f"Wall time: {total_wall:.1f}s  ({total_wall/len(seeds):.1f}s/sample sequential equiv)")
    print(f"{'='*60}")

    # Save results
    out_dir = os.environ.get("SCRATCH_DIR", "/scratch/memoozd/diffusion-ts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "parallel_results.json")
    serializable = results  # ts_coords already .tolist()'d
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
