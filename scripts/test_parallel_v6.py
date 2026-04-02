#!/usr/bin/env python
"""Parallel TS search v6: post-hoc rescue for stuck saddles.

Changes from v5:
  - Post-hoc scoop-and-retry: after all candidates, if best has morse=2 + low force,
    re-minimize from that point and try GAD from the new minimum
  - Post-hoc fine GAD: if best has morse=1 + moderate force (< 1.0), run fine GAD
  - Better integration of multi-start candidates (interleave A and B candidates)
  - Expanded eigenvector perturbation scales for minimum perturbation
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


def collect_candidates_from_nr(nr_result, predict_fn, vib_eig, label_prefix=""):
    """Extract handoff candidates from NR trajectory."""
    traj = nr_result["trajectory"]
    nneg1_points = [rec for rec in traj if rec["n_neg"] == 1 and "coords" in rec]
    nneg2_points = [rec for rec in traj if rec["n_neg"] == 2 and "coords" in rec]
    nneg3_points = [rec for rec in traj if rec["n_neg"] == 3 and "coords" in rec]
    nneg0_points = [rec for rec in traj if rec["n_neg"] == 0 and "coords" in rec]

    candidates = []
    pfx = label_prefix

    if nneg1_points:
        candidates.append((f"{pfx}last_nneg1", nneg1_points[-1]))
        if len(nneg1_points) > 2:
            candidates.append((f"{pfx}mid_nneg1", nneg1_points[len(nneg1_points)//2]))
        if len(nneg1_points) > 4:
            candidates.append((f"{pfx}q1_nneg1", nneg1_points[len(nneg1_points)//4]))
            candidates.append((f"{pfx}q3_nneg1", nneg1_points[3*len(nneg1_points)//4]))
        if len(nneg1_points) > 1:
            candidates.append((f"{pfx}first_nneg1", nneg1_points[0]))

    if nneg2_points:
        candidates.append((f"{pfx}last_nneg2", nneg2_points[-1]))
        if len(nneg2_points) > 1:
            candidates.append((f"{pfx}first_nneg2", nneg2_points[0]))

    if nneg3_points:
        candidates.append((f"{pfx}last_nneg3", nneg3_points[-1]))

    # Minimum perturbation
    if nr_result["converged"] and nneg0_points:
        min_coords = nr_result["final_coords"]
        out_min = predict_fn(min_coords, ATOMIC_NUMS, do_hessian=True)
        evals_min, evecs_min, _ = vib_eig(out_min["hessian"], min_coords, ATOMSYMBOLS)
        for evec_idx in range(min(3, evecs_min.shape[1])):
            v = evecs_min[:, evec_idx].reshape(-1, 3).to(min_coords.dtype)
            for scale in [0.3, -0.3, 0.5, -0.5]:
                perturbed = min_coords + scale * v
                if _min_dist(perturbed) >= 0.3:
                    fake_rec = {"step": -1, "n_neg": 0, "coords": perturbed}
                    candidates.append((f"{pfx}min_ev{evec_idx}_{scale}", fake_rec))

    return candidates


def scoop_and_retry(predict_fn, stuck_coords, atomic_nums, atomsymbols,
                     GADConfig, vib_eig, run_nr_minimization, NRConfig,
                     run_gad_saddle_search, max_steps=8000):
    """Re-minimize from stuck geometry, then try GAD from the minimum."""
    nr_cfg = NRConfig(n_steps=500, force_converged=0.01, min_interatomic_dist=0.3)
    nr_result = run_nr_minimization(predict_fn, stuck_coords, atomic_nums, atomsymbols, nr_cfg)
    steps_used = nr_result["total_steps"]

    if not nr_result["converged"]:
        return None, steps_used

    min_coords = nr_result["final_coords"]
    out_min = predict_fn(min_coords, atomic_nums, do_hessian=True)
    evals_min, evecs_min, _ = vib_eig(out_min["hessian"], min_coords, atomsymbols)

    remaining = max_steps - steps_used
    for evec_idx in range(min(3, evecs_min.shape[1])):
        v = evecs_min[:, evec_idx].reshape(-1, 3).to(min_coords.dtype)
        for scale in [0.3, -0.3, 0.5, -0.5, 0.7]:
            if remaining <= 500:
                return None, steps_used
            perturbed = min_coords + scale * v
            if _min_dist(perturbed) < 0.3:
                continue

            gad_steps = min(2000, remaining)
            gad_cfg = GADConfig(
                n_steps=gad_steps, dt=0.01, dt_max=0.3,
                max_atom_disp=0.5, min_interatomic_dist=0.3)
            result = run_gad_saddle_search(
                predict_fn, perturbed, atomic_nums, atomsymbols, gad_cfg)
            steps_used += result["total_steps"]
            remaining = max_steps - steps_used

            if result["converged"]:
                return result, steps_used

            # If this gets close (morse=1, low force), try fine
            if result["final_morse_index"] == 1 and result.get("final_force_norm", 999) < 0.5 and remaining > 500:
                fine_cfg = GADConfig(
                    n_steps=min(2000, remaining), dt=0.003, dt_max=0.1,
                    max_atom_disp=0.2, min_interatomic_dist=0.3)
                fine_result = run_gad_saddle_search(
                    predict_fn, result["final_coords"], atomic_nums, atomsymbols, fine_cfg)
                steps_used += fine_result["total_steps"]
                remaining = max_steps - steps_used
                if fine_result["converged"]:
                    return fine_result, steps_used

    return None, steps_used


def worker(seed, noise_std=0.3):
    """Single TS search with v6 strategy."""
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

    # ========== Phase 1: Multi-start NR ==========
    all_candidates = []
    total_nr_steps = 0

    nr_steps = 2000 if noise_std >= 0.4 else 1000
    nr_cfg = NRConfig(n_steps=nr_steps, force_converged=0.01, min_interatomic_dist=0.3)

    # Start A: full noise
    nr_result_a = run_nr_minimization(predict_fn, noisy_coords, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
    total_nr_steps += nr_result_a["total_steps"]
    cands_a = collect_candidates_from_nr(nr_result_a, predict_fn, vib_eig, "A_")

    # Start B: reduced noise (60%), only for high noise when A lacks nneg1
    cands_b = []
    n_nneg1_a = len([c for c in cands_a if "nneg1" in c[0]])
    if noise_std >= 0.4 and n_nneg1_a < 3:
        reduced_noise = ISOPROPANOL_COORDS + (noisy_coords - ISOPROPANOL_COORDS) * 0.6
        if _min_dist(reduced_noise) >= 0.3:
            nr_result_b = run_nr_minimization(predict_fn, reduced_noise, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
            total_nr_steps += nr_result_b["total_steps"]
            cands_b = collect_candidates_from_nr(nr_result_b, predict_fn, vib_eig, "B_")

    # Interleave A and B candidates (nneg1 first, then nneg2, etc.)
    def priority(item):
        label = item[0]
        p = 0
        if "nneg1" in label:
            if "last_" in label: p = 0
            elif "mid_" in label: p = 1
            elif "q" in label: p = 2
            else: p = 3
        elif "nneg2" in label: p = 10
        elif "min_ev0" in label: p = 20
        elif "min_ev" in label: p = 21
        elif "nneg3" in label: p = 30
        else: p = 40
        # Prefer A over B at same priority
        if label.startswith("B_"): p += 0.5
        return p

    all_candidates = sorted(cands_a + cands_b, key=priority)

    if not all_candidates:
        if nr_result_a["trajectory"]:
            all_candidates.append(("A_last_point", nr_result_a["trajectory"][-1]))
        else:
            return {
                "seed": seed, "status": "no_handoff",
                "reason": "no suitable handoff point found",
                "nr_steps": total_nr_steps, "wall": time.time() - t0,
            }

    # ========== Phase 2: GAD with candidate loop ==========
    best_result = None
    best_label = None
    best_step = None
    best_nneg = None
    total_gad_steps = 0

    for label, rec in all_candidates:
        gad_cfg = SOGADConfig(
            n_steps=5000, dt=0.01, dt_max=0.3,
            max_atom_disp=0.5, min_interatomic_dist=0.3)
        gad_result = run_gad_saddle_search(
            predict_fn, rec["coords"], ATOMIC_NUMS, ATOMSYMBOLS, gad_cfg)
        total_gad_steps += gad_result["total_steps"]

        if gad_result["converged"]:
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]
            break

        # Fine GAD if close (morse=1, force < 0.5)
        if gad_result["final_morse_index"] == 1 and gad_result.get("final_force_norm", 999) < 0.5:
            fine_cfg = SOGADConfig(
                n_steps=3000, dt=0.003, dt_max=0.1,
                max_atom_disp=0.2, min_interatomic_dist=0.3)
            fine_result = run_gad_saddle_search(
                predict_fn, gad_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS, fine_cfg)
            total_gad_steps += fine_result["total_steps"]
            if fine_result["converged"]:
                best_result = fine_result
                best_label = label + "_fine"
                best_step = rec["step"]
                best_nneg = rec["n_neg"]
                break
            gad_result = fine_result  # use the finer result for tracking

        # Scoop-and-retry if stuck at n_neg=2
        if gad_result["final_morse_index"] == 2 and gad_result.get("final_force_norm", 999) < 0.1:
            scoop_result, scoop_steps = scoop_and_retry(
                predict_fn, gad_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS,
                SOGADConfig, vib_eig, run_nr_minimization, NRConfig,
                run_gad_saddle_search, max_steps=8000)
            total_gad_steps += scoop_steps
            if scoop_result and scoop_result["converged"]:
                best_result = scoop_result
                best_label = label + "_scooped"
                best_step = rec["step"]
                best_nneg = rec["n_neg"]
                break

        # Track best
        if best_result is None or (
            gad_result["final_morse_index"] == 1 and
            gad_result.get("final_force_norm", 999) < best_result.get("final_force_norm", 999)
        ):
            best_result = gad_result
            best_label = label
            best_step = rec["step"]
            best_nneg = rec["n_neg"]

    # ========== Phase 3: Post-hoc rescue for best result ==========
    if best_result and not best_result["converged"]:
        # Rescue 1: morse=2 with low force -> scoop and retry
        if best_result["final_morse_index"] == 2 and best_result.get("final_force_norm", 999) < 0.1:
            scoop_result, scoop_steps = scoop_and_retry(
                predict_fn, best_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS,
                SOGADConfig, vib_eig, run_nr_minimization, NRConfig,
                run_gad_saddle_search, max_steps=10000)
            total_gad_steps += scoop_steps
            if scoop_result and scoop_result["converged"]:
                best_result = scoop_result
                best_label = best_label + "_posthoc_scoop"

        # Rescue 2: morse=1 with moderate force -> extended fine GAD
        if (not best_result["converged"] and
            best_result["final_morse_index"] == 1 and
            best_result.get("final_force_norm", 999) < 1.0):
            fine_cfg = SOGADConfig(
                n_steps=5000, dt=0.002, dt_max=0.05,
                max_atom_disp=0.15, min_interatomic_dist=0.3)
            fine_result = run_gad_saddle_search(
                predict_fn, best_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS, fine_cfg)
            total_gad_steps += fine_result["total_steps"]
            if fine_result["converged"]:
                best_result = fine_result
                best_label = best_label + "_posthoc_fine"

        # Rescue 3: wrong morse (3+) with low force -> re-minimize and try from scratch
        if (not best_result["converged"] and
            best_result["final_morse_index"] >= 3 and
            best_result.get("final_force_norm", 999) < 0.1):
            scoop_result, scoop_steps = scoop_and_retry(
                predict_fn, best_result["final_coords"], ATOMIC_NUMS, ATOMSYMBOLS,
                SOGADConfig, vib_eig, run_nr_minimization, NRConfig,
                run_gad_saddle_search, max_steps=10000)
            total_gad_steps += scoop_steps
            if scoop_result and scoop_result["converged"]:
                best_result = scoop_result
                best_label = best_label + "_posthoc_scoop3"

    wall = time.time() - t0
    gad_result = best_result

    result = {
        "seed": seed,
        "status": "ts" if gad_result["converged"] else "no_ts",
        "nr_steps": total_nr_steps,
        "handoff_step": best_step,
        "handoff_nneg": best_nneg,
        "handoff_label": best_label,
        "gad_steps": total_gad_steps,
        "gad_converged": gad_result["converged"],
        "final_energy": gad_result["final_energy"],
        "final_morse_idx": gad_result["final_morse_index"],
        "final_force": gad_result.get("final_force_norm", None),
        "n_candidates_tried": len(all_candidates),
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

    print(f"Parallel TS search v6: {args.n_seeds} seeds, {args.n_workers} workers, "
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
    out_path = os.path.join(out_dir, "parallel_results_v6.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
