"""Test how far GAD can push force norm with original params."""
import torch
from src.core_algos.saddle_optimizer import (
    vib_eig, run_gad_saddle_search, GADConfig, run_nr_minimization, NRConfig,
)
from src.dependencies.scine_calculator import ScineSparrowCalculator
from src.dependencies.calculators import make_scine_predict_fn

calc = ScineSparrowCalculator(functional='DFTB0')
predict_fn = make_scine_predict_fn(calc)

ISOPROPANOL_COORDS = torch.tensor([
    [ 0.000,  0.000,  0.000], [-1.270,  0.760,  0.000], [ 1.270,  0.760,  0.000],
    [ 0.000, -0.930,  1.100], [ 0.000, -0.670, -0.870], [ 0.000, -0.400,  1.920],
    [-1.270,  1.410,  0.870], [-1.270,  1.410, -0.870], [-2.160,  0.120,  0.000],
    [ 1.270,  1.410,  0.870], [ 1.270,  1.410, -0.870], [ 2.160,  0.120,  0.000],
], dtype=torch.float32)
ATOMIC_NUMS = torch.tensor([6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])
ATOMSYMBOLS = ['C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']

torch.manual_seed(17)
noisy = ISOPROPANOL_COORDS + torch.randn_like(ISOPROPANOL_COORDS) * 0.3

# NR + initial GAD
nr_cfg = NRConfig(n_steps=500, force_converged=0.01)
nr_result = run_nr_minimization(predict_fn, noisy, ATOMIC_NUMS, ATOMSYMBOLS, nr_cfg)
nneg1 = [r for r in nr_result['trajectory'] if r['n_neg'] == 1 and 'coords' in r]
handoff = nneg1[-1]['coords']
gad_cfg = GADConfig(n_steps=3000, dt=0.01, dt_max=0.3, max_atom_disp=0.5)
gad_r = run_gad_saddle_search(predict_fn, handoff, ATOMIC_NUMS, ATOMSYMBOLS, gad_cfg)
print(f"Initial GAD: F={gad_r['final_force_norm']:.6f}, steps={gad_r['total_steps']}")

# Try progressively tighter thresholds
coords = gad_r['final_coords']
for thresh in [0.005, 0.003, 0.002, 0.001]:
    gad_refine = GADConfig(n_steps=5000, dt=0.01, dt_max=0.3, max_atom_disp=0.5)
    r = run_gad_saddle_search(predict_fn, coords, ATOMIC_NUMS, ATOMSYMBOLS,
                              gad_refine, force_converged=thresh)
    print(f"  GAD to F<{thresh}: F={r['final_force_norm']:.6f} conv={r['converged']} steps={r['total_steps']}")
    if r['converged']:
        coords = r['final_coords']
    else:
        # Find minimum force in trajectory
        min_rec = min(r['trajectory'], key=lambda x: x['force_norm'])
        print(f"    min F={min_rec['force_norm']:.6f} at step {min_rec['step']}")
        break
