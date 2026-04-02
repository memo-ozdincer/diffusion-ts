"""Quick analysis of pairwise RMSD within energy bands."""
import json
import numpy as np

data = json.load(open("/scratch/memoozd/diffusion-ts/parallel_results.json"))
ts = [r for r in data if r.get("status") == "ts"]
rmsd = np.load("/scratch/memoozd/diffusion-ts/ts_rmsd_matrix.npy")

# Assign energy bands
bands = {}
for i, r in enumerate(ts):
    e = r["final_energy"]
    if e < -311:
        band = "A: E~-311.9"
    elif e < -310.3:
        band = "B: E~-310.36"
    elif e > -310.05 and e < -309.9:
        band = "C: E~-310.02"
    elif e < -310:
        band = "D: E~-310.0x outliers"
    elif e < -309:
        band = "E: E~-309.84"
    elif e < -308.35:
        band = "F: E~-308.37"
    elif e < -308.1:
        band = "G: E~-308.22"
    elif e < -307:
        band = "H: E~-307.09"
    else:
        band = "I: E~-306.56"

    bands.setdefault(band, []).append(i)

print("=" * 70)
print("PAIRWISE RMSD WITHIN ENERGY BANDS")
print("=" * 70)

for name in sorted(bands.keys()):
    indices = bands[name]
    seeds = [ts[i]["seed"] for i in indices]
    energies = [ts[i]["final_energy"] for i in indices]

    print(f"\n{name}: {len(indices)} member(s)")
    print(f"  Seeds: {seeds}")
    print(f"  Energies: {[f'{e:.4f}' for e in energies]}")

    if len(indices) < 2:
        continue

    # Pairwise RMSD
    rmsds = []
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            r = rmsd[indices[ii], indices[jj]]
            rmsds.append(r)
    rmsds_arr = np.array(rmsds)
    print(f"  RMSD stats: min={rmsds_arr.min():.4f} max={rmsds_arr.max():.4f} mean={rmsds_arr.mean():.4f}")

    # Flag large RMSDs that indicate geometric sub-clusters
    print("  Pairs with RMSD > 0.15 A:")
    count = 0
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            r = rmsd[indices[ii], indices[jj]]
            if r > 0.15:
                print(f"    seed {seeds[ii]}-{seeds[jj]}: {r:.4f}")
                count += 1
    if count == 0:
        print("    (none -- all within 0.15 A)")

# Also show: within cluster 1 vs 2 (both E~-310.36), are they really different TS types?
print("\n" + "=" * 70)
print("DETAILED: E~-310.36 band sub-structure")
print("=" * 70)
band_b = bands.get("B: E~-310.36", [])
if len(band_b) >= 2:
    seeds = [ts[i]["seed"] for i in band_b]
    print(f"Full RMSD matrix for {len(band_b)} members (seeds: {seeds}):")
    for ii in range(len(band_b)):
        row = []
        for jj in range(len(band_b)):
            if ii == jj:
                row.append("  ---  ")
            else:
                row.append(f"{rmsd[band_b[ii], band_b[jj]]:7.4f}")
        print(f"  seed {seeds[ii]:3d}: " + " ".join(row))

print("\n" + "=" * 70)
print("DETAILED: E~-308.37 band sub-structure")
print("=" * 70)
band_f = bands.get("F: E~-308.37", [])
if len(band_f) >= 2:
    seeds = [ts[i]["seed"] for i in band_f]
    print(f"Full RMSD matrix for {len(band_f)} members (seeds: {seeds}):")
    for ii in range(len(band_f)):
        row = []
        for jj in range(len(band_f)):
            if ii == jj:
                row.append("  ---  ")
            else:
                row.append(f"{rmsd[band_f[ii], band_f[jj]]:7.4f}")
        print(f"  seed {seeds[ii]:3d}: " + " ".join(row))
