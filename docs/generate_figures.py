"""Generate clustering figures for the Step 2 LaTeX report."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

OUT = "/project/rrg-aspuru/memoozd/diffusion-ts/docs/figures"
DATA = "/scratch/memoozd/diffusion-ts/clustered_results_67ts.json"

with open(DATA) as f:
    data = json.load(f)

D = np.array(data["rmsd_matrix"])
n = D.shape[0]

# Extract per-TS info from cluster members
energies, eig0s, seeds = [], [], []
for cl in data["clusters"]:
    for m in cl["members"]:
        energies.append(m["energy"])
        eig0s.append(m["eig0"])
        seeds.append(m["seed"])
energies = np.array(energies)
eig0s = np.array(eig0s)

# Assign family labels by energy
def assign_family(e):
    if abs(e - (-310.36)) < 0.1: return "A"
    if abs(e - (-310.03)) < 0.15: return "B"
    if abs(e - (-308.37)) < 0.1: return "C"
    if abs(e - (-308.22)) < 0.15: return "D"
    if abs(e - (-311.91)) < 0.15: return "E"
    return "Other"

families = [assign_family(e) for e in energies]
fam_names = ["A", "B", "C", "D", "E", "Other"]
fam_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728", "E": "#9467bd", "Other": "#7f7f7f"}

# ── Figure 1: RMSD heatmap with dendrogram ──
D_condensed = squareform(D, checks=False)
Z = linkage(D_condensed, method="average")
labels = fcluster(Z, t=0.2, criterion="distance")

# Sort by cluster label then energy
order = np.lexsort((energies, labels))
D_sorted = D[np.ix_(order, order)]

fig, (ax_dend, ax_heat) = plt.subplots(1, 2, figsize=(12, 5),
    gridspec_kw={"width_ratios": [1, 3], "wspace": 0.02})

# Dendrogram
dn = dendrogram(Z, orientation="left", ax=ax_dend, color_threshold=0.2,
                above_threshold_color="#999", no_labels=True, count_sort=True)
ax_dend.axvline(x=0.2, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
ax_dend.set_xlabel("RMSD (A)")
ax_dend.set_title("Dendrogram")

# Heatmap
im = ax_heat.imshow(D_sorted, cmap="viridis", aspect="auto", vmin=0, vmax=1.5)
plt.colorbar(im, ax=ax_heat, label="Aligned RMSD (A)", shrink=0.8)
ax_heat.set_xlabel("TS index (sorted by cluster)")
ax_heat.set_ylabel("TS index (sorted by cluster)")
ax_heat.set_title(f"Pairwise RMSD Matrix ({n} TS)")

plt.tight_layout()
plt.savefig(f"{OUT}/rmsd_heatmap_dendrogram.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"{OUT}/rmsd_heatmap_dendrogram.png", bbox_inches="tight", dpi=150)
plt.close()
print("Figure 1: RMSD heatmap + dendrogram")

# ── Figure 2: MDS embedding colored by family ──
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
coords_2d = mds.fit_transform(D)

fig, ax = plt.subplots(figsize=(7, 5.5))
for fam in fam_names:
    mask = [f == fam for f in families]
    if not any(mask): continue
    idx = np.where(mask)[0]
    count = len(idx)
    ax.scatter(coords_2d[idx, 0], coords_2d[idx, 1],
               c=fam_colors[fam], label=f"Family {fam} (n={count})",
               s=50, alpha=0.8, edgecolors="k", linewidths=0.3)

ax.legend(loc="best", fontsize=8)
ax.set_xlabel("MDS dimension 1")
ax.set_ylabel("MDS dimension 2")
ax.set_title("TS Geometry Clustering (MDS of aligned RMSD)")
plt.tight_layout()
plt.savefig(f"{OUT}/mds_clusters.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"{OUT}/mds_clusters.png", bbox_inches="tight", dpi=150)
plt.close()
print("Figure 2: MDS embedding")

# ── Figure 3: Energy vs eigenvalue scatter ──
fig, ax = plt.subplots(figsize=(7, 5))
for fam in fam_names:
    mask = [f == fam for f in families]
    if not any(mask): continue
    idx = np.where(mask)[0]
    ax.scatter(energies[idx], eig0s[idx],
               c=fam_colors[fam], label=f"Family {fam}",
               s=50, alpha=0.8, edgecolors="k", linewidths=0.3)

ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Lowest eigenvalue (eV/A$^2$)")
ax.set_title("TS Energy vs Curvature")
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/energy_vs_eig0.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"{OUT}/energy_vs_eig0.png", bbox_inches="tight", dpi=150)
plt.close()
print("Figure 3: Energy vs eigenvalue")

# ── Figure 4: RMSD histogram with gap annotation ──
triu = D[np.triu_indices(n, k=1)]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(triu, bins=60, color="#1f77b4", alpha=0.7, edgecolor="k", linewidth=0.3)
ax.axvline(x=0.2, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.2 A)")
ax.axvspan(0.144, 0.256, alpha=0.15, color="red", label="Natural gap")
ax.set_xlabel("Aligned RMSD (A)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Pairwise Aligned RMSD")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/rmsd_histogram.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"{OUT}/rmsd_histogram.png", bbox_inches="tight", dpi=150)
plt.close()
print("Figure 4: RMSD histogram")

# ── Figure 5: Convergence improvement bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 0.3A
cats = ["Baseline", "Improved"]
vals = [50, 74]
colors = ["#ff7f0e", "#2ca02c"]
axes[0].bar(cats, vals, color=colors, edgecolor="k", linewidth=0.5)
axes[0].set_ylabel("TS found (out of 100)")
axes[0].set_title("0.3 A Noise")
axes[0].set_ylim(0, 100)
for i, v in enumerate(vals):
    axes[0].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")

# 0.5A
vals5 = [28, 35]
axes[1].bar(cats, vals5, color=colors, edgecolor="k", linewidth=0.5)
axes[1].set_ylabel("TS found (out of 100)")
axes[1].set_title("0.5 A Noise")
axes[1].set_ylim(0, 100)
for i, v in enumerate(vals5):
    axes[1].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")

plt.suptitle("Convergence Rate Improvement", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/convergence_improvement.pdf", bbox_inches="tight", dpi=150)
plt.savefig(f"{OUT}/convergence_improvement.png", bbox_inches="tight", dpi=150)
plt.close()
print("Figure 5: Convergence improvement")

print("\nAll figures saved to", OUT)
