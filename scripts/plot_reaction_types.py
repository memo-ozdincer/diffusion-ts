#!/usr/bin/env python
"""Classify and visualize 87 TS by reaction type (stretched bonds)."""

import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "/scratch/memoozd/diffusion-ts/structural_clustering_v3"

with open('/scratch/memoozd/diffusion-ts/adaptive_results.json') as f:
    data = json.load(f)
ts = [r for r in data if r['status'] == 'ts' and 'ts_coords' in r]
D = np.load('/scratch/memoozd/diffusion-ts/structural_clustering/rmsd_matrix_87.npy')
seeds = [r['seed'] for r in ts]
energies = np.array([r['final_energy'] for r in ts])
eig0s = np.array([r.get('ts_eig0', 0) for r in ts])

def classify_ts(coords):
    c = np.array(coords)
    bonds = {
        'C-C': max(np.linalg.norm(c[0]-c[1]), np.linalg.norm(c[0]-c[2])),
        'C-H': np.linalg.norm(c[0]-c[4]),
        'C-O': np.linalg.norm(c[0]-c[3]),
        'O-H': np.linalg.norm(c[3]-c[5]),
    }
    ref = {'C-C': 1.52, 'C-H': 1.09, 'C-O': 1.43, 'O-H': 0.97}
    stretched = []
    for name, d in bonds.items():
        if d / ref[name] > 1.3:
            stretched.append(name)
    if not stretched:
        return 'angular/torsional'
    return ' + '.join(sorted(stretched))

rxn_types = [classify_ts(r['ts_coords']) for r in ts]
rxn_counts = Counter(rxn_types)

simple_types = []
for rt in rxn_types:
    if rxn_counts[rt] >= 3:
        simple_types.append(rt)
    else:
        simple_types.append('other multi-bond')
simple_types = np.array(simple_types)
type_counts = Counter(simple_types)
type_order = [t for t, _ in type_counts.most_common()]

tsne = TSNE(n_components=2, metric="precomputed", perplexity=10,
            random_state=42, init="random", max_iter=3000)
tsne_xy = tsne.fit_transform(D)

type_colors = {
    'angular/torsional': '#1f77b4',
    'C-H': '#ff7f0e',
    'C-C': '#2ca02c',
    'C-C + C-H': '#d62728',
    'C-O': '#9467bd',
    'other multi-bond': '#7f7f7f',
}
MARKERS = {
    'angular/torsional': 'o',
    'C-H': 's',
    'C-C': '^',
    'C-C + C-H': 'D',
    'C-O': 'v',
    'other multi-bond': 'x',
}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

ax = axes[0, 0]
for rt in type_order:
    mask = simple_types == rt
    c = type_colors.get(rt, '#7f7f7f')
    m = MARKERS.get(rt, 'o')
    ax.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], c=c, marker=m,
              s=60, edgecolors='k', linewidths=0.3, alpha=0.85,
              label=f'{rt} ({mask.sum()})')
ax.set_title('t-SNE (p=10) colored by Reaction Type', fontsize=11)
ax.legend(fontsize=7, loc='best')
ax.set_xticks([]); ax.set_yticks([])

ax = axes[0, 1]
sc = ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=energies, cmap='RdYlBu_r',
               s=50, edgecolors='k', linewidths=0.3, alpha=0.85)
plt.colorbar(sc, ax=ax, shrink=0.8)
ax.set_title('t-SNE colored by Energy (eV)', fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[0, 2]
sc = ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=eig0s, cmap='RdYlBu_r',
               s=50, edgecolors='k', linewidths=0.3, alpha=0.85)
plt.colorbar(sc, ax=ax, shrink=0.8)
ax.set_title('t-SNE colored by eig0 (curvature)', fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1, 0]
data_list = [energies[simple_types == rt] for rt in type_order]
bp = ax.boxplot(data_list, labels=[f'{rt}\n(n={type_counts[rt]})' for rt in type_order],
               patch_artist=True, widths=0.6)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(type_colors.get(type_order[i], '#7f7f7f'))
    patch.set_alpha(0.6)
ax.set_ylabel('Energy (eV)')
ax.set_title('Energy Distribution by Reaction Type', fontsize=11)
ax.tick_params(axis='x', labelsize=7, rotation=30)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
data_list = [eig0s[simple_types == rt] for rt in type_order]
bp = ax.boxplot(data_list, labels=[f'{rt}\n(n={type_counts[rt]})' for rt in type_order],
               patch_artist=True, widths=0.6)
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(type_colors.get(type_order[i], '#7f7f7f'))
    patch.set_alpha(0.6)
ax.set_ylabel('eig0')
ax.set_title('Curvature Distribution by Reaction Type', fontsize=11)
ax.tick_params(axis='x', labelsize=7, rotation=30)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 2]
sizes = [type_counts[rt] for rt in type_order]
colors = [type_colors.get(rt, '#7f7f7f') for rt in type_order]
labels_pie = [f'{rt} ({type_counts[rt]})' for rt in type_order]
ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.0f%%',
       startangle=90, textprops={'fontsize': 8})
ax.set_title('Reaction Type Distribution (87 TS)', fontsize=11)

plt.suptitle('87 TS Classified by Reaction Type (bonds stretched > 30% from equilibrium)',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_reaction_types.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved 06_reaction_types.png')

# Summary
print("\nReaction types:")
for rt in type_order:
    es = energies[simple_types == rt]
    eigs = eig0s[simple_types == rt]
    print(f"  {rt:25s} n={type_counts[rt]:3d}  E={es.mean():.2f}+/-{es.std():.2f}  eig0={eigs.mean():.2f}")

# Save classification
classification = {
    "n_ts": len(ts),
    "n_types": len(type_counts),
    "types": {rt: {"count": int(c), "seeds": [seeds[i] for i in range(len(ts)) if simple_types[i] == rt]}
              for rt, c in type_counts.most_common()},
}
with open(f'{OUT_DIR}/reaction_types.json', 'w') as f:
    json.dump(classification, f, indent=2)
print(f"Saved reaction_types.json")
