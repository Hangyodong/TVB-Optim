#!/usr/bin/env python3
"""m116 simple optim 결과: 대표 6마리 × (SC, sim FC, emp FC, optimized wLRE) matrix plot."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAG = "_ns01"
OUT = "output_mouse116/_fopt"
# 그룹별 best corr 4 + 차순위 2 (ns01 기준)
PICKS = [11, 26, 27, 21, 16, 32]  # MPTP_ctr×2, MPTP_kh×2, veh_ctr, veh_kh

fig, axes = plt.subplots(len(PICKS), 4, figsize=(16, 4 * len(PICKS)))
for r, idx in enumerate(PICKS):
    z = np.load(f"{OUT}/idx{idx}{TAG}_sim.npz")
    meta = json.load(open(f"{OUT}/idx{idx}{TAG}.json"))
    sc = np.loadtxt(f"data/Mouse_allen_116/csv/idx{idx}_weight.csv", delimiter=",")
    sc /= sc.max()
    panels = [
        (np.log1p(sc * 100), "viridis", None, "SC (log1p)"),
        (z["fc_sim"], "coolwarm", 0.8, "sim FC"),
        (z["fc_target"], "coolwarm", 0.8, "emp FC"),
        (z["wLRE"], "magma", None, "optimized wLRE"),
    ]
    for c, (M, cmap, vmax, name) in enumerate(panels):
        ax = axes[r, c]
        kw = dict(vmin=-vmax, vmax=vmax) if vmax else {}
        im = ax.imshow(M, cmap=cmap, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if r == 0:
            ax.set_title(name, fontsize=13)
        if c == 0:
            ax.set_ylabel(f"idx{idx} {meta['group']}\nsub-{meta['sub_num']}  "
                          f"r={meta['corr']:+.3f}", fontsize=10)
fig.suptitle(f"Mouse_allen_116 simple optim ({TAG.strip('_')})", fontsize=15, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = f"figures/m116_fopt_matrices{TAG}.png"
fig.savefig(out, dpi=130)
print(out)
