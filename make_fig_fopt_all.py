#!/usr/bin/env python3
"""make_fig_fopt_all.py — simple optim(β/λ) 9명 결과: [FC_emp | FC_sim | c_ei 분포].

corr(240TR)·720TR 재채점·mean_S_e·θ 주석 포함. 데이터: _fopt/idx*_sew2{_sim.npz,.json}.
출력: figures/fopt_all_results.png
실행: python3 make_fig_fopt_all.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_wlre_fc import DIV, IU, MUTED, INK, INK2, SURFACE, GRID, SERIES
from fit_wlre_fc import SUBS


def main():
    rows = []
    for idx, sub in SUBS.items():
        npz = f"output_ppmi_pd/_fopt/idx{idx}_sew2_sim.npz"
        js = f"output_ppmi_pd/_fopt/idx{idx}_sew2.json"
        if not (os.path.exists(npz) and os.path.exists(js)):
            print(f"idx {idx}: 결과 없음 → skip")
            continue
        d, j = np.load(npz), json.load(open(js))
        fce = np.asarray(d["fc_target"], np.float64)
        fcs = np.asarray(d["fc_sim"], np.float64)
        for M_ in (fce, fcs):
            np.fill_diagonal(M_, 0.0)
        rows.append((idx, sub, float(d["corr"]), j.get("corr_720"),
                     float(d["mean_se"]) if "mean_se" in d else float("nan"),
                     j["theta"], fce, fcs, np.asarray(d["c_ei"], np.float64)))

    vf = np.percentile(np.abs(np.concatenate(
        [np.concatenate([r[6][IU], r[7][IU]]) for r in rows])), 99)
    ce_max = max(r[8].max() for r in rows) * 1.05
    bins = np.linspace(0, ce_max, 36)

    fig, axes = plt.subplots(len(rows), 3, figsize=(14.5, 4.1 * len(rows) + 1.6),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    fig.set_facecolor(SURFACE)
    for i, (idx, sub, c240, c720, se, theta, fce, fcs, ce) in enumerate(rows):
        for j, (Mx, title) in enumerate(((fce, "FC_emp"), (fcs, "FC_sim"))):
            ax = axes[i, j]
            im = ax.imshow(Mx, cmap=DIV, norm=TwoSlopeNorm(0.0, -vf, vf),
                           interpolation="nearest")
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=13, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(f"idx {idx}  ({sub})", fontsize=12, color=INK)
            if j == 1:
                c7 = f"  /  720TR {c720:+.3f}" if c720 else ""
                rmse = float(np.sqrt(np.mean((fcs[IU] - fce[IU]) ** 2)))
                ax.text(0.5, -0.03, f"corr {c240:+.3f}{c7}   RMSE={rmse:.4f}   S_e={se:.3f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=INK2)
        ax = axes[i, 2]
        ax.hist(ce, bins=bins, color=SERIES["wLRE"], edgecolor=SURFACE, linewidth=0.3)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_title("c_ei 분포 (해석해)", fontsize=13, color=INK, pad=8)
        th = " ".join(f"{v:.2f}" for v in theta)
        ax.text(0.97, 0.95, f"mean={ce.mean():.2f}\nSD={ce.std():.2f}\nθ=[{th}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=INK2, linespacing=1.4)
        if i == len(rows) - 1:
            ax.set_xlabel("c_ei", fontsize=9, color=INK2)
    fig.suptitle("simple optim (β/λ + 해석해 c_ei, CMA-ES) — subject 별 결과\n"
                 "(FC 공유 스케일 ±99%ile · c_ei 공유 bin · corr 은 240TR / 720TR 재채점)",
                 fontsize=13, color=INK)
    out = "figures/fopt_all_results.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
