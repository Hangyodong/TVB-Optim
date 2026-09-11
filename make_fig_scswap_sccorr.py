#!/usr/bin/env python3
"""make_fig_scswap_sccorr.py — SC 유사도 vs simple optim 성능.

x = corr(log1p(SC_donor), log1p(SC_self))  (상삼각 전체, 0 포함 — 토폴로지+가중치)
y = 그 donor SC 로 optim 한 corr / rmse.  self = (1.0, 자기 성능).
출력: figures/fig_scswap_sccorr.png (산점 2패널),
     figures/fig_scswap_bars.png (donor 별 [SC유사도 | optim corr | RMSE] 묶음 막대)
실행: python3 make_fig_scswap_sccorr.py
"""
import csv
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wlre_fc import GRID, INK, INK2, IU, MUTED, SERIES, SURFACE

FOPT = "output_ppmi_pd/_fopt"
BASE = 4


def main():
    sub_of = {int(r["idx"]): r["sub"] for r in csv.DictReader(open(f"{FOPT}/fopt_results.csv"))}
    sc_self = np.log1p(np.loadtxt(f"output_ppmi_pd/{sub_of[BASE]}/inputs/weight.csv",
                                  delimiter=","))[IU]

    j0 = json.load(open(f"{FOPT}/idx{BASE}_sew2_vm.json"))
    d0 = np.load(f"{FOPT}/idx{BASE}_sew2_vm_sim.npz")
    rmse0 = float(np.sqrt(np.mean((np.asarray(d0["fc_sim"])[IU]
                                   - np.asarray(d0["fc_target"])[IU]) ** 2)))

    pts = []
    for f in sorted(glob.glob(f"{FOPT}/idx{BASE}_sew2_vm_sc*.json"),
                    key=lambda s: int(re.search(r"_sc(\d+)\.json$", s).group(1))):
        dnr = int(re.search(r"_sc(\d+)\.json$", f).group(1))
        j = json.load(open(f))
        d = np.load(f.replace(".json", "_sim.npz"))
        rmse = float(np.sqrt(np.mean((np.asarray(d["fc_sim"])[IU]
                                      - np.asarray(d["fc_target"])[IU]) ** 2)))
        sc = np.log1p(np.loadtxt(f"output_ppmi_pd/{sub_of[dnr]}/inputs/weight.csv",
                                 delimiter=","))[IU]
        pts.append((dnr, float(np.corrcoef(sc, sc_self)[0, 1]), j["corr"], rmse))
        print(f"donor {dnr:>2}  SC~self r={pts[-1][1]:.3f}  corr={j['corr']:+.4f}  rmse={rmse:.4f}")

    xs = [p[1] for p in pts]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), facecolor=SURFACE,
                             constrained_layout=True)
    for ax, yi, y0, name, col in ((axes[0], 2, j0["corr"], "simple optim corr", SERIES["wLRE"]),
                                  (axes[1], 3, rmse0, "simple optim RMSE", SERIES["wFFI"])):
        ys = [p[yi] for p in pts]
        ax.scatter(xs, ys, s=48, color=col, alpha=0.85, zorder=3)
        for p in pts:
            ax.annotate(str(p[0]), (p[1], p[yi]), textcoords="offset points",
                        xytext=(5, 3), fontsize=7.5, color=INK2)
        ax.scatter([1.0], [y0], marker="*", s=240, color=INK, zorder=4, label="self")
        b, a = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), 1.0, 50)
        ax.plot(gx, a + b * gx, color=INK, lw=1.4, ls="--")
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.set_title(f"{name}  vs  SC 유사도   (donor r={r:+.2f})", color=INK, fontsize=11.5)
        ax.set_xlabel("corr(log1p SC_donor, log1p SC_self)  — 상삼각 전체", color=INK2, fontsize=9.5)
        ax.set_ylabel(name, color=INK2, fontsize=9.5)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        leg = ax.legend(fontsize=9, frameon=True)
        leg.get_frame().set_edgecolor(GRID); leg.get_frame().set_facecolor(SURFACE)
        for t in leg.get_texts():
            t.set_color(INK)
    fig.suptitle("SC-swap: donor SC 가 원본 SC 와 닮을수록 성능이 좋아지는가 (idx4 FC 고정)",
                 color=INK, fontsize=13)
    out = "figures/fig_scswap_sccorr.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", out)

    # ── 묶음 막대: [SC유사도 | optim corr | RMSE] × (self + donor 20) ──
    groups = [("self", 1.0, j0["corr"], rmse0)] + [(str(p[0]),) + p[1:] for p in pts]
    xg = np.arange(len(groups))
    series = [("SC 유사도 (vs 원본 SC)", 1, "#7a5fb5"),
              ("simple optim corr", 2, SERIES["wLRE"]),
              ("simple optim RMSE", 3, SERIES["wFFI"])]
    fig, ax = plt.subplots(figsize=(19, 5.6), facecolor=SURFACE, constrained_layout=True)
    for off, (nm, i_, col) in zip((-0.27, 0.0, 0.27), series):
        ax.bar(xg + off, [g[i_] for g in groups], width=0.25, color=col, label=nm)
    for i_, ls in ((1, ":"), (2, "--"), (3, "-.")):
        ax.axhline(groups[0][i_], color=series[i_ - 1][2], lw=1.1, ls=ls, alpha=0.8)
    ax.set_xticks(xg)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_xlabel("SC donor idx", color=INK2, fontsize=9.5)
    ax.set_ylabel("값 (전부 무차원, [0,1] 축 공유)", color=INK2, fontsize=9.5)
    ax.set_title("SC-swap 한눈 비교 — donor 별 [SC 유사도 | optim corr | RMSE]  "
                 "(수평선 = self 기준값)", color=INK, fontsize=12)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    leg = ax.legend(fontsize=9.5, frameon=True, ncols=3, loc="upper right")
    leg.get_frame().set_edgecolor(GRID); leg.get_frame().set_facecolor(SURFACE)
    for t in leg.get_texts():
        t.set_color(INK)
    out = "figures/fig_scswap_bars.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
