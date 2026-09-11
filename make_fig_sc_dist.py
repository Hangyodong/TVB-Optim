#!/usr/bin/env python3
"""make_fig_sc_dist.py — 정규화(log1pm)된 SC 의 SC>0 엣지값 분포, subject 별 3×3.

log1pm: log1p(w) 를 "max" 정규화의 노드입력 평균에 재스케일 (data_loader 재현).
출력: figures/sc_log1pm_dist.png
실행: python3 make_fig_sc_dist.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wlre_fc import IU, MUTED, INK, INK2, SURFACE, GRID, SERIES
from fit_wlre_fc import SUBS

BLUE = SERIES["wLRE"]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for ax, (idx, sub) in zip(axes.flat, SUBS.items()):
        SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        w_max_mean = (SCr / SCr.max()).sum(1).mean()
        SC = np.log1p(SCr)
        SC *= w_max_mean / SC.sum(1).mean()
        v = SC[IU][(SCr > 0)[IU]]
        ax.hist(v, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"idx {idx}  ({sub})", fontsize=11, color=INK)
        ax.text(0.97, 0.95,
                f"n={len(v):,}\nmean={v.mean():.4f}\nmedian={np.median(v):.4f}\n"
                f"p99={np.percentile(v,99):.4f}\nmax={v.max():.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=INK2, linespacing=1.4)
    for ax in axes[-1]:
        ax.set_xlabel("SC (log1pm)", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("엣지 수", fontsize=9, color=INK2)
    fig.suptitle("정규화(log1pm) SC 엣지값 분포 — SC>0 상삼각 엣지만 (0 엣지 제외)",
                 fontsize=13, color=INK)
    out = "figures/sc_log1pm_dist.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
