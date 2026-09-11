#!/usr/bin/env python3
"""make_fig_wlre_dist.py — 최적화된 wLRE 의 SC>0 엣지값 분포, subject 별 3×3.

출력: figures/wlre_dist.png
실행: python3 make_fig_wlre_dist.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wlre_fc import IU, MUTED, INK, INK2, SURFACE, GRID, SERIES
from fit_wlre_fc import SUBS
from make_fig_residual import load_ref_bak

BLUE = SERIES["wLRE"]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for ax, (idx, sub) in zip(axes.flat, SUBS.items()):
        ref, corr, tag = load_ref_bak(sub)
        SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        v = ref["wLRE"][IU][(SCr > 0)[IU]]
        ax.hist(v, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(1.0, color=INK2, linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"idx {idx}  ({sub})", fontsize=11, color=INK)
        ax.text(0.97, 0.95,
                f"n={len(v):,}\nmean={v.mean():.3f}\nmedian={np.median(v):.3f}\n"
                f"min={v.min():.3f}\nmax={v.max():.3f}\n<1 비율={np.mean(v<1)*100:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=INK2, linespacing=1.4)
    for ax in axes[-1]:
        ax.set_xlabel("wLRE (optimized)", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("엣지 수", fontsize=9, color=INK2)
    fig.suptitle("최적화된 wLRE 엣지값 분포 — SC>0 상삼각 엣지만 (점선 = 초기값 1)",
                 fontsize=13, color=INK)
    out = "figures/wlre_dist.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
