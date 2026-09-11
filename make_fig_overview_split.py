#!/usr/bin/env python3
"""make_fig_overview_split.py — [SC(log1pm) | wLRE | wFFI | FC_emp | FC_sim] 5열,
idx 4~12(10 제외) 를 4명씩 두 장으로. 가중치·FC_sim = 정식 optim(grad 캐시).

출력: figures/overview5_a.png (idx 4,5,6,7) / figures/overview5_b.png (idx 8,9,11,12)
실행: python3 make_fig_overview_split.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_wlre_fc import DIV, SEQ, IU, MUTED, INK, INK2, SURFACE, GRID
from fit_wlre_fc import SUBS
from make_fig_residual import BAK, load_ref_bak

GROUPS = {"a": [4, 5, 6, 7], "b": [8, 9, 11, 12]}


def load_row(idx):
    sub = SUBS[idx]
    ref, corr, _ = load_ref_bak(sub)
    SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    w_max_mean = (SCr / SCr.max()).sum(1).mean()
    SC = np.log1p(SCr)
    SC *= w_max_mean / SC.sum(1).mean()
    FCe = np.loadtxt(f"{BAK}/{sub}/FC.csv", delimiter=",")
    FCs = ref["FC_sim"].copy()
    for M_ in (FCe, FCs):
        np.fill_diagonal(M_, 0.0)
    wL, wF = ref["wLRE"].copy(), ref["wFFI"].copy()
    for M_ in (wL, wF):
        M_[SCr <= 0] = np.nan
        np.fill_diagonal(M_, np.nan)
    return idx, sub, corr, SC, wL, wF, FCe, FCs


def main():
    all_rows = {k: [load_row(i) for i in idxs] for k, idxs in GROUPS.items()}
    flat = [r for rows in all_rows.values() for r in rows]
    # 두 장 공통 스케일 (비교 일관성)
    vs = np.percentile(np.concatenate([r[3][IU] for r in flat]), 99)
    dw = np.nanpercentile(np.abs(np.concatenate(
        [np.concatenate([r[4][IU], r[5][IU]]) for r in flat]) - 1.0), 99)
    vf = np.percentile(np.abs(np.concatenate(
        [np.concatenate([r[6][IU], r[7][IU]]) for r in flat])), 99)
    COLS = [("SC (log1pm)", SEQ, dict(vmin=0.0, vmax=vs)),
            ("wLRE (optimized)", DIV, dict(norm=TwoSlopeNorm(1.0, 1.0 - dw, 1.0 + dw))),
            ("wFFI (optimized)", DIV, dict(norm=TwoSlopeNorm(1.0, 1.0 - dw, 1.0 + dw))),
            ("FC_emp", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf))),
            ("FC_sim", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf)))]

    for key, rows in all_rows.items():
        fig, axes = plt.subplots(len(rows), 5, figsize=(20, 4.0 * len(rows) + 1.7),
                                 constrained_layout=True)
        fig.set_facecolor(SURFACE)
        for i, (idx, sub, corr, *Ms) in enumerate(rows):
            for j, (M_, (title, cmap, kw)) in enumerate(zip(Ms, COLS)):
                ax = axes[i, j]
                ax.imshow(M_, cmap=cmap, interpolation="nearest", **kw)
                ax.set_facecolor(SURFACE)
                for s in ax.spines.values():
                    s.set_color(GRID)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(title, fontsize=13, color=INK, pad=8)
                if j == 0:
                    ax.set_ylabel(f"idx {idx}  ({sub})", fontsize=12, color=INK)
                if j == 4:
                    ax.text(0.5, -0.03, f"corr(FC_sim, FC_emp)={corr:.3f}",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=9, color=INK2)
        for j, (title, cmap, kw) in enumerate(COLS):
            cb = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], location="bottom",
                              shrink=0.9, pad=0.008, aspect=26)
            cb.set_label(title, fontsize=10, color=INK2)
            cb.ax.tick_params(colors=MUTED, labelsize=9)
            cb.outline.set_edgecolor(GRID)
        fig.suptitle("정식 optim — SC(log1pm) · wLRE · wFFI · FC_emp · FC_sim"
                     f"  (그룹 {key.upper()})\n"
                     "(wLRE/wFFI: 발산 중심=1(초기값), >1 빨강·<1 파랑, SC=0 회색 · 두 장 공통 스케일 ±99%ile)",
                     fontsize=14, color=INK)
        out = f"figures/overview5_{key}.png"
        fig.savefig(out, dpi=110, facecolor=SURFACE)
        print("saved:", out)


if __name__ == "__main__":
    main()
