#!/usr/bin/env python3
"""make_fig_resid_sc01.py — subject 별 [SC(0~1) | FC_emp | 잔차 | wLRE] 한 fig.

SC 0~1 정규화: log1p(w)/max(log1p(w)). 잔차 = FC_emp − SC(0~1).
출력: figures/resid_sc01.png
실행: python3 make_fig_resid_sc01.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_wlre_fc import DIV, SEQ, IU, MUTED, INK, INK2, SURFACE, GRID
from fit_wlre_fc import SUBS
from make_fig_residual import BAK, load_ref_bak


def main():
    rows = []
    for idx, sub in SUBS.items():
        ref, corr, tag = load_ref_bak(sub)
        SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        SC = np.log1p(SCr)
        SC /= SC.max()
        FCe = np.loadtxt(f"{BAK}/{sub}/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        R = FCe - SC
        np.fill_diagonal(R, 0.0)
        m = (SCr > 0)[IU]
        r_fc = np.corrcoef(R[IU][m], FCe[IU][m])[0, 1]
        r_wl = np.corrcoef(R[IU][m], ref["wLRE"][IU][m])[0, 1]
        wL = ref["wLRE"].copy()
        wL[SCr <= 0] = np.nan
        np.fill_diagonal(wL, np.nan)
        rows.append((idx, sub, r_fc, r_wl, SC, FCe, R, wL))
        print(f"idx {idx} ({sub}): corr(R,FC)={r_fc:+.4f}  corr(R,wLRE)={r_wl:+.4f}")

    vs = np.percentile(np.concatenate([r[4][IU] for r in rows]), 99)
    vf = np.percentile(np.abs(np.concatenate([r[5][IU] for r in rows])), 99)
    vr = np.percentile(np.abs(np.concatenate([r[6][IU] for r in rows])), 99)
    vw = np.nanpercentile(np.concatenate([r[7][IU] for r in rows]), 99)
    COLS = [("SC (log1p/max, 0~1)", SEQ, dict(vmin=0.0, vmax=vs)),
            ("FC_emp", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf))),
            ("잔차  FC_emp − SC", DIV, dict(norm=TwoSlopeNorm(0.0, -vr, vr))),
            ("wLRE (optimized)", SEQ, dict(vmin=0.0, vmax=vw))]

    fig, axes = plt.subplots(len(rows), 4, figsize=(16.5, 4.0 * len(rows) + 1.8),
                             constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for i, (idx, sub, r_fc, r_wl, *Ms) in enumerate(rows):
        for j, (M, (title, cmap, kw)) in enumerate(zip(Ms, COLS)):
            ax = axes[i, j]
            ax.imshow(M, cmap=cmap, interpolation="nearest", **kw)
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=13, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(f"idx {idx}  ({sub})", fontsize=12, color=INK)
            if j == 2:
                ax.text(0.5, -0.03, f"corr(R,FC)={r_fc:+.3f}   corr(R,wLRE)={r_wl:+.3f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=INK2)
    for j, (title, cmap, kw) in enumerate(COLS):
        cb = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], location="bottom",
                          shrink=0.9, pad=0.006, aspect=28)
        cb.set_label(title, fontsize=10, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle("subject 별 SC(0~1) · FC_emp · 잔차(FC − SC) · wLRE\n"
                 "(wLRE: SC=0 엣지 회색 · 열별 공유 스케일 ±99%ile · corr 은 SC>0 상삼각)",
                 fontsize=14, color=INK)
    out = "figures/resid_sc01.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
