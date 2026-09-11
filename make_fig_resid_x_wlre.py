#!/usr/bin/env python3
"""make_fig_resid_x_wlre.py — 잔차(z(FC_emp)−z(SC)) ⊙ wLRE 가 FC_emp 패턴과 닮는지.

스케일 차이(log1pm SC 가 FC 의 ~12%) 때문에 단순 뺄셈은 FC 사본이 됨 → SC>0 상삼각에서
z-score 로 맞춘 뒤 뺌. subject 별 [잔차 | wLRE | 잔차⊙wLRE | FC_emp], corr 은 SC>0 기준.
SC 는 log1pm(파이프라인 정규화), FC_emp 는 백업(_inputs_pre_final).

출력: figures/resid_x_wlre.png
실행: python3 make_fig_resid_x_wlre.py
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
        if ref is None:
            continue
        SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        w_max_mean = (SCr / SCr.max()).sum(1).mean()
        SC = np.log1p(SCr)
        SC *= w_max_mean / SC.sum(1).mean()
        FCe = np.loadtxt(f"{BAK}/{sub}/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        # SC>0 상삼각 통계로 z-score 후 뺄셈
        m = (SCr > 0)[IU]
        fv, sv = FCe[IU][m], SC[IU][m]
        R = (FCe - fv.mean()) / fv.std() - (SC - sv.mean()) / sv.std()
        wL = ref["wLRE"].copy()
        P = R * wL
        r_res = np.corrcoef(R[IU][m], FCe[IU][m])[0, 1]
        r_pos = np.corrcoef(P[IU][m], FCe[IU][m])[0, 1]
        for M in (R, P):
            M[SCr <= 0] = np.nan
            np.fill_diagonal(M, np.nan)
        wLm = wL.copy(); wLm[SCr <= 0] = np.nan; np.fill_diagonal(wLm, np.nan)
        rows.append((idx, sub, r_pos, r_res, R, wLm, P, FCe))
        print(f"idx {idx} ({sub}): corr(잔차,FC)={r_res:+.4f}  corr(잔차⊙wLRE,FC)={r_pos:+.4f}")

    vr = np.nanpercentile(np.abs(np.concatenate([r[4][IU] for r in rows])), 99)
    vw = np.nanpercentile(np.concatenate([r[5][IU] for r in rows]), 99)
    vp = np.nanpercentile(np.abs(np.concatenate([r[6][IU] for r in rows])), 99)
    vf = np.percentile(np.abs(np.concatenate([r[7][IU] for r in rows])), 99)
    COLS = [("잔차  z(FC_emp) − z(SC)", DIV, dict(norm=TwoSlopeNorm(0.0, -vr, vr))),
            ("wLRE (optimized)", SEQ, dict(vmin=0.0, vmax=vw)),
            ("잔차 ⊙ wLRE", DIV, dict(norm=TwoSlopeNorm(0.0, -vp, vp))),
            ("FC_emp", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf)))]

    fig, axes = plt.subplots(len(rows), 4, figsize=(16.5, 4.0 * len(rows) + 1.8),
                             constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for i, (idx, sub, r_pos, r_all, *Ms) in enumerate(rows):
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
                ax.text(0.5, -0.03, f"corr(P, FC_emp)={r_pos:+.3f} (SC>0)",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=INK2)
    for j, (title, cmap, kw) in enumerate(COLS):
        cb = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], location="bottom",
                          shrink=0.9, pad=0.006, aspect=28)
        cb.set_label(title, fontsize=10, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle("잔차(z(FC_emp) − z(SC)) ⊙ wLRE 와 FC_emp 패턴 비교\n"
                 "(z-score 는 SC>0 상삼각 기준 · 잔차/wLRE/곱: SC=0 엣지 회색 · 열별 공유 스케일 ±99%ile)",
                 fontsize=14, color=INK)
    out = "figures/resid_x_wlre.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
