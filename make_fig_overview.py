#!/usr/bin/env python3
"""make_fig_overview.py — subject 별 [SC(log1pm) | wLRE | wFFI | FC_emp | FC_sim | SC−FC 잔차] 한 fig.

잔차 = SC(log1pm) − FC_emp (단순 뺄셈). SC 는 data_loader 의 log1pm 정규화 그대로 재현
(log1p(w) 를 "max" 노드입력 평균에 재스케일). FC_sim 은 grad 캐시의 post_grad_fc_matrix.
FC_emp 는 최적화가 본 백업 FC(_inputs_pre_final). wLRE/wFFI 는 SC=0 엣지 회색 마스크.

출력: figures/matrices_overview.png
실행: python3 make_fig_overview.py
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
            print(f"idx {idx} ({sub}): 캐시 없음 → skip")
            continue
        SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        # data_loader 의 log1pm 정규화 재현
        w_max_mean = (SCr / SCr.max()).sum(1).mean()
        SC = np.log1p(SCr)
        SC *= w_max_mean / SC.sum(1).mean()
        FCe = np.loadtxt(f"{BAK}/{sub}/FC.csv", delimiter=",")
        FCs = ref["FC_sim"].copy()
        for M in (FCe, FCs):
            np.fill_diagonal(M, 0.0)
        R = SC - FCe
        np.fill_diagonal(R, 0.0)
        m = (SCr > 0)[IU]
        wL, wF = ref["wLRE"].copy(), ref["wFFI"].copy()
        for M in (wL, wF):
            M[SCr <= 0] = np.nan
            np.fill_diagonal(M, np.nan)
        rows.append((idx, sub, corr, SC, wL, wF, FCe, FCs, R))
        print(f"idx {idx} ({sub}): corr={corr:.4f}  SC max={SC.max():.3f}  "
              f"잔차RMS(SC>0)={np.sqrt((R[IU][m]**2).mean()):.4f}")

    # 열별 공유 스케일. FC_emp/FC_sim 은 동일 발산 스케일, wLRE/wFFI 도 동일 순차 스케일.
    vs = np.percentile(np.concatenate([r[3][IU] for r in rows]), 99)
    vw = np.nanpercentile(np.concatenate([np.concatenate([r[4][IU], r[5][IU]]) for r in rows]), 99)
    vf = np.percentile(np.abs(np.concatenate([np.concatenate([r[6][IU], r[7][IU]]) for r in rows])), 99)
    vr = np.percentile(np.abs(np.concatenate([r[8][IU] for r in rows])), 99)
    COLS = [("SC (log1pm)", SEQ, dict(vmin=0.0, vmax=vs)),
            ("wLRE (optimized)", SEQ, dict(vmin=0.0, vmax=vw)),
            ("wFFI (optimized)", SEQ, dict(vmin=0.0, vmax=vw)),
            ("FC_emp", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf))),
            ("FC_sim", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf))),
            ("잔차  SC − FC_emp", DIV, dict(norm=TwoSlopeNorm(0.0, -vr, vr)))]

    fig, axes = plt.subplots(len(rows), 6, figsize=(24, 4.0 * len(rows) + 1.8),
                             constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for i, (idx, sub, corr, *Ms) in enumerate(rows):
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
            if j == 4:
                ax.text(0.5, -0.03, f"corr(FC_sim, FC_emp)={corr:.3f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=INK2)
    for j, (title, cmap, kw) in enumerate(COLS):
        cb = fig.colorbar(axes[0, j].images[0], ax=axes[:, j], location="bottom",
                          shrink=0.9, pad=0.006, aspect=28)
        cb.set_label(title, fontsize=10, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle("subject 별 SC(log1pm) · wLRE · wFFI · FC_emp · FC_sim · 잔차(SC − FC_emp)\n"
                 "(wLRE/wFFI: SC=0 엣지 회색 · FC_emp/FC_sim 동일 스케일 · 열별 공유 스케일 ±99%ile)",
                 fontsize=14, color=INK)
    out = "figures/matrices_overview.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
