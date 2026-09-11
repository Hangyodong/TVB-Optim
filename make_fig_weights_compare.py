#!/usr/bin/env python3
"""make_fig_weights_compare.py — idx 4~12(10 제외): 정식 vs simple optim 의 wLRE/wFFI 행렬.

정식 = delay3 grad 캐시 params. simple = formula_optim θ 로 재구성
(wLRE = max[0, 1+β_L·FC+λ_L·(FC−a−b·SC)], wFFI 거울상; clip[0,10]·SC마스크 = sanitize 동일).
행 = subject, 열 = [wLRE 정식 | wLRE simple | wFFI 정식 | wFFI simple]. SC=0 회색.

출력: figures/weights_compare.png
실행: python3 make_fig_weights_compare.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wlre_fc import SEQ, IU, MUTED, INK, INK2, SURFACE, GRID
from fit_wlre_fc import SUBS
from make_fig_residual import load_ref_bak

CAP = 10.0
IDXS = [4, 5, 6, 7, 8, 9, 11, 12]


def simple_weights(idx, sub):
    j = json.load(open(f"output_ppmi_pd/_fopt/idx{idx}_sew2.json"))
    th, a_ab, b_ab = j["theta"], j["a"], j["b"]
    bL, lL, bF, lF = th[:4]
    SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
    w_max_mean = (SCr / SCr.max()).sum(1).mean()
    Wn = np.log1p(SCr)
    Wn *= w_max_mean / Wn.sum(1).mean()
    FCe = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
    np.fill_diagonal(FCe, 0.0)
    resid = FCe - (a_ab + b_ab * Wn)
    mask = (SCr > 0)
    wl = np.clip(1.0 + bL * FCe + lL * resid, 0.0, CAP) * mask
    wf = np.clip(1.0 - bF * FCe - lF * resid, 0.0, CAP) * mask
    return wl, wf, mask


def main():
    rows = []
    for idx in IDXS:
        sub = SUBS[idx]
        ref, corr, _ = load_ref_bak(sub)
        wl_s, wf_s, mask = simple_weights(idx, sub)
        wl_f, wf_f = ref["wLRE"].copy(), ref["wFFI"].copy()
        Ms = []
        for M_ in (wl_f, wl_s, wf_f, wf_s):
            M_ = M_.copy()
            M_[~mask] = np.nan
            np.fill_diagonal(M_, np.nan)
            Ms.append(M_)
        m = mask[IU]
        r_l = np.corrcoef(wl_f[IU][m], wl_s[IU][m])[0, 1]
        r_f = np.corrcoef(wf_f[IU][m], wf_s[IU][m])[0, 1]
        rows.append((idx, sub, r_l, r_f, Ms))
        print(f"idx {idx}: corr(정식,simple) wLRE={r_l:+.4f}  wFFI={r_f:+.4f}")

    vw = np.nanpercentile(np.concatenate(
        [np.concatenate([M[IU] for M in r[4]]) for r in rows]), 99)
    COLS = ["wLRE — 정식 optim", "wLRE — simple optim",
            "wFFI — 정식 optim", "wFFI — simple optim"]

    fig, axes = plt.subplots(len(rows), 4, figsize=(16.5, 3.9 * len(rows) + 1.6),
                             constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for i, (idx, sub, r_l, r_f, Ms) in enumerate(rows):
        for j, (M_, title) in enumerate(zip(Ms, COLS)):
            ax = axes[i, j]
            ax.imshow(M_, cmap=SEQ, vmin=0.0, vmax=vw, interpolation="nearest")
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=12, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(f"idx {idx}  ({sub})", fontsize=12, color=INK)
            if j in (1, 3):
                r = r_l if j == 1 else r_f
                ax.text(0.5, -0.03, f"corr(정식, simple) = {r:+.3f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, color=INK2)
    cb = fig.colorbar(axes[0, 0].images[0], ax=axes, location="bottom",
                      shrink=0.5, pad=0.006, aspect=40)
    cb.set_label("가중치", fontsize=10, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=9)
    cb.outline.set_edgecolor(GRID)
    fig.suptitle("정식 optim vs simple optim — wLRE · wFFI 행렬 (idx 4~12, 10 제외)\n"
                 "(공유 순차 스케일 0~99%ile · SC=0 엣지 회색 · corr 은 SC>0 상삼각)",
                 fontsize=13, color=INK)
    out = "figures/weights_compare.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
