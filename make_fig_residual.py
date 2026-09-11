#!/usr/bin/env python3
"""make_fig_residual.py — subject 별 [FC_emp | wLRE(optimized) | 잔차] 행렬을 한 fig 에.

잔차 R = wLRE(optimized) − ŵ(공식). 공식은 S442 절편 고정형:
ŵ = 1 + b·FC_emp,  b = Σ FC·(w−1) / Σ FC²  (SC>0 상삼각 원점 OLS).
FC_emp / 캐시 해시는 최적화가 실제로 본 백업 FC(_inputs_pre_final) 기준 — 현 inputs/FC.csv 는
final .mat 교체로 해시가 달라 grad 캐시와 매칭되지 않는다.

출력: figures/wlre_residual_matrices.png
실행: python3 make_fig_residual.py
"""
import glob
import hashlib
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_wlre_fc import DIV, SEQ, IU, MUTED, INK, INK2, SURFACE, GRID, _get
from fit_wlre_fc import SUBS

BAK = "output_ppmi_pd/_inputs_pre_final"


def load_ref_bak(sub):
    """백업 FC 해시와 일치하는 grad 캐시 중 post_grad_fc_corr 최대."""
    hs = {hashlib.sha1(np.asarray(np.loadtxt(f"{BAK}/{sub}/FC.csv",
                                             delimiter=",")[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc, btag = None, -9.0, ""
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        dirn = os.path.basename(os.path.dirname(f))
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest")):
            continue
        if not any(h in dirn for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best = {k: np.asarray(_get(p, k), np.float64) for k in ("wLRE", "wFFI")}
            best["FC_sim"] = np.asarray(
                (_get(b, "metadata", {}) or {}).get("post_grad_fc_matrix"), np.float64)
            bc, btag = c, dirn
    return best, bc, btag


def main():
    panels = []
    for idx, sub in SUBS.items():
        ref, corr, tag = load_ref_bak(sub)
        if ref is None:
            print(f"idx {idx} ({sub}): 캐시 없음 → skip")
            continue
        w = ref["wLRE"]
        SC = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"{BAK}/{sub}/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        m = (SC > 0)[IU]
        x, y = FCe[IU][m], w[IU][m]
        b = float(np.sum(x * (y - 1.0)) / np.sum(x * x))
        R = w - (1.0 + b * FCe)
        W = w.copy()
        for M in (R, W):
            M[SC <= 0] = np.nan
            np.fill_diagonal(M, np.nan)
        rms = float(np.sqrt(np.nanmean(R[IU][m] ** 2)))
        panels.append((idx, sub, b, corr, rms, FCe, W, R))
        print(f"idx {idx} ({sub}): b={b:+.3f}  RMS잔차={rms:.3f}  corr={corr:.3f}")

    # 열별 공유 스케일 (pooled 99%ile): FC=발산, wLRE=순차(0 원점), 잔차=발산
    vf = np.nanpercentile(np.abs(np.concatenate([p[5][IU] for p in panels])), 99)
    vw = np.nanpercentile(np.concatenate([p[6][IU] for p in panels]), 99)
    vr = np.nanpercentile(np.abs(np.concatenate([p[7][IU] for p in panels])), 99)
    COLS = [("FC_emp", DIV, dict(norm=TwoSlopeNorm(0.0, -vf, vf))),
            ("wLRE (optimized)", SEQ, dict(vmin=0.0, vmax=vw)),
            ("잔차  wLRE − (1 + b·FC_emp)", DIV, dict(norm=TwoSlopeNorm(0.0, -vr, vr)))]

    fig, axes = plt.subplots(len(panels), 3, figsize=(12.5, 4.05 * len(panels) + 1.6),
                             constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for i, (idx, sub, b, corr, rms, FCe, W, R) in enumerate(panels):
        for j, (M, (title, cmap, kw)) in enumerate(zip((FCe, W, R), COLS)):
            ax = axes[i, j]
            ims = ax.imshow(M, cmap=cmap, interpolation="nearest", **kw)
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=12, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(f"idx {idx}  ({sub})", fontsize=11, color=INK)
            if j == 2:
                ax.text(0.5, -0.03, f"b={b:+.3f}   RMS={rms:.3f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=9, color=INK2)
    # 열별 colorbar (하단 가로)
    for j, (title, cmap, kw) in enumerate(COLS):
        im = axes[0, j].images[0]
        cb = fig.colorbar(im, ax=axes[:, j], location="bottom", shrink=0.9,
                          pad=0.008, aspect=30)
        cb.set_label(title, fontsize=10, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=9)
        cb.outline.set_edgecolor(GRID)
    fig.suptitle("subject 별 FC_emp · wLRE(optimized) · 잔차 — 절편고정 공식 ŵ = 1 + b·FC_emp\n"
                 "(잔차: 파랑 = 공식이 과대, 빨강 = 공식이 과소 · 회색 = SC=0 엣지 · 열별 공유 스케일 ±99%ile)",
                 fontsize=13, color=INK)
    out = "figures/wlre_residual_matrices.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("saved:", out)


if __name__ == "__main__":
    main()
