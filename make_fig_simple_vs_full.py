#!/usr/bin/env python3
"""make_fig_simple_vs_full.py — idx4: simple optim(β/λ) vs 정식 optim(풀 파이프라인).

행1 = simple optim (formula_optim 최신 결과 npz: S_e 페널티판 우선, 현 final FC 타깃)
행2 = 정식 optim (delay3 grad 캐시 post_grad_fc_matrix, 구 백업 FC 타깃)
열 = [FC_emp(타깃) | FC_sim | c_ei 분포]. 타깃 FC 가 서로 달라 corr 은 각자 타깃 기준.

출력: figures/simple_vs_full_idx4.png
실행: python3 make_fig_simple_vs_full.py
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

from plot_wlre_fc import DIV, IU, MUTED, INK, INK2, SURFACE, GRID, SERIES, _get
from make_fig_residual import BAK

IDX, SUB = 4, "100878"


def simple_result():
    """formula_optim npz — S_e 페널티판(sew*) 우선, 없으면 무페널티판."""
    cands = sorted(glob.glob(f"output_ppmi_pd/_fopt/idx{IDX}_sew*_sim.npz")) \
        or [f"output_ppmi_pd/_fopt/idx{IDX}_sim.npz"]
    d = np.load(cands[-1])
    se = float(d["mean_se"]) if "mean_se" in d else float("nan")
    return (np.asarray(d["fc_sim"], np.float64), np.asarray(d["c_ei"], np.float64),
            float(d["corr"]), np.asarray(d["fc_target"], np.float64), se,
            os.path.basename(cands[-1]))


def full_result():
    """구 FC 해시 grad 캐시 → post_grad FC_sim, c_ei, 백업 FC 타깃."""
    FCb = np.loadtxt(f"{BAK}/{SUB}/FC.csv", delimiter=",")
    np.fill_diagonal(FCb, 0.0)
    hs = {hashlib.sha1(np.asarray(FCb[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc = None, -9
    for f in glob.glob(f"output_ppmi_pd/{SUB}/cache/*/grad_*.pkl"):
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest", "bench")):
            continue
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            bc, best = c, b
    md = _get(best, "metadata", {}) or {}
    return (np.asarray(md.get("post_grad_fc_matrix"), np.float64),
            np.asarray(_get(_get(best, "params"), "c_ei"), np.float64), bc, FCb)


def full_se():
    """정식 optim 파라미터의 상태 시뮬 실측 mean_S_e (full_se.py 산출물)."""
    f = f"output_ppmi_pd/_fopt/idx{IDX}_full_se.npz"
    return float(np.load(f)["mean_se"]) if os.path.exists(f) else float("nan")


def main():
    fs_s, ce_s, corr_s, fce_s, se_s, src = simple_result()
    fs_f, ce_f, corr_f, fce_f = full_result()
    for M_ in (fs_s, fs_f):
        np.fill_diagonal(M_, 0.0)
    print(f"simple: {src}  corr={corr_s:+.4f}  S_e={se_s:.3f}")

    v = np.percentile(np.abs(np.concatenate(
        [fs_s[IU], fs_f[IU], fce_s[IU], fce_f[IU]])), 99)
    bins = np.linspace(0, max(ce_s.max(), ce_f.max()) * 1.05, 40)
    rows = [("simple optim (β/λ + 해석해 c_ei)", fce_s, fs_s, ce_s, corr_s,
             f"현 final FC · mean_S_e={se_s:.3f}"),
            ("정식 optim (FIC+EIB+gradient)", fce_f, fs_f, ce_f, corr_f,
             f"구 백업 FC · mean_S_e={full_se():.3f}")]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    fig.set_facecolor(SURFACE)
    for i, (name, fce, fs, ce, corr, note) in enumerate(rows):
        for j, (Mx, title) in enumerate(((fce, "FC_emp (타깃)"), (fs, "FC_sim"))):
            ax = axes[i, j]
            im = ax.imshow(Mx, cmap=DIV, norm=TwoSlopeNorm(0.0, -v, v), interpolation="nearest")
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_xticks([]); ax.set_yticks([])
            ttl = f"{name}\n{title}" if j == 0 else f"{title}  (corr={corr:+.4f})\n{note}"
            ax.set_title(ttl, fontsize=10.5, color=INK)
            if j == 1:
                cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                cb.ax.tick_params(colors=MUTED, labelsize=8)
                cb.outline.set_edgecolor(GRID)

        ax = axes[i, 2]
        ax.hist(ce, bins=bins, color=SERIES["wLRE"], edgecolor=SURFACE, linewidth=0.3)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title("c_ei 분포 (163 노드)", fontsize=10.5, color=INK)
        ax.text(0.97, 0.95, f"mean={ce.mean():.3f}\nSD={ce.std():.3f}\n"
                            f"min={ce.min():.3f}\nmax={ce.max():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=INK2, linespacing=1.4)
        ax.set_xlabel("c_ei", fontsize=9, color=INK2)
        ax.set_ylabel("노드 수", fontsize=9, color=INK2)

    fig.suptitle(f"idx {IDX} ({SUB}) — simple optim vs 정식 optim: FC_emp · FC_sim · c_ei\n"
                 "(FC 행렬 공유 스케일 ±99%ile · c_ei 공유 bin · 타깃 FC 는 행별 상이)",
                 fontsize=13, color=INK)
    out = "figures/simple_vs_full_idx4.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("saved:", out, f"| simple corr={corr_s:+.4f}  full corr={corr_f:+.4f}")


if __name__ == "__main__":
    main()
