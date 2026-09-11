#!/usr/bin/env python3
"""make_fig_param_scatter.py — wLRE·wFFI vs FC_emp 산점도 + 선형회귀 (idx 4~12, 10 제외).

정식 optim 가중치, SC>0 상삼각 엣지. subject 당 [wLRE|wFFI] 패널 한 쌍, 4×4 배치.
출력: figures/param_fc_scatter.png
실행: python3 make_fig_param_scatter.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from plot_wlre_fc import IU, MUTED, INK, INK2, SURFACE, GRID, SERIES
from fit_wlre_fc import SUBS
from make_fig_residual import BAK, load_ref_bak

GROUPS = {"a": [4, 5, 6, 7], "b": [8, 9, 11, 12]}
COL = {"wLRE": SERIES["wLRE"], "wFFI": SERIES["wFFI"]}


def draw(key, idxs):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), constrained_layout=True)
    fig.set_facecolor(SURFACE)
    for k, idx in enumerate(idxs):
        sub = SUBS[idx]
        ref, _, _ = load_ref_bak(sub)
        SCr = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"{BAK}/{sub}/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        m = (SCr > 0)[IU]
        x = FCe[IU][m]
        for j, nm in enumerate(("wLRE", "wFFI")):
            y = ref[nm][IU][m]
            if nm == "wFFI":
                # hinge 적합: y = max(0, a + b·FC) — 클립 포화를 모델에 포함 (전 엣지)
                b0, a0 = np.polyfit(x[y > 1e-6], y[y > 1e-6], 1)
                sol = least_squares(lambda p: np.maximum(0.0, p[0] + p[1] * x) - y,
                                    x0=[a0, b0], loss="linear")
                a, b = sol.x
                yh = np.maximum(0.0, a + b * x)
            else:
                b, a = np.polyfit(x, y, 1)
                yh = a + b * x
            r2 = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
            ax = axes[k // 2, (k % 2) * 2 + j]
            ax.scatter(x, y, s=2.5, alpha=0.12, color=COL[nm], edgecolors="none",
                       rasterized=True)
            xs = np.linspace(x.min(), x.max(), 200)
            ys = np.maximum(0.0, a + b * xs) if nm == "wFFI" else a + b * xs
            ax.plot(xs, ys, color=INK, linewidth=1.6)
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"idx {idx}  {nm}", fontsize=10.5, color=INK)
            ax.text(0.03, 0.03 if nm == "wLRE" else 0.97,
                    (f"y = max(0, {a:.2f} {b:+.2f}·FC)\nR² = {r2:.3f}  (hinge)"
                     if nm == "wFFI" else f"y = {a:.2f} {b:+.2f}·FC\nR² = {r2:.3f}"),
                    transform=ax.transAxes, ha="left",
                    va="bottom" if nm == "wLRE" else "top",
                    fontsize=8.5, color=INK2, linespacing=1.3)
            if k // 2 == 1:
                ax.set_xlabel("FC_emp", fontsize=9, color=INK2)
            if (k % 2) * 2 + j == 0:
                ax.set_ylabel("가중치", fontsize=9, color=INK2)
    fig.suptitle(f"정식 optim 가중치 vs FC_emp — 선형회귀 (SC>0 상삼각 엣지, 그룹 {key.upper()})\n"
                 "(파랑=wLRE, 주황=wFFI · 검은 선=OLS 적합)",
                 fontsize=13, color=INK)
    out = f"figures/param_fc_scatter_{key}.png"
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print("saved:", out)


def main():
    for key, idxs in GROUPS.items():
        draw(key, idxs)


if __name__ == "__main__":
    main()
