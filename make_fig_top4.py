#!/usr/bin/env python3
"""make_fig_top4.py — 정식 optim 없는 idx 중 simple corr top4 를 2×2 블록으로.

블록 1개 = [FC_emp | simple FC_sim | simple wLRE]. top4 는 실행 시점 _fopt 결과에서 재선정.
출력: figures/fopt_top4_noformal.png
실행: python3 make_fig_top4.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_fig_formal_vs_simple import FOPT, load_simple
from plot_wlre_fc import INK, INK2, SURFACE, _heat

FORMAL = set(range(4, 13))


def main():
    cand = []
    for f in glob.glob(f"{FOPT}/idx*_sew2.json"):
        j = json.load(open(f))
        if j["idx"] not in FORMAL:
            cand.append((j["corr"], j["idx"]))
    top4 = [i for _, i in sorted(cand, reverse=True)[:4]]
    print("top4:", top4)
    S = {i: load_simple(i) for i in top4}

    v_fc = float(np.percentile(np.abs(np.concatenate(
        [np.concatenate([s["fce"].ravel(), s["fcs"].ravel()]) for s in S.values()])), 99))
    wmax = max(float(s["wl"][s["scm"]].max()) for s in S.values())

    fig, axes = plt.subplots(2, 6, figsize=(28.5, 10.5), facecolor=SURFACE,
                             constrained_layout=True)
    for k, i in enumerate(top4):
        s, ax = S[i], axes[k // 2, (k % 2) * 3:(k % 2) * 3 + 3]
        _heat(ax[0], s["fce"], f"idx {i} ({s['sub']})   FC_emp", 0.0, v=v_fc)
        _heat(ax[1], s["fcs"], f"simple FC_sim   r={s['corr']:+.3f}", 0.0, v=v_fc)
        _heat(ax[2], s["wl"], "simple wLRE", None, s["scm"], vmin=0.0, vmax=wmax)
    fig.suptitle("정식 optim 미보유 idx — simple optim corr top 4", color=INK, fontsize=15)
    fig.text(0.5, -0.015,
             f"FC: 0 기준 발산, 스케일 ±{v_fc:.2f}(99%ile) 공유 · 전체 엣지.  "
             f"wLRE: 순차 램프(흰=0), [0, {wmax:.2f}] 공유, 회색=SC 엣지 없음.  "
             "r = 현 final FC_emp 기준.  검은 선 = cortex|subcortex 경계.",
             ha="center", color=INK2, fontsize=9)
    out = "figures/fopt_top4_noformal.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
