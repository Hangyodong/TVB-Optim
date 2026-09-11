#!/usr/bin/env python3
"""make_fig_formal_vs_simple.py — 정식 optim(FIC+EIB+grad) vs simple optim(β/λ) 행렬 비교.

figA 2장: idx 4,5,6,7 / 8,9,11,12 (idx10 = SC 밀도 이상치 제외)
  열 = [FC_emp | 정식 FC_sim | 정식 wLRE | simple FC_sim | simple wLRE]
figB 1장: 정식 optim 없는 idx 중 simple corr top5 — 열 = [FC_emp | simple FC_sim | simple wLRE]
데이터: 정식 = grad 캐시(make_fig_residual.load_ref_bak — 구 백업 FC 타깃 해시, corr 최대),
       simple = _fopt npz + θ 로 wLRE 재구성 (formula_optim.make_bundle 과 동일 경로).
정식 optim 은 구 백업 FC 로 적합된 결과라 r 을 두 개 표기: r현=현 final FC 기준, r구=자기 타깃 기준.
출력: figures/formal_vs_simple_1.png, formal_vs_simple_2.png, fopt_top5_noformal.png
실행: python3 make_fig_formal_vs_simple.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from pipeline_contracts import ParamSet
from make_fig_residual import load_ref_bak
from plot_wlre_fc import GRID, INK, INK2, IU, MUTED, SURFACE, _heat

FOPT = "output_ppmi_pd/_fopt"
GROUPS = [[4, 5, 6, 7], [8, 9, 11, 12]]   # 정식 optim 보유 idx, idx10(SC outlier) 제외
TOP5 = [59, 50, 19, 82, 96]               # 정식 optim 없는 idx 중 simple corr top5


def load_simple(idx):
    """npz(FC_emp/FC_sim) + θ→wLRE 재구성 (sanitize 까지 formula_optim 과 동일)."""
    j = json.load(open(f"{FOPT}/idx{idx}_sew2.json"))
    d = np.load(f"{FOPT}/idx{idx}_sew2_sim.npz")
    fce = np.asarray(d["fc_target"], np.float64)
    fcs = np.asarray(d["fc_sim"], np.float64)
    for M_ in (fce, fcs):
        np.fill_diagonal(M_, 0.0)
    p = M.prepare_pd_data(idx, 0.02)
    cfg = M.make_config(p, idx, use_delay=True)
    data = load_data(cfg)
    fc = np.asarray(data["fc_target"], np.float64)
    np.fill_diagonal(fc, 0.0)
    Wn = np.asarray(data["weights"], np.float64)
    bL, lL, bF, lF = j["theta"][:4]
    resid = fc - (j["a"] + j["b"] * Wn)
    ps = ParamSet(c_ei=np.ones(data["n_nodes"], np.float32),
                  wLRE=np.asarray(1.0 + bL * fc + lL * resid, np.float32),
                  wFFI=np.asarray(1.0 - bF * fc - lF * resid, np.float32),
                  c_ei_frozen=True).sanitize(data["sc_mask"], cfg.connectivity_weight_max)
    return dict(sub=str(p["sub_num"]), fce=fce, fcs=fcs, corr=float(d["corr"]),
                wl=np.asarray(ps.wLRE, np.float64),
                scm=np.asarray(data["sc_mask"], bool))


def main():
    # ---- 데이터 일괄 로드 → 전 figure 공유 스케일 ---------------------------
    simple = {i: load_simple(i) for i in sum(GROUPS, []) + TOP5}
    formal = {}
    for i in sum(GROUPS, []):
        ref, corr_bak, tag = load_ref_bak(simple[i]["sub"])
        assert ref is not None, f"idx {i}: 정식 optim 캐시 없음"
        fcs = ref["FC_sim"]
        np.fill_diagonal(fcs, 0.0)
        formal[i] = dict(fcs=fcs, wl=ref["wLRE"], corr_bak=corr_bak,
                         corr=float(np.corrcoef(fcs[IU], simple[i]["fce"][IU])[0, 1]))

    pool = [s["fce"][IU] for s in simple.values()] + \
           [s["fcs"][IU] for s in simple.values()] + [f["fcs"][IU] for f in formal.values()]
    v_fc = float(np.percentile(np.abs(np.concatenate(pool)), 99))
    wmax = max([float(s["wl"][s["scm"]].max()) for s in simple.values()] +
               [float(formal[i]["wl"][simple[i]["scm"]].max()) for i in formal])

    note = (f"FC 3종: 0 기준 발산, 스케일 ±{v_fc:.2f}(전 패널 |값| 99%ile) 공유 · 전체 엣지.  "
            f"wLRE: 순차 램프(흰=0), 스케일 [0, {wmax:.2f}] 공유, 회색=SC 엣지 없음.  "
            "r현 = 현 final FC_emp 기준, r구 = 정식 optim 자기 타깃(구 백업 FC) 기준.  "
            "검은 선 = cortex|subcortex 경계.")

    # ---- figA: 정식 vs simple (4행 × 5열) × 2 ------------------------------
    for g, idxs in enumerate(GROUPS, 1):
        fig, axes = plt.subplots(len(idxs), 5, figsize=(24, 4.9 * len(idxs) + 1.2),
                                 facecolor=SURFACE, constrained_layout=True)
        for r, i in enumerate(idxs):
            s, f = simple[i], formal[i]
            _heat(axes[r, 0], s["fce"], "FC_emp", 0.0, v=v_fc)
            _heat(axes[r, 1], f["fcs"],
                  f"정식 FC_sim   r현={f['corr']:+.3f} r구={f['corr_bak']:+.3f}", 0.0, v=v_fc)
            _heat(axes[r, 2], f["wl"], "정식 wLRE", None, s["scm"], vmin=0.0, vmax=wmax)
            _heat(axes[r, 3], s["fcs"], f"simple FC_sim   r={s['corr']:+.3f}", 0.0, v=v_fc)
            _heat(axes[r, 4], s["wl"], "simple wLRE", None, s["scm"], vmin=0.0, vmax=wmax)
            axes[r, 0].set_ylabel(f"idx {i}  ({s['sub']})", color=INK, fontsize=12)
        fig.suptitle("정식 optim (FIC+EIB+gradient)  vs  simple optim (β/λ + 해석해 c_ei)"
                     f"  —  {g}/2", color=INK, fontsize=15)
        fig.text(0.5, -0.012, note, ha="center", color=INK2, fontsize=9)
        out = f"figures/formal_vs_simple_{g}.png"
        fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print("saved:", out)

    # ---- figB: 정식 optim 없는 top5 (5행 × 3열) ----------------------------
    fig, axes = plt.subplots(len(TOP5), 3, figsize=(14.5, 4.9 * len(TOP5) + 1.2),
                             facecolor=SURFACE, constrained_layout=True)
    for r, i in enumerate(TOP5):
        s = simple[i]
        _heat(axes[r, 0], s["fce"], "FC_emp", 0.0, v=v_fc)
        _heat(axes[r, 1], s["fcs"], f"simple FC_sim   r={s['corr']:+.3f}", 0.0, v=v_fc)
        _heat(axes[r, 2], s["wl"], "simple wLRE", None, s["scm"], vmin=0.0, vmax=wmax)
        axes[r, 0].set_ylabel(f"idx {i}  ({s['sub']})", color=INK, fontsize=12)
    fig.suptitle("정식 optim 미보유 idx — simple optim corr top 5", color=INK, fontsize=15)
    fig.text(0.5, -0.012, note, ha="center", color=INK2, fontsize=9)
    out = "figures/fopt_top5_noformal.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
