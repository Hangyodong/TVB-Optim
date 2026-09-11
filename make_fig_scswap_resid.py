#!/usr/bin/env python3
"""make_fig_scswap_resid.py — SC-swap 파라미터·잔차 분포.

fig_scswap_resid.png: 잔차 분포 violin + 그 런이 실제 쓴 λ_L 오버레이.
fig_scswap_param_dist.png: [wLRE | c_ei] 2-violin — donor 21종 분포 비교.
fig_scswap_coef.png: optim 계수 — FC 계수 β(L/F) | 잔차 계수 λ(L/F), donor 별.
잔차 = FC(idx4 고정) − (a + b·SC_donor), 각자 SC>0 상삼각, a/b 는 max-norm SC 재적합.
실행: python3 make_fig_scswap_resid.py
"""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from plot_wlre_fc import GRID, INK, INK2, IU, MUTED, SERIES, SURFACE

FOPT = "output_ppmi_pd/_fopt"
BASE = 4


def resid_of(donor):
    """런과 동일 경로: idx4 FC 타깃 + donor SC(log1pm→max-norm) → resid, SC>0 상삼각.
    a,b 는 여기서 재적합 — OLS 는 SC 재스케일에 불변이라 json 값과 같은 잔차."""
    p = M.prepare_pd_data(BASE, 0.02)
    cfg = M.make_config(p, BASE, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_fopt"
    if donor is not None:
        pd_ = M.prepare_pd_data(donor, 0.02)
        cfg.sc_csv, cfg.length_csv = pd_["sc_csv"], pd_["length_csv"]
        cfg.cache_version = f"{cfg.cache_version}_scfrom{donor}"
    data = load_data(cfg)
    fc = np.asarray(data["fc_target"], np.float64)
    np.fill_diagonal(fc, 0.0)
    Wsc = np.asarray(data["weights"], np.float64)
    Wsc = Wsc / max(float(Wsc.max()), 1e-12)
    m = np.asarray(data["sc_mask"], bool)[IU]
    b, a = np.polyfit(Wsc[IU][m], fc[IU][m], 1)
    return (fc - (a + b * Wsc))[IU][m]


def main():
    j0 = json.load(open(f"{FOPT}/idx{BASE}_sew2_vm.json"))
    runs = [(None, j0)]
    for f in sorted(glob.glob(f"{FOPT}/idx{BASE}_sew2_vm_sc*.json"),
                    key=lambda s: int(re.search(r"_sc(\d+)\.json$", s).group(1))):
        runs.append((int(re.search(r"_sc(\d+)\.json$", f).group(1)), json.load(open(f))))

    labels, data, lam, wl_data, ce_data = [], [], [], [], []
    for dnr, j in runs:
        labels.append("self" if dnr is None else str(dnr))
        data.append(resid_of(dnr))
        lam.append(j["theta"][1])          # λ_L — 잔차 항이 실제 쓰인 세기
        tag = f"idx{BASE}_sew2_vm" + ("" if dnr is None else f"_sc{dnr}")
        d = np.load(f"{FOPT}/{tag}_sim.npz")
        m = np.asarray(d["sc_mask"], bool)[IU]
        wl_data.append(np.asarray(d["wLRE"], np.float64)[IU][m])
        ce_data.append(np.asarray(d["c_ei"], np.float64))
        print(f"{labels[-1]:>5}  n={len(data[-1]):>5}  SD={data[-1].std():.4f}  "
              f"[{data[-1].min():+.2f},{data[-1].max():+.2f}]  λL={lam[-1]:.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(16, 5.2), facecolor=SURFACE, constrained_layout=True)
    vp = ax.violinplot(data, positions=np.arange(len(data)), widths=0.78,
                       showextrema=False)
    for k, b in enumerate(vp["bodies"]):
        b.set_facecolor(INK if k == 0 else SERIES["wLRE"])
        b.set_alpha(0.9 if k == 0 else 0.55)
        b.set_edgecolor("none")
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.scatter(np.arange(len(data)), [v.std() for v in data], s=14, color="#b32c2b",
               zorder=3, label="SD")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(data)), lam, "o--", color=SERIES["wFFI"], ms=4, lw=1.0,
             alpha=0.8, label="λL (사용 세기)")
    ax2.set_ylabel("λ_L", color=SERIES["wFFI"], fontsize=9)
    ax2.tick_params(colors=SERIES["wFFI"], labelsize=8)
    for s in ax2.spines.values():
        s.set_color(GRID)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("SC donor idx (검정=self)", color=INK2, fontsize=9)
    ax.set_ylabel("resid = FC − (a + b·SC)", color=INK2, fontsize=9)
    ax.set_title(f"SC-swap 잔차 분포 — idx{BASE} FC 고정, donor SC 별 (각자 SC>0 상삼각)\n"
                 "빨강 점=SD, 주황 점선=그 donor 런이 실제 잔차 항에 쓴 λ_L",
                 color=INK, fontsize=12)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    out = "figures/fig_scswap_resid.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", out)

    # ── 통합 3-violin: wLRE | c_ei | 잔차 ────────────────────────
    def violin_row(ax, dat, col, title, ylab, zero_line=False):
        vp = ax.violinplot(dat, positions=np.arange(len(dat)), widths=0.78,
                           showextrema=False)
        for k, b in enumerate(vp["bodies"]):
            b.set_facecolor(INK if k == 0 else col)
            b.set_alpha(0.9 if k == 0 else 0.55)
            b.set_edgecolor("none")
        ax.scatter(np.arange(len(dat)), [float(np.mean(v)) for v in dat],
                   s=11, color=INK, zorder=3)
        if zero_line:
            ax.axhline(0.0, color=MUTED, lw=1.0)
        ax.axhline(float(np.mean(dat[0])), color=INK, lw=1.0, ls="--", alpha=0.6)
        ax.set_xticks(np.arange(len(dat)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("SC donor idx (검정=self)", color=INK2, fontsize=9)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(19, 5.4), facecolor=SURFACE,
                             constrained_layout=True)
    violin_row(axes[0], wl_data, SERIES["wLRE"],
               "wLRE 분포 — 각자 SC>0 엣지 (배율·꼬리만 SC 종속)", "wLRE")
    violin_row(axes[1], ce_data, SERIES["wFFI"],
               "c_ei 분포 — 163 노드 (분포 통째로 이동)", "c_ei")
    fig.suptitle("SC-swap 파라미터 분포 — idx4 FC 고정, donor SC 21종 "
                 "(점선=self 평균, 점=각 평균)", color=INK, fontsize=13)
    out = "figures/fig_scswap_param_dist.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", out)

    # ── fig: optim 계수 2종 — FC 계수 β, 잔차 계수 λ (L/F) ───────
    th = {lab: j["theta"] for lab, (dnr, j) in zip(labels, runs)}
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(19, 5.0), facecolor=SURFACE,
                             constrained_layout=True)
    for ax, (i_l, i_f, name) in zip(axes, ((0, 2, "FC 계수 β"), (1, 3, "잔차 계수 λ"))):
        for off, i_, nm, col in ((-0.2, i_l, "LRE", SERIES["wLRE"]),
                                 (+0.2, i_f, "FFI", SERIES["wFFI"])):
            ys = [th[l][i_] for l in labels]
            ax.bar(xs + off, ys, width=0.38, color=col, label=nm)
            ax.axhline(ys[0], color=col, lw=1.1, ls="--", alpha=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("SC donor idx", color=INK2, fontsize=9)
        ax.set_title(f"{name} — donor 별 optim 값 (점선=self)", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        leg = ax.legend(fontsize=9, frameon=True)
        leg.get_frame().set_edgecolor(GRID); leg.get_frame().set_facecolor(SURFACE)
        for t in leg.get_texts():
            t.set_color(INK)
    fig.suptitle("SC-swap optim 계수 — wLRE=1+β·FC+λ·resid, wFFI=1−β·FC−λ·resid",
                 color=INK, fontsize=13)
    out = "figures/fig_scswap_coef.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
