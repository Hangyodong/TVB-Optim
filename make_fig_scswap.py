#!/usr/bin/env python3
"""make_fig_scswap.py — SC-swap 통제 실험 정리: idx4 FC 고정, donor SC 20종.

표: donor별 corr/rmse/S_e/b + self 대비 wLRE·c_ei 유사도(공통 SC 엣지).
fig1(fig_scswap_summary.png): corr | wLRE~self corr | c_ei~self corr | c_ei 분포.
fig2(fig_scswap_wlre.png): wLRE 행렬 — self + [최고, 최저, idx10(outlier), 중간] donor.
fig3(fig_scswap_corr.png): corr 분포 — donor 20종 정렬 lollipop + self 기준선.
(파라미터 분포 violin 은 make_fig_scswap_resid.py 의 fig_scswap_param_dist.png)
실행: python3 make_fig_scswap.py
"""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_wlre_fc import GRID, INK, INK2, IU, MUTED, SERIES, SURFACE, _heat

FOPT = "output_ppmi_pd/_fopt"
BASE = 4


def load(tag):
    j = json.load(open(f"{FOPT}/{tag}.json"))
    d = np.load(f"{FOPT}/{tag}_sim.npz")
    return j, d


def main():
    j0, d0 = load(f"idx{BASE}_sew2_vm")
    wl0, ce0 = np.asarray(d0["wLRE"], np.float64), np.asarray(d0["c_ei"], np.float64)
    m0 = np.asarray(d0["sc_mask"], bool)

    rows = []
    for f in sorted(glob.glob(f"{FOPT}/idx{BASE}_sew2_vm_sc*.json"),
                    key=lambda s: int(re.search(r"_sc(\d+)\.json$", s).group(1))):
        dnr = int(re.search(r"_sc(\d+)\.json$", f).group(1))
        j, d = load(f"idx{BASE}_sew2_vm_sc{dnr}")
        wl, ce = np.asarray(d["wLRE"], np.float64), np.asarray(d["c_ei"], np.float64)
        m = np.asarray(d["sc_mask"], bool)
        both = (m0 & m)[IU]                       # 공통 SC 엣지에서만 가중치 비교
        rows.append(dict(
            dnr=dnr, corr=j["corr"], rmse=float(np.sqrt(np.mean(
                (np.asarray(d["fc_sim"])[IU] - np.asarray(d["fc_target"])[IU]) ** 2))),
            se=j["mean_se"], b=j["b"], dens=float(m[IU].mean()),
            overlap=float((m0 & m)[IU].sum() / max((m0 | m)[IU].sum(), 1)),
            wl_r=float(np.corrcoef(wl0[IU][both], wl[IU][both])[0, 1]),
            ce_r=float(np.corrcoef(ce0, ce)[0, 1]),
            ce_mean=float(ce.mean()), ce_sd=float(ce.std()), wl=wl, ce=ce, mask=m))

    r0 = dict(corr=j0["corr"], rmse=float(np.sqrt(np.mean(
        (np.asarray(d0["fc_sim"])[IU] - np.asarray(d0["fc_target"])[IU]) ** 2))),
        se=j0["mean_se"], b=j0["b"], dens=float(m0[IU].mean()),
        ce_mean=float(ce0.mean()), ce_sd=float(ce0.std()))

    print(f"{'donor':>6} {'corr':>7} {'Δcorr':>7} {'rmse':>7} {'S_e':>6} {'b':>6} "
          f"{'밀도':>6} {'겹침':>6} {'wLRE~r':>7} {'c_ei~r':>7} {'c_ei μ':>7}")
    print(f"{'self':>6} {r0['corr']:>7.4f} {'—':>7} {r0['rmse']:>7.4f} {r0['se']:>6.3f} "
          f"{r0['b']:>6.2f} {r0['dens']:>6.3f} {'—':>6} {'—':>7} {'—':>7} {r0['ce_mean']:>7.3f}")
    for r in rows:
        print(f"{r['dnr']:>6} {r['corr']:>7.4f} {r['corr']-r0['corr']:>+7.4f} "
              f"{r['rmse']:>7.4f} {r['se']:>6.3f} {r['b']:>6.2f} {r['dens']:>6.3f} "
              f"{r['overlap']:>6.3f} {r['wl_r']:>7.4f} {r['ce_r']:>7.4f} {r['ce_mean']:>7.3f}")
    cs = [r["corr"] for r in rows]
    print(f"\nself {r0['corr']:.4f}  |  donor 평균 {np.mean(cs):.4f}±{np.std(cs):.4f} "
          f"[{min(cs):.4f},{max(cs):.4f}]  Δ={np.mean(cs)-r0['corr']:+.4f}")

    # ── fig1: 요약 4패널 ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(23, 4.8), facecolor=SURFACE,
                             constrained_layout=True)
    xs = np.arange(len(rows))
    labels = [str(r["dnr"]) for r in rows]
    panels = [("corr (FC 성능)", [r["corr"] for r in rows], r0["corr"], SERIES["wLRE"]),
              ("wLRE ~ self  corr (공통 엣지)", [r["wl_r"] for r in rows], None, SERIES["wLRE"]),
              ("c_ei ~ self  corr (163 노드)", [r["ce_r"] for r in rows], None, SERIES["wFFI"])]
    for ax, (title, ys, ref, col) in zip(axes[:3], panels):
        ax.bar(xs, ys, color=col, width=0.7)
        if ref is not None:
            ax.axhline(ref, color=INK, lw=1.4, ls="--")
            ax.text(len(rows) - 0.3, ref, f" self {ref:.3f}", color=INK, fontsize=9,
                    va="bottom", ha="right")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("SC donor idx", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        lo = min(ys + ([ref] if ref else [])); hi = max(ys + ([ref] if ref else []))
        pad = 0.06 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    ax = axes[3]
    ax.errorbar(xs, [r["ce_mean"] for r in rows], yerr=[r["ce_sd"] for r in rows],
                fmt="o", color=SERIES["wFFI"], ms=5, capsize=3, lw=1.2)
    ax.axhline(r0["ce_mean"], color=INK, lw=1.4, ls="--")
    ax.text(len(rows) - 0.3, r0["ce_mean"], f" self μ={r0['ce_mean']:.2f}", color=INK,
            fontsize=9, va="bottom", ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("SC donor idx", color=INK2, fontsize=9)
    ax.set_title("c_ei 분포 (노드 평균 ± SD)", color=INK, fontsize=11)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(f"SC-swap: idx{BASE} FC 고정, donor SC {len(rows)}종 — 성능·파라미터 민감도",
                 color=INK, fontsize=13)
    out = "figures/fig_scswap_summary.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)

    # ── fig2: wLRE 행렬 self + 4 donor + c_ei 분포 ───────────────
    srt = sorted(rows, key=lambda r: r["corr"])
    picks, names = [], []

    def add(r, nm):
        if r is not None and all(r["dnr"] != p["dnr"] for p in picks):
            picks.append(r); names.append(nm)

    add(srt[-1], "최고")
    r10 = next((r for r in rows if r["dnr"] == 10), None)
    add(srt[0], "최저·SC outlier" if srt[0] is r10 else "최저")
    add(r10, "idx10(SC outlier)")
    add(srt[len(srt) // 2], "중간")
    for r in srt[1:]:                      # 중복으로 빈 자리는 차하위부터 채움
        if len(picks) >= 4:
            break
        add(r, "차하위")
    wmax = max([float(wl0[m0].max())] + [float(r["wl"][r["mask"]].max()) for r in picks])
    fig, axes = plt.subplots(1, 6, figsize=(29, 5.0), facecolor=SURFACE,
                             constrained_layout=True)
    _heat(axes[0], wl0, f"self wLRE   corr={r0['corr']:+.3f}", None, m0, vmin=0, vmax=wmax)
    for ax, r, nm in zip(axes[1:5], picks, names):
        _heat(ax, r["wl"], f"donor {r['dnr']} ({nm})   corr={r['corr']:+.3f}\n"
              f"wLRE~self r={r['wl_r']:+.3f}", None, r["mask"], vmin=0, vmax=wmax)
    ax = axes[5]
    bins = np.linspace(0, max(ce0.max(), max(r["ce"].max() for r in picks)) * 1.03, 32)
    ax.hist(ce0, bins=bins, histtype="step", lw=2.2, color=INK, label="self")
    for r, col in zip(picks, ("#2a78d6", "#eb6834", "#b32c2b", "#898781")):
        ax.hist(r["ce"], bins=bins, histtype="step", lw=1.4, color=col,
                label=f"donor {r['dnr']}")
    ax.set_title("c_ei 분포 (163 노드)", color=INK, fontsize=11)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=8, frameon=True)
    leg.get_frame().set_edgecolor(GRID); leg.get_frame().set_facecolor(SURFACE)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle(f"SC-swap: wLRE 행렬 비교 (순차 램프 [0,{wmax:.2f}] 공유, 회색=SC 엣지 없음)",
                 color=INK, fontsize=13)
    out = "figures/fig_scswap_wlre.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)

    # ── fig3: corr 분포 (정렬 lollipop + self 기준선) ─────────────
    srt_all = sorted(rows, key=lambda r: r["corr"])
    ys = np.arange(len(srt_all))
    cs_ = [r["corr"] for r in srt_all]
    mu, sd = float(np.mean(cs_)), float(np.std(cs_))
    fig, ax = plt.subplots(figsize=(9.5, 7.2), facecolor=SURFACE, constrained_layout=True)
    ax.axvspan(mu - sd, mu + sd, color=GRID, alpha=0.45, zorder=0)
    ax.axvline(mu, color=MUTED, lw=1.2, zorder=1)
    ax.axvline(r0["corr"], color=INK, lw=1.6, ls="--", zorder=2)
    ax.text(r0["corr"], len(srt_all) - 0.2, f" self {r0['corr']:.3f}", color=INK,
            fontsize=9.5, va="top", ha="left")
    for y, r in zip(ys, srt_all):
        hot = r["overlap"] < 0.4          # 마스크 겹침 낮은 저품질 SC
        col = "#b32c2b" if hot else SERIES["wLRE"]
        ax.plot([min(cs_) - 0.01, r["corr"]], [y, y], color=col, lw=1.0, alpha=0.5)
        ax.plot([r["corr"]], [y], "o", ms=6.5, color=col)
        if hot:
            ax.text(r["corr"] - 0.003, y, f"겹침 {r['overlap']:.2f}  ", color="#b32c2b",
                    fontsize=8, va="center", ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"donor {r['dnr']}" for r in srt_all], fontsize=8.5)
    ax.set_xlabel("corr (FC_sim vs FC_emp, idx4 타깃)", color=INK2, fontsize=10)
    ax.set_title(f"SC-swap corr 분포 — donor {len(srt_all)}종 (정렬)\n"
                 f"donor 평균 {mu:.3f}±{sd:.3f} (회색 밴드) · self {r0['corr']:.3f} · "
                 f"빨강 = SC 마스크 겹침<0.4", color=INK, fontsize=11.5)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    out = "figures/fig_scswap_corr.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
