#!/usr/bin/env python3
"""fit_cei_transfer.py — 정식 optim(EIB 보정) c_ei 를 닫힌형 특징으로 회귀 → 전이 계수.

모델: c_ei_formal ≈ a0 + a1·c_ei_analytic + a2·SC_strength + a3·FC⁺_strength
  c_ei_analytic 은 그 subject 의 정식 wLRE/wFFI 로 계산 (se* = cfg.fic_target_se).
  표본 = 정식 optim 보유 8명(idx10 제외) × 163 노드. subject-LOO 로 전이 성능 검증.

출력: output_ppmi_pd/_fopt/cei_transfer.npz (coef, 특징 정의는 formula_optim 과 공유)
실행: python3 fit_cei_transfer.py
"""
import glob
import hashlib
import os
import pickle

import numpy as np

import main_ppmi_pd as M
from analytic_fic import analytic_c_ei
from data_loader import load_data
from make_fig_residual import BAK
from plot_wlre_fc import _get

SUBS = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025",
        9: "101070", 11: "101146", 12: "101174"}


def load_formal(sub):
    """구 백업 FC 해시 grad 캐시 중 corr 최대 → wLRE/wFFI/c_ei."""
    hs = {hashlib.sha1(np.asarray(np.loadtxt(f"{BAK}/{sub}/FC.csv",
                                             delimiter=",")[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest", "bench")):
            continue
        if not any(h in os.path.basename(os.path.dirname(f)) for h in hs):
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        b = _get(d, "bundle", d)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best = {k: np.asarray(_get(p, k), np.float64) for k in ("wLRE", "wFFI", "c_ei")}
            bc = c
    return best


def features(idx):
    """formula_optim 과 동일 정의: [c_ei_analytic, SC strength, FC⁺ strength]."""
    p = M.prepare_pd_data(idx, 0.02)
    cfg = M.make_config(p, idx, use_delay=True)
    data = load_data(cfg)
    fc = np.asarray(data["fc_target"], np.float64)
    np.fill_diagonal(fc, 0.0)
    Wn = np.asarray(data["weights"], np.float64)
    return Wn, fc, float(cfg.fic_target_se)


def main():
    X, y, gid = [], [], []
    for g, (idx, sub) in enumerate(SUBS.items()):
        ref = load_formal(sub)
        assert ref is not None, f"idx {idx}: 캐시 없음"
        Wn, fc, se_t = features(idx)
        cei_a = analytic_c_ei(Wn, ref["wLRE"], ref["wFFI"], se_t)[0]
        X.append(np.stack([np.ones(len(cei_a)), cei_a, Wn.sum(1),
                           np.clip(fc, 0, None).sum(1)], 1))
        y.append(ref["c_ei"])
        gid.append(np.full(len(cei_a), g))
    X, y, gid = np.concatenate(X), np.concatenate(y), np.concatenate(gid)

    def r2(yy, yh):
        return 1 - np.sum((yy - yh) ** 2) / np.sum((yy - yy.mean()) ** 2)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"pooled OLS  coef={np.round(coef, 4)}  R2={r2(y, X @ coef):.4f}")
    print(f"  기준선: analytic 그대로(c_ei_a) R2={r2(y, X[:, 1]):.4f}")

    for g, idx in enumerate(SUBS):
        tr, te = gid != g, gid == g
        c, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        print(f"  LOO idx{idx:>2}:  전이 R2={r2(y[te], X[te] @ c):+.4f}   "
              f"analytic R2={r2(y[te], X[te, 1]):+.4f}")

    np.savez("output_ppmi_pd/_fopt/cei_transfer.npz", coef=coef)
    print("saved: output_ppmi_pd/_fopt/cei_transfer.npz")


if __name__ == "__main__":
    main()
