#!/usr/bin/env python3
"""twopoint_fic.py — subject 마다 연립방정식 2개로 아핀 계수를 풀고, 그 가중치 위에서 FIC.

계수는 idx 마다 따로 구한다 (고정값·LOO 아님):
    식1:  wLRE(e0) = a + b * FC(e0)      e0 = |FC| 가 가장 작은 엣지
    식2:  wLRE(ef) = a + b * FC(ef)      ef = FC 가 최대인 엣지
  → b = (w[ef] - w[e0]) / (FC[ef] - FC[e0]),   a = w[e0] - b*FC[e0]
  → wFFI 는 합=2 로 유도:  a_F = 2 - a,  b_F = -b   (part2_eib.py:578-579 의 보존량)

앵커 좌표(w[e0], w[ef])는 그 subject 의 기존 최적화 결과에서 읽는다. final .mat 으로
파이프라인을 돌리면 inputs/FC.csv 가 덮어써져 캐시 해시가 어긋나므로, 계수 산출은
_inputs_pre_final/ 백업 + 그 해시에 맞는 grad 캐시에서 한다. 가중치를 만들 때 쓰는
FC 는 파이프라인이 로드한 최신 fc_target 이다.

절차: 계수 → 가중치 → FIC(c_ei) → 정방향 시뮬 1회 → FC corr
실행: python3 twopoint_fic.py --idx 4
"""
import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part1_fic import run_fic
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from tvboptim.observations.observation import fc_corr

IU = np.triu_indices(163, 1)
# final .mat (PD 187) 기준 idx → subject. 구 idx10~13 이 -1 시프트된 결과.
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025",
          9: "101070", 10: "101124", 11: "101146", 12: "101174"}
BACKUP = "output_ppmi_pd/_inputs_pre_final"
OUT = "output_ppmi_pd/_2pt"


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def best_ref(sub, FC_old):
    """FC_old 해시와 일치하는 grad 캐시 중 post_grad_fc_corr 최대 → params."""
    hs = {hashlib.sha1(np.asarray(FC_old[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc, btag = None, -9.0, ""
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        d = os.path.basename(os.path.dirname(f))
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest", "bench")):
            continue
        if not any(h in d for h in hs):
            continue
        with open(f, "rb") as fh:
            o = pickle.load(fh)
        b = _get(o, "bundle", o)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best = {k: np.asarray(_get(p, k), np.float64) for k in ("wLRE", "wFFI")}
            bc, btag = c, d
    return best, bc, btag


def two_point_coefs(sub):
    """앵커 두 개로 연립방정식을 풀어 (a, b) 반환."""
    FC = np.loadtxt(f"{BACKUP}/{sub}/FC.csv", delimiter=",")
    W = np.loadtxt(f"{BACKUP}/{sub}/weight.csv", delimiter=",")
    np.fill_diagonal(FC, 0.0)
    ref, corr, tag = best_ref(sub, FC)
    if ref is None:
        raise SystemExit(f"[{sub}] 해시 일치 grad 캐시 없음 → 앵커를 못 읽는다")
    m = (W > 0)[IU]
    fc, wl = FC[IU][m], ref["wLRE"][IU][m]
    e0 = int(np.argmin(np.abs(fc)))          # FC 가 0 에 가장 가까운 엣지
    ef = int(np.argmax(fc))                  # FC 최대 엣지
    b = (wl[ef] - wl[e0]) / (fc[ef] - fc[e0])
    a = wl[e0] - b * fc[e0]
    return a, b, dict(corr=corr, tag=tag, fc0=fc[e0], w0=wl[e0], fcmax=fc[ef],
                      wmax_at_fcmax=wl[ef], wmax=float(wl.max()),
                      same_edge=bool(ef == int(np.argmax(wl))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--posthoc-ms", type=int, default=240_000)
    a_ = ap.parse_args()
    idx, sub = a_.idx, SUBDIR[a_.idx]
    os.makedirs(OUT, exist_ok=True)

    a, b, info = two_point_coefs(sub)
    print(f"[idx {idx} / {sub}] 연립방정식 앵커", flush=True)
    print(f"  식1  FC={info['fc0']:+.5f} → wLRE={info['w0']:.4f}")
    print(f"  식2  FC={info['fcmax']:+.5f} → wLRE={info['wmax_at_fcmax']:.4f}"
          f"   (그 subject 최대 wLRE={info['wmax']:.4f}, 같은 엣지={info['same_edge']})")
    print(f"  해   wLRE = {a:+.4f} {b:+.4f}*FC     wFFI = {2-a:+.4f} {-b:+.4f}*FC")
    print(f"  앵커 출처 캐시 {info['tag']} (post_grad corr {info['corr']:.4f})", flush=True)

    p = M.prepare_pd_data(idx, a_.noise_level)
    cfg = M.make_config(p, idx, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_2pt"
    cfg.fic_posthoc_duration_ms = a_.posthoc_ms
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    tgt, mask, cap, n = data["fc_target"], data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

    wl = a + b * tgt
    wf = (2.0 - a) - b * tgt
    ps = ParamSet(c_ei=np.ones(n, np.float32), wLRE=np.asarray(wl, np.float32),
                  wFFI=np.asarray(wf, np.float32), c_ei_frozen=False).sanitize(mask, cap)
    scm = mask.astype(bool)
    print(f"[가중치] wLRE {np.asarray(ps.wLRE)[scm].mean():.4f} "
          f"(0 인 엣지 {(np.asarray(ps.wLRE)[scm] <= 1e-6).mean()*100:.2f}%)   "
          f"wFFI {np.asarray(ps.wFFI)[scm].mean():.4f} "
          f"(0 {(np.asarray(ps.wFFI)[scm] <= 1e-6).mean()*100:.2f}%)", flush=True)

    bundle = StateBundle.from_warmup(
        warmup_result=warmup_result, bold_monitor_template=bold_monitor,
        initial_params=ps, internal_state=capture_internal_state(initial_state),
        delay_history=capture_network_delay_history(network), stage="warmup")

    print(f"\n[FIC] 2점 가중치 위에서 S_e → {cfg.fic_target_se} 탐색...", flush=True)
    bundle = run_fic(network=network, bundle_in=bundle, cfg=cfg, data=data)
    c = np.asarray(bundle.params.c_ei, np.float64)
    print(f"[FIC] c_ei mean={c.mean():.4f} sd={c.std():.4f} "
          f"range=[{c.min():.3f},{c.max():.3f}]", flush=True)

    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    fc_sim = np.asarray(compute_simulated_fc(network, bundle, cfg, sim_duration_ms=dur,
                                             skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
    t32 = np.nan_to_num(np.asarray(tgt, np.float32))
    corr = float(fc_corr(fc_sim, t32))
    rmse = float(np.sqrt(np.mean((fc_sim[IU] - t32[IU]) ** 2)))
    mt = scm[IU]
    corr_sc = float(np.corrcoef(fc_sim[IU][mt], t32[IU][mt])[0, 1])
    corr_no = float(np.corrcoef(fc_sim[IU][~mt], t32[IU][~mt])[0, 1])
    res = dict(idx=idx, sub=sub, a=float(a), b=float(b), corr=corr, rmse=rmse,
               corr_sc=corr_sc, corr_nosc=corr_no, c_ei_mean=float(c.mean()),
               full_pipeline_corr=info["corr"], anchor=info)
    with open(f"{OUT}/idx{idx}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nRESULT idx={idx} sub={sub} corr={corr:+.4f} (SC>0 {corr_sc:+.4f} / "
          f"SC==0 {corr_no:+.4f}) rmse={rmse:.4f}  |  full pipeline {info['corr']:.4f}"
          f"  차이 {corr - info['corr']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
