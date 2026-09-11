#!/usr/bin/env python3
"""fopt_mouse.py — Mouse_allen_116 (MPTP 34마리, 116노드) simple optim (β/λ + 해석해 c_ei).

formula_optim_vmap 의 vmap 배치 CMA 경로를 mouse 데이터로. target mean S_e 기본 0.161.
선행: data/Mouse_allen_116/csv/ (mat → csv 변환, subjects.csv 포함)
실행: python3 fopt_mouse.py --idx 0 [--se-target 0.161]
결과: output_mouse116/_fopt/idx<N>.{json,_sim.npz}
"""
import argparse
import json
import os
import time

import numpy as np

import formula_optim_vmap as V
from config import Config
from data_loader import load_data
from formula_optim import X0
from model import build_network
from pipeline_contracts import (
    capture_internal_state, capture_network_delay_history,
)

N_NODES = 116
IU = np.triu_indices(N_NODES, 1)
V.IU = IU                                  # vmap 경로의 163노드 IU 교체
OUT = "output_mouse116/_fopt"
CSV_DIR = "data/Mouse_allen_116/csv"


def make_config(idx, se_target, noise):
    return Config(
        region_txt=f"{CSV_DIR}/roi_names.txt",
        sc_csv=f"{CSV_DIR}/idx{idx}_weight.csv",
        length_csv=f"{CSV_DIR}/idx{idx}_length.csv",
        fc_csv=f"{CSV_DIR}/idx{idx}_fc.csv",
        # σ는 cache_tag 에 안 들어감 → cache_version 에 박아 캐시 혼선 차단
        cache_version=f"v_m116_fopt_se{se_target:g}_ns{noise:g}",
        cache_run_label="m116",
        integration_dt_ms=1.0,
        warmup_duration_ms=720_000,
        bold_repetition_time_ms=1000.0,
        tract_conduction_speed=3.0,
        additive_noise_sigma=noise,
        fic_target_se=se_target,
        optimizer_bold_skip_tr=60,
        connectivity_weight_max=1.5,
        bold_hrf_duration_ms=32_000.0,      # mouse HRF (나머지 HRF 기본값이 이미 mouse)
    )


def build_ctx(a_):
    """formula_optim_vmap.build_ctx 의 mouse판 (PPMI prepare 대신 직접 Config)."""
    cfg = make_config(a_.idx, a_.se_target, a_.noise_level)
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    fc = np.asarray(data["fc_target"], np.float64)
    Wn = np.asarray(data["weights"], np.float64)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    assert n == N_NODES, n
    m = mask.astype(bool)[IU]
    Wsc = Wn / max(float(Wn.max()), 1e-12)
    b_ab, a_ab = np.polyfit(Wsc[IU][m], fc[IU][m], 1)
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    internal = capture_internal_state(initial_state)
    delay_hist = capture_network_delay_history(network)
    dur = int(a_.window_tr * cfg.bold_repetition_time_ms)
    return dict(cfg=cfg, data=data, fc=fc, Wn=Wn, mask=mask, cap=cap, n=n,
                a_ab=float(a_ab), b_ab=float(b_ab), resid=fc - (a_ab + b_ab * Wsc),
                se_target=float(cfg.fic_target_se), network=network, internal=internal,
                delay_hist=delay_hist, dur=dur, skip=int(cfg.optimizer_bold_skip_tr),
                warmup_result=warmup_result, bold_monitor=bold_monitor,
                t32=np.nan_to_num(np.asarray(fc, np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True, help="0..33 (subjects.csv 참조)")
    ap.add_argument("--se-target", type=float, default=0.161)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--window-tr", type=int, default=600)
    ap.add_argument("--maxfev", type=int, default=64)
    ap.add_argument("--se-w", type=float, default=2.0)
    ap.add_argument("--se-tol", type=float, default=0.03)
    ap.add_argument("--corr-w", type=float, default=0.7)
    ap.add_argument("--rmse-w", type=float, default=0.3)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--tag", default="", help="출력 파일명 접미사 (기존 결과 보존용)")
    ap.add_argument("--seed", type=int, default=1, help="CMA seed")
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()

    sub_meta = {}
    with open(f"{CSV_DIR}/subjects.csv") as f:
        for ln in f.readlines()[1:]:
            i, s, g = ln.strip().split(",")
            sub_meta[int(i)] = (int(s), g)
    sub_num, group = sub_meta[a_.idx]

    ctx = build_ctx(a_)
    se_target = ctx["se_target"]
    print(f"[0] idx{a_.idx} sub={sub_num} group={group}  a={ctx['a_ab']:+.4f} "
          f"b={ctx['b_ab']:+.4f}  se_target={se_target}", flush=True)
    batch_eval = V.make_batch_eval(ctx, a_)
    print(f"[1] warmup+빌드 {time.perf_counter()-t0:.0f}s", flush=True)

    def thetas_to_batch(thetas):
        ps = [V.make_bundle(ctx, th) for th in thetas]
        return (np.stack([p[0] for p in ps]), np.stack([p[1] for p in ps]),
                np.stack([p[2] for p in ps]))

    def score_of(c, rmse, se):
        pen = a_.se_w * max(0.0, abs(se - se_target) - a_.se_tol) if a_.se_w > 0 else 0.0
        return a_.corr_w * c - a_.rmse_w * rmse - pen

    import cma
    se_lo, se_hi = max(0.05, se_target - 0.06), se_target + 0.08
    x0 = list(X0) + [se_target]
    es = cma.CMAEvolutionStrategy(x0, 1.0, dict(
        bounds=[[0, 0, 0, 0, se_lo], [8, 8, 8, 8, se_hi]],
        CMA_stds=[0.4, 0.4, 0.4, 0.4, 0.02],
        popsize=a_.popsize, maxfevals=a_.maxfev, seed=a_.seed, verbose=-9))
    hist = []
    print(f"[2] CMA 시작 (maxfev={a_.maxfev}, popsize={a_.popsize}, "
          f"S_e* ∈ [{se_lo:.3f}, {se_hi:.3f}])", flush=True)
    while not es.stop():
        xs = es.ask()
        t = time.perf_counter()
        cei_b, wl_b, wf_b = thetas_to_batch(xs)
        cs, rs, ses, _ = batch_eval(cei_b, wl_b, wf_b)
        scores = [score_of(float(cs[k]), float(rs[k]), float(ses[k]))
                  for k in range(len(xs))]
        for k, x in enumerate(xs):
            hist.append((list(map(float, x)), float(cs[k]), float(ses[k]),
                         scores[k], float(rs[k])))
        es.tell(xs, [-s for s in scores])
        k = int(np.argmax(scores))
        print(f"  gen {es.countiter:>2} ({len(hist):>3}ev)  best corr={cs[k]:+.4f} "
              f"rmse={rs[k]:.4f} S_e={ses[k]:.3f} score={scores[k]:+.4f}  "
              f"({time.perf_counter()-t:.1f}s/{len(xs)}ev)", flush=True)

    k = int(np.argmax([h[3] for h in hist]))
    best_theta = np.array(hist[k][0])
    print(f"[2] 최적 θ={np.round(best_theta,3)}  corr={hist[k][1]:+.4f}  "
          f"S_e={hist[k][2]:.3f}  score={hist[k][3]:+.4f}  ({len(hist)} evals)", flush=True)

    cei_b, wl_b, wf_b = thetas_to_batch([list(best_theta)] * a_.popsize)
    cs, rs, ses, fcs = batch_eval(cei_b, wl_b, wf_b)
    corr_fin, rmse_fin, se_fin = float(cs[0]), float(rs[0]), float(ses[0])
    print(f"[3] {a_.window_tr}TR corr={corr_fin:+.4f}  rmse={rmse_fin:.4f}  "
          f"S_e={se_fin:.3f}", flush=True)

    tag = f"idx{a_.idx}{a_.tag}"
    np.savez(f"{OUT}/{tag}_sim.npz", fc_sim=np.asarray(fcs[0], np.float32),
             c_ei=cei_b[0], corr=corr_fin, fc_target=ctx["fc"], mean_se=se_fin,
             wLRE=wl_b[0], wFFI=wf_b[0], sc_mask=ctx["mask"].astype(np.uint8))
    res = dict(idx=a_.idx, sub_num=sub_num, group=group, se_target=se_target,
               a=ctx["a_ab"], b=ctx["b_ab"], algo="cma_vmap",
               se_w=a_.se_w, se_tol=a_.se_tol, popsize=a_.popsize,
               theta=list(map(float, best_theta)), corr=corr_fin, rmse=rmse_fin,
               mean_se=se_fin, score=hist[k][3], n_evals=len(hist),
               c_ei_mean=float(cei_b[0].mean()),
               elapsed_s=round(time.perf_counter() - t0, 1), hist=hist)
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nRESULT idx={a_.idx} ({group})  corr={corr_fin:+.4f}  S_e={se_fin:.3f}  "
          f"총 {res['elapsed_s']/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
