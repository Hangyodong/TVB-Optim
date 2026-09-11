#!/usr/bin/env python3
"""formula_optim_vmap.py — formula_optim 의 CMA eval 을 개체군 단위 GPU vmap 배치로.

기존: 후보 1개 = 프로세스 1개 (BOLD 시뮬 + 별도 60s S_e 시뮬), 프로세스당 CPU 2코어.
여기: popsize(8)개 후보의 (c_ei, wLRE, wFFI)를 state pytree 의 leaf 3개만 배치 축으로
바꿔 jax.vmap 한 번의 GPU 호출로 평가 — 프로세스 1개·2코어로 GPU를 채운다.
S_e 는 별도 시뮬 대신 같은 시뮬 state trace 의 마지막 60s 평균 (측정창만 다름).

검증: --check 는 같은 θ 를 기존 경로(formula_optim 과 동일 함수)와 vmap 경로로 평가해
corr/rmse(≈1e-3)·S_e(측정창 차이로 ±0.02) 일치, eval 처리량, CPU 코어 사용량을 출력.
--sc-from D: FC 타깃은 --idx 것 그대로, SC(weight+tract_length)만 donor D 걸로 교체
  (a,b OLS·마스크·delay 전부 donor SC 기준 재계산 — SC 특이성 통제 실험용).
실행: python3 formula_optim_vmap.py --idx 4 [--check] [--chunk 8] [--sc-from 5]
결과: output_ppmi_pd/_fopt/idx<N>_sew2_vm[_sc<D>].{json,_sim.npz}
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import main_ppmi_pd as M
from analytic_fic import analytic_c_ei
from data_loader import load_data
from formula_optim import IU, OUT, X0
from model import build_network
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr

SE_WIN_MS = 60_000            # S_e 측정: state trace 마지막 60s (downsample 4ms)


def build_ctx(a_):
    """formula_optim [0]+[1] 과 동일: 데이터, a/b, warmup, base (model, state, monitor)."""
    p = M.prepare_pd_data(a_.idx, a_.noise_level)
    cfg = M.make_config(p, a_.idx, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_fopt"
    if a_.sc_from is not None and a_.sc_from != a_.idx:
        pd_ = M.prepare_pd_data(a_.sc_from, a_.noise_level)
        cfg.sc_csv, cfg.length_csv = pd_["sc_csv"], pd_["length_csv"]
        cfg.cache_version = f"{cfg.cache_version}_scfrom{a_.sc_from}"
        print(f"[SC-SWAP] SC/length ← idx{a_.sc_from} (sub {pd_['sub_num']}), "
              f"FC 타깃 = idx{a_.idx}", flush=True)
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    fc = np.asarray(data["fc_target"], np.float64)
    Wn = np.asarray(data["weights"], np.float64)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    m = mask.astype(bool)[IU]
    Wsc = Wn / max(float(Wn.max()), 1e-12)   # SC 보정(OLS·resid)용 max-norm [0,1] — b 스케일 프리
    b_ab, a_ab = np.polyfit(Wsc[IU][m], fc[IU][m], 1)
    se_target = float(cfg.fic_target_se)

    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    internal = capture_internal_state(initial_state)
    delay_hist = capture_network_delay_history(network)
    dur = int(a_.window_tr * cfg.bold_repetition_time_ms)
    skip = int(cfg.optimizer_bold_skip_tr)

    ctx = dict(cfg=cfg, data=data, fc=fc, Wn=Wn, mask=mask, cap=cap, n=n,
               a_ab=float(a_ab), b_ab=float(b_ab), resid=fc - (a_ab + b_ab * Wsc),
               se_target=se_target, network=network, internal=internal,
               delay_hist=delay_hist, dur=dur, skip=skip,
               warmup_result=warmup_result, bold_monitor=bold_monitor,
               t32=np.nan_to_num(np.asarray(fc, np.float32)))
    return ctx


def make_bundle(ctx, theta):
    """formula_optim.make_bundle 과 동일 (5-param: βL λL βF λF S_e*)."""
    bL, lL, bF, lF = theta[:4]
    se_star = float(theta[4]) if len(theta) > 4 else ctx["se_target"]
    wl = 1.0 + bL * ctx["fc"] + lL * ctx["resid"]
    wf = 1.0 - bF * ctx["fc"] - lF * ctx["resid"]
    ps = ParamSet(c_ei=np.ones(ctx["n"], np.float32),
                  wLRE=np.asarray(wl, np.float32), wFFI=np.asarray(wf, np.float32),
                  c_ei_frozen=True).sanitize(ctx["mask"], ctx["cap"])
    c_ei = analytic_c_ei(ctx["Wn"], np.asarray(ps.wLRE, np.float64),
                         np.asarray(ps.wFFI, np.float64), se_star)[0]
    c_ei = np.asarray(np.clip(c_ei, 0.1, 20.0), np.float32)
    return c_ei, np.asarray(ps.wLRE, np.float32), np.asarray(ps.wFFI, np.float32)


def make_batch_eval(ctx, a_):
    """(c_ei[K,N], wLRE[K,N,N], wFFI[K,N,N]) → (corr, rmse, se, fc)[K] 단일 GPU 호출."""
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    cei0, wl0, wf0 = make_bundle(ctx, list(X0) + [ctx["se_target"]])
    bundle0 = StateBundle.from_warmup(
        warmup_result=ctx["warmup_result"], bold_monitor_template=ctx["bold_monitor"],
        initial_params=ParamSet(c_ei=cei0, wLRE=wl0, wFFI=wf0, c_ei_frozen=True),
        internal_state=ctx["internal"], delay_history=ctx["delay_hist"], stage="warmup")
    model, base_state = bundle0.to_tvb_state(
        ctx["network"], solver, t1=ctx["dur"], dt=ctx["cfg"].integration_dt_ms)
    monitor = bundle0.build_bold_monitor(ctx["cfg"])

    # state pytree 에서 param leaf 3개를 "동일 객체" 로 찾아 교체 (타입 무관, tree_at 불요)
    leaves, treedef = jax.tree_util.tree_flatten(base_state)
    tgt_ids = [id(base_state.dynamics.c_ei),
               id(base_state.coupling.coupling.wLRE),
               id(base_state.coupling.coupling.wFFI)]
    idxs = [next(i for i, l in enumerate(leaves) if id(l) == t) for t in tgt_ids]
    assert len(set(idxs)) == 3, "param leaf 식별 실패"

    tgt = jnp.asarray(ctx["t32"])
    iu0, iu1 = jnp.asarray(IU[0]), jnp.asarray(IU[1])
    skip = ctx["skip"]
    se_win = int(SE_WIN_MS / 4.0)          # state 모니터 downsample 4ms

    def run_one(cei, wl, wf):
        ls = list(leaves)
        ls[idxs[0]], ls[idxs[1]], ls[idxs[2]] = cei, wl, wf
        st = jax.tree_util.tree_unflatten(treedef, ls)
        sim = model(st)
        bold = monitor(sim)
        ys = bold.ys[:, 0, :] if bold.ys.ndim == 3 else bold.ys
        ts = jnp.nan_to_num(ys[skip:])
        ts = ts - ts.mean(0, keepdims=True)
        tsn = ts / jnp.maximum(ts.std(0, keepdims=True), 1e-6)
        fcm = (tsn.T @ tsn) / jnp.maximum(tsn.shape[0] - 1, 1)
        fcm = jnp.clip(fcm, -1.0, 1.0) * (1.0 - jnp.eye(tgt.shape[0]))
        corr = jnp.corrcoef(fcm.flatten(), tgt.flatten())[0, 1]      # == fc_corr
        rmse = jnp.sqrt(jnp.mean((fcm[iu0, iu1] - tgt[iu0, iu1]) ** 2))
        se = jnp.mean(sim.data[-se_win:, 0, :])
        return corr, rmse, se, fcm

    vmapped = jax.jit(jax.vmap(run_one))

    def batch_eval(cei_b, wl_b, wf_b):
        outs = []
        for s in range(0, len(cei_b), a_.chunk):
            sl = slice(s, s + a_.chunk)
            outs.append(vmapped(jnp.asarray(cei_b[sl]), jnp.asarray(wl_b[sl]),
                                jnp.asarray(wf_b[sl])))
        return [np.concatenate([np.asarray(o[k]) for o in outs]) for k in range(4)]

    return batch_eval


def reference_eval(ctx, a_, theta):
    """기존 formula_optim 경로 그대로 (검증 기준): 순차 시뮬 + 별도 S_e 시뮬."""
    cei, wl, wf = make_bundle(ctx, theta)
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    bundle = StateBundle.from_warmup(
        warmup_result=ctx["warmup_result"], bold_monitor_template=ctx["bold_monitor"],
        initial_params=ParamSet(c_ei=cei, wLRE=wl, wFFI=wf, c_ei_frozen=True),
        internal_state=ctx["internal"], delay_history=ctx["delay_hist"], stage="warmup")
    fc_sim = np.asarray(compute_simulated_fc(ctx["network"], bundle, ctx["cfg"],
                                             sim_duration_ms=ctx["dur"],
                                             skip_tr=ctx["skip"]), np.float32)
    c = float(fc_corr(fc_sim, ctx["t32"]))
    rmse = float(np.sqrt(np.mean((fc_sim[IU] - ctx["t32"][IU]) ** 2)))
    model, st = bundle.to_tvb_state(ctx["network"], solver, t1=a_.se_sim_ms,
                                    dt=ctx["cfg"].integration_dt_ms)
    out = np.asarray(model(st).data)
    se = float(out[a_.se_sim_ms // 3:, 0, :].mean())
    return c, rmse, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--window-tr", type=int, default=240)
    ap.add_argument("--maxfev", type=int, default=64)
    ap.add_argument("--se-w", type=float, default=2.0)
    ap.add_argument("--se-tol", type=float, default=0.03)
    ap.add_argument("--corr-w", type=float, default=0.7)
    ap.add_argument("--rmse-w", type=float, default=0.3)
    ap.add_argument("--se-sim-ms", type=int, default=60_000)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8, help="vmap 동시 후보 수 (OOM 시 축소)")
    ap.add_argument("--check", action="store_true", help="기존 경로와 일치·속도 검증만")
    ap.add_argument("--sc-from", type=int, default=None,
                    help="SC(weight+length)만 이 idx 걸로 교체 (FC 타깃은 --idx)")
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()

    ctx = build_ctx(a_)
    print(f"[0] a={ctx['a_ab']:+.4f} b={ctx['b_ab']:+.4f}  se_target={ctx['se_target']}"
          f"  popsize={a_.popsize} chunk={a_.chunk}", flush=True)
    batch_eval = make_batch_eval(ctx, a_)
    print(f"[1] warmup+빌드 {time.perf_counter()-t0:.0f}s", flush=True)

    se_target = ctx["se_target"]

    def thetas_to_batch(thetas):
        ps = [make_bundle(ctx, th) for th in thetas]
        return (np.stack([p[0] for p in ps]), np.stack([p[1] for p in ps]),
                np.stack([p[2] for p in ps]))

    def score_of(c, rmse, se):
        pen = a_.se_w * max(0.0, abs(se - se_target) - a_.se_tol) if a_.se_w > 0 else 0.0
        return a_.corr_w * c - a_.rmse_w * rmse - pen

    if a_.check:
        noise = ctx["internal"].get("noise_samples") if ctx["internal"] else None
        if noise is not None:
            print(f"[check] noise_samples shape={np.asarray(noise).shape}"
                  f"  {np.asarray(noise).nbytes/1e6:.0f}MB/후보", flush=True)
        rng = np.random.default_rng(0)
        thetas = [list(X0) + [se_target]] + \
                 [list(np.clip(X0 + rng.normal(0, 0.4, 4), 0, 8)) +
                  [float(np.clip(se_target + rng.normal(0, 0.02), 0.20, 0.32))]
                  for _ in range(a_.popsize - 1)]
        # vmap 경로 — 1회 컴파일 후 재호출 계측
        cei_b, wl_b, wf_b = thetas_to_batch(thetas)
        tw, tc = time.perf_counter(), time.process_time()
        cs, rs, ses, _ = batch_eval(cei_b, wl_b, wf_b)
        w1, c1 = time.perf_counter() - tw, time.process_time() - tc
        tw, tc = time.perf_counter(), time.process_time()
        cs, rs, ses, _ = batch_eval(cei_b, wl_b, wf_b)
        w2, c2 = time.perf_counter() - tw, time.process_time() - tc
        print(f"[check] vmap {a_.popsize}후보: 컴파일포함 {w1:.1f}s → 재호출 {w2:.1f}s "
              f"({w2/a_.popsize:.1f}s/eval)  CPU {c2/w2:.2f}코어", flush=True)
        # 기존 경로 2개 (컴파일 1회 후 2번째 계측)
        ref = []
        for k, th in enumerate(thetas[:2]):
            tw, tc = time.perf_counter(), time.process_time()
            ref.append(reference_eval(ctx, a_, th))
            w, c = time.perf_counter() - tw, time.process_time() - tc
            print(f"[check] 기존 경로 θ{k}: {w:.1f}s  CPU {c/w:.2f}코어", flush=True)
        for k in range(2):
            print(f"  θ{k}  기존 corr={ref[k][0]:+.4f} rmse={ref[k][1]:.4f} S_e={ref[k][2]:.3f}"
                  f"   vmap corr={cs[k]:+.4f} rmse={rs[k]:.4f} S_e={ses[k]:.3f}"
                  f"   Δcorr={cs[k]-ref[k][0]:+.5f} ΔS_e={ses[k]-ref[k][2]:+.4f}", flush=True)
        return

    # CMA — 세대 단위 배치 평가
    import cma
    x0 = list(X0) + [se_target]
    es = cma.CMAEvolutionStrategy(x0, 1.0, dict(
        bounds=[[0, 0, 0, 0, 0.20], [8, 8, 8, 8, 0.32]],
        CMA_stds=[0.4, 0.4, 0.4, 0.4, 0.02],
        popsize=a_.popsize, maxfevals=a_.maxfev, seed=1, verbose=-9))
    hist = []
    print(f"[2] CMA 시작 (maxfev={a_.maxfev}, popsize={a_.popsize})", flush=True)
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

    # [3] 최종: best θ 재평가 (popsize 로 패딩 — 재컴파일 회피) + 저장
    cei_b, wl_b, wf_b = thetas_to_batch([list(best_theta)] * a_.popsize)
    cs, rs, ses, fcs = batch_eval(cei_b, wl_b, wf_b)
    corr_fin, rmse_fin, se_fin = float(cs[0]), float(rs[0]), float(ses[0])
    print(f"[3] {a_.window_tr}TR corr={corr_fin:+.4f}  rmse={rmse_fin:.4f}  "
          f"S_e={se_fin:.3f}", flush=True)

    tag = f"idx{a_.idx}" + (f"_sew{a_.se_w:g}" if a_.se_w > 0 else "") + "_vm" \
        + (f"_sc{a_.sc_from}" if a_.sc_from is not None else "")
    np.savez(f"{OUT}/{tag}_sim.npz", fc_sim=np.asarray(fcs[0], np.float32),
             c_ei=cei_b[0], corr=corr_fin, fc_target=ctx["fc"], mean_se=se_fin,
             wLRE=wl_b[0], wFFI=wf_b[0], sc_mask=ctx["mask"].astype(np.uint8))
    res = dict(idx=a_.idx, sc_from=a_.sc_from, a=ctx["a_ab"], b=ctx["b_ab"], algo="cma_vmap",
               se_w=a_.se_w, se_tol=a_.se_tol, popsize=a_.popsize,
               theta=list(map(float, best_theta)), corr=corr_fin, mean_se=se_fin,
               score=hist[k][3], eib={}, n_evals=len(hist),
               c_ei_mean=float(cei_b[0].mean()),
               elapsed_s=round(time.perf_counter() - t0, 1), hist=hist)
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nRESULT idx={a_.idx}  corr={corr_fin:+.4f}  S_e={se_fin:.3f}  "
          f"총 {res['elapsed_s']/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
