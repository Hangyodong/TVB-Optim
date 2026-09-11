#!/usr/bin/env python3
"""formula_optim.py — β/λ 스칼라 최적화로 Part1(FIC)+Part2(EIB) 대체 실험.

모델 (edge 별 조각 선형):
    wLRE = max[0, 1 + β_L·FC + λ_L·(FC − a − b·SC) + q_L·FC²]   (q 항은 --quad)
    wFFI = max[0, 1 − β_F·FC − λ_F·(FC − a − b·SC) − q_F·FC²]
  a,b 는 그 subject 의 FC~SC OLS (SC = 파이프라인 log1pm 정규화, SC>0 상삼각).
  --cei-model NPZ: 해석해 c_ei 에 전이 회귀(fit_cei_transfer.py) 보정 적용.

목적함수:  corr(FC_sim, FC_emp) − se_w·max(0, |mean_S_e − target| − se_tol)
  S_e 페널티로 생리학적 작동점(기본 0.25) 이탈을 벌점. se_w=0 이면 corr 만.

절차: [0] a,b → [1] warmup → [2] CMA-ES(기본) 또는 Nelder-Mead → [3] 최종 검증.
  c_ei 는 매 평가 해석해(닫힌형). 시뮬 FIC 폴리시는 corr 을 훼손해 제거함
  (0.645→0.532 실측, S_e 작동점 파괴).

실행: python3 formula_optim.py --idx 4 [--se-w 2.0] [--algo cma]
"""
import argparse
import json
import os
import time

import numpy as np

import main_ppmi_pd as M
from analytic_fic import analytic_c_ei
from data_loader import load_data
from model import build_network
from part3_gradient import compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
from tvboptim.observations.observation import fc_corr

IU = np.triu_indices(163, 1)
OUT = "output_ppmi_pd/_fopt"
X0 = np.array([2.57, 0.32, 2.03, 0.27])   # 역산 집단 평균 (idx10 제외 8명) 초기값


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--window-tr", type=int, default=240)
    ap.add_argument("--maxfev", type=int, default=120)
    ap.add_argument("--algo", choices=("cma", "nm"), default="cma")
    ap.add_argument("--se-w", type=float, default=2.0, help="S_e 페널티 가중 (0=끔)")
    ap.add_argument("--se-tol", type=float, default=0.03)
    ap.add_argument("--opt-se", action="store_true", default=True,
                    help="해석해 타깃 S_e* 를 5번째 CMA 파라미터로 (0.20~0.32)")
    ap.add_argument("--no-opt-se", dest="opt_se", action="store_false")
    ap.add_argument("--eib-steps", type=int, default=0,
                    help=">0 이면 CMA 최적점에서 EIB-lite (c_ei 동결, 엣지 가중치만)")
    ap.add_argument("--corr-w", type=float, default=1.0,
                    help="score = corr_w·corr − rmse_w·RMSE − S_e페널티")
    ap.add_argument("--rmse-w", type=float, default=0.25,
                    help="score = corr − rmse_w·RMSE − S_e페널티 (Part3 의 a0.8/g0.2 비율)")
    ap.add_argument("--se-sim-ms", type=int, default=60_000)
    ap.add_argument("--quad", action="store_true",
                    help="wLRE/wFFI 에 FC² 항 추가 (θ 5→7개)")
    ap.add_argument("--cei-model", default=None,
                    help="cei_transfer.npz 경로 — 해석해 c_ei 에 전이 보정 적용")
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()

    # [0] 데이터 + a,b
    p = M.prepare_pd_data(a_.idx, a_.noise_level)
    cfg = M.make_config(p, a_.idx, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_fopt"
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    fc, Wn = np.asarray(data["fc_target"], np.float64), np.asarray(data["weights"], np.float64)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    m = mask.astype(bool)[IU]
    Wsc = Wn / max(float(Wn.max()), 1e-12)   # SC 보정(OLS·resid)용 max-norm [0,1] — b 스케일 프리
    b_ab, a_ab = np.polyfit(Wsc[IU][m], fc[IU][m], 1)
    resid = fc - (a_ab + b_ab * Wsc)
    se_target = float(cfg.fic_target_se)
    cei_coef = np.load(a_.cei_model)["coef"] if a_.cei_model else None
    x_sw, x_fs = Wn.sum(1), np.clip(fc, 0, None).sum(1)   # cei 전이 특징 (fit_cei_transfer 와 동일)
    print(f"[0] a={a_ab:+.4f} b={b_ab:+.4f} (n={m.sum()})  se_w={a_.se_w} target={se_target}"
          f"  quad={a_.quad}  cei_model={bool(a_.cei_model)}", flush=True)

    # [1] warmup
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
    internal = capture_internal_state(initial_state)
    delay_hist = capture_network_delay_history(network)
    dur = int(a_.window_tr * cfg.bold_repetition_time_ms)
    skip = int(cfg.optimizer_bold_skip_tr)
    t32 = np.nan_to_num(np.asarray(fc, np.float32))
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    print(f"[1] warmup 완료 {time.perf_counter()-t0:.0f}s  eval={a_.window_tr}TR", flush=True)

    def make_bundle(theta):
        bL, lL, bF, lF = theta[:4]
        qL, qF = (theta[4], theta[5]) if a_.quad else (0.0, 0.0)
        se_star = float(theta[-1]) if a_.opt_se else se_target
        wl = 1.0 + bL * fc + lL * resid + qL * fc * fc
        wf = 1.0 - bF * fc - lF * resid - qF * fc * fc
        ps = ParamSet(c_ei=np.ones(n, np.float32),
                      wLRE=np.asarray(wl, np.float32), wFFI=np.asarray(wf, np.float32),
                      c_ei_frozen=True).sanitize(mask, cap)
        c_ei = analytic_c_ei(Wn, np.asarray(ps.wLRE, np.float64),
                             np.asarray(ps.wFFI, np.float64), se_star)[0]
        if cei_coef is not None:
            c_ei = cei_coef[0] + cei_coef[1] * c_ei + cei_coef[2] * x_sw + cei_coef[3] * x_fs
        ps = ParamSet(c_ei=np.asarray(np.clip(c_ei, 0.1, 20.0), np.float32),
                      wLRE=ps.wLRE, wFFI=ps.wFFI, c_ei_frozen=True)
        return StateBundle.from_warmup(
            warmup_result=warmup_result, bold_monitor_template=bold_monitor,
            initial_params=ps, internal_state=internal, delay_history=delay_hist,
            stage="warmup")

    def measure_se(bundle):
        model, st = bundle.to_tvb_state(network, solver, t1=a_.se_sim_ms, dt=cfg.integration_dt_ms)
        out = np.asarray(model(st).data)
        return float(out[a_.se_sim_ms // 3:, 0, :].mean())   # 앞 1/3 과도 제외

    hist = []

    def objective(theta):
        theta = np.asarray(theta, np.float64)
        if np.any(theta[:4] < 0) or np.any(theta[:4] > 8):
            return 1.0
        t = time.perf_counter()
        bundle = make_bundle(theta)
        se = measure_se(bundle) if a_.se_w > 0 else float("nan")
        pen = a_.se_w * max(0.0, abs(se - se_target) - a_.se_tol) if a_.se_w > 0 else 0.0
        fc_sim = np.asarray(compute_simulated_fc(network, bundle, cfg,
                                                 sim_duration_ms=dur, skip_tr=skip), np.float32)
        c = float(fc_corr(fc_sim, t32))
        rmse = float(np.sqrt(np.mean((fc_sim[IU] - t32[IU]) ** 2)))
        score = a_.corr_w * c - a_.rmse_w * rmse - pen
        hist.append((list(map(float, theta)), c, se, score, rmse))
        star = f" S_e*={theta[-1]:.3f}" if a_.opt_se else ""
        qs = f" qL={theta[4]:.3f} qF={theta[5]:.3f}" if a_.quad else ""
        print(f"  eval{len(hist):>3}  βL={theta[0]:.3f} λL={theta[1]:.3f} βF={theta[2]:.3f} "
              f"λF={theta[3]:.3f}{qs}{star}  corr={c:+.4f}  rmse={rmse:.4f}  S_e={se:.3f}  score={score:+.4f}  "
              f"({time.perf_counter()-t:.1f}s)", flush=True)
        return -score

    # [2] 최적화
    print(f"[2] {a_.algo.upper()} 시작 (maxfev={a_.maxfev})", flush=True)
    x0, lo, hi, st = list(X0), [0.0] * 4, [8.0] * 4, [0.4] * 4
    if a_.quad:
        x0 += [0.0, 0.0]; lo += [-4.0, -4.0]; hi += [4.0, 4.0]; st += [0.2, 0.2]
    if a_.opt_se:
        x0 += [se_target]; lo += [0.20]; hi += [0.32]; st += [0.02]
    if a_.algo == "cma":
        import cma
        es = cma.CMAEvolutionStrategy(x0, 1.0, dict(
            bounds=[lo, hi], CMA_stds=st,
            popsize=8, maxfevals=a_.maxfev, seed=1, verbose=-9))
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [objective(x) for x in xs])
    else:
        from scipy.optimize import minimize
        minimize(objective, x0, method="Nelder-Mead",
                 options=dict(maxfev=a_.maxfev, xatol=0.02, fatol=5e-4))
    k = int(np.argmax([h[3] for h in hist]))
    best_theta, best_corr, best_se, best_score = np.array(hist[k][0]), hist[k][1], hist[k][2], hist[k][3]
    print(f"[2] 최적 θ={np.round(best_theta,3)}  corr={best_corr:+.4f}  S_e={best_se:.3f}  "
          f"score={best_score:+.4f}  ({len(hist)} evals, {time.perf_counter()-t0:.0f}s)", flush=True)

    # [3] 최종 검증 (재구성 재현 + c_ei 저장)
    bundle = make_bundle(best_theta)
    c_ei = np.asarray(bundle.params.c_ei, np.float64)
    fc_sim = np.asarray(compute_simulated_fc(network, bundle, cfg,
                                             sim_duration_ms=dur, skip_tr=skip), np.float32)
    corr_fin = float(fc_corr(fc_sim, t32))
    se_fin = measure_se(bundle)
    print(f"[3] {a_.window_tr}TR corr={corr_fin:+.4f}  S_e={se_fin:.3f}", flush=True)

    eib = {}
    if a_.eib_steps > 0:
        from part2_eib import run_eib
        cfg.eib_max_iterations = a_.eib_steps
        cfg.eib_posthoc_duration_ms = dur
        print(f"[3b] EIB-lite {a_.eib_steps} steps (c_ei 동결)...", flush=True)
        bundle_e = run_eib(network=network, bundle_in=make_bundle(best_theta), cfg=cfg, data=data)
        fc_e = np.asarray(compute_simulated_fc(network, bundle_e, cfg,
                                               sim_duration_ms=dur, skip_tr=skip), np.float32)
        se_e = measure_se(bundle_e)
        eib = dict(corr=float(fc_corr(fc_e, t32)), mean_se=se_e)
        print(f"[3b] EIB-lite 후 corr={eib['corr']:+.4f}  S_e={se_e:.3f}", flush=True)
    tag = f"idx{a_.idx}" + (f"_sew{a_.se_w:g}" if a_.se_w > 0 else "") \
        + ("_q" if a_.quad else "") + ("_ct" if a_.cei_model else "")
    np.savez(f"{OUT}/{tag}_sim.npz", fc_sim=fc_sim, c_ei=c_ei, corr=corr_fin,
             fc_target=fc, mean_se=se_fin)
    res = dict(idx=a_.idx, a=float(a_ab), b=float(b_ab), algo=a_.algo,
               quad=a_.quad, cei_model=bool(a_.cei_model),
               se_w=a_.se_w, se_tol=a_.se_tol, theta=list(map(float, best_theta)),
               corr=corr_fin, mean_se=se_fin, score=best_score,
               eib=eib, n_evals=len(hist),
               c_ei_mean=float(c_ei.mean()), elapsed_s=round(time.perf_counter() - t0, 1),
               hist=hist)
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nRESULT idx={a_.idx}  corr={corr_fin:+.4f}  mean_S_e={se_fin:.3f}  "
          f"θ={np.round(best_theta,3)}  총 {res['elapsed_s']/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
