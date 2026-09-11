#!/usr/bin/env python3
"""match_se_rerun.py — B(변형 w) 를 full pipeline 과 같은 작동점(mean_S_e)에 맞춰 재평가.

FIC 가 종점 S_e 를 통제 못 하므로([[rww-bistable-se-regime]]), B 와 full pipeline 의
corr 차이에 "가중치 효과"와 "가지 착지 효과"가 섞여 있다. c_ei 를 스칼라 k 로 스케일해
B 의 안착 mean_S_e 를 full pipeline 값(0.2628, 6실현 평균)에 맞추면 가중치 효과만 남는다.

절차: k 격자 -> 각 k 마다 600s settle 후 S_e(2 실현×60s) -> 목표에 가장 가까운 k 선택
      -> 그 k 에서 6 실현 S_e + FC corr (SC>0 / SC==0 분해)

S_e 는 c_ei 에 대해 단조감소 → 목표가 현재보다 낮으면 k>1.

실행: python3 match_se_rerun.py
"""
import glob
import sys

import numpy as np
import jax
import jax.numpy as jnp

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import _settle_bundle, compute_simulated_fc
from pipeline_contracts import ParamSet
from stage_trace import load_bundle
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

IDX = 4
TARGET_SE = 0.2628          # full pipeline 6실현 평균
B_PKL = sys.argv[1] if len(sys.argv) > 1 else "output_ppmi_pd/_pfic_idx4/B_perf.pkl"
KGRID = ([float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [1.0010, 1.0020, 1.0030, 1.0040])
SEEDS = [42, 1, 7, 123, 2024, 31337]


def se_of(network, bundle, cfg, seeds, dur_ms=60_000):
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    out = []
    for s in seeds:
        model, st = bundle.to_tvb_state(network, solver, t1=int(dur_ms),
                                        dt=cfg.integration_dt_ms)
        ns = jnp.asarray(st._internal.noise_samples)
        _, sk = jax.random.split(jax.random.PRNGKey(int(s)))
        st._internal.noise_samples = jax.random.normal(sk, ns.shape, ns.dtype)
        out.append(float(np.mean(np.asarray(model(st).data[:, 0, :]))))
    return float(np.mean(out)), float(np.std(out)), out


def main():
    p = M.prepare_pd_data(IDX, 0.02)
    cfg = M.make_config(p, IDX, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_sematch"
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, *_ = build_network(cfg, data)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    iu = np.triu_indices(n, 1)
    mt = mask.astype(bool)[iu]
    t1 = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

    b0, _ = load_bundle(B_PKL)
    c0 = np.asarray(b0.params.c_ei, np.float32)
    wl, wf = np.asarray(b0.params.wLRE), np.asarray(b0.params.wFFI)
    print(f"시작 bundle: {B_PKL}\n시작점: c_ei mean {c0.mean():.4f}   목표 mean_S_e {TARGET_SE:.4f}")
    print(f"\n{'k':>6}{'c_ei mean':>11}{'mean_S_e':>11}{'sd':>8}   실현별")
    print("-" * 70)

    scan = []
    for k in KGRID:
        ps = ParamSet(c_ei=(c0 * k).astype(np.float32), wLRE=wl, wFFI=wf,
                      c_ei_frozen=False).sanitize(mask, cap)
        b = b0.advance(new_params=ps)
        b, _, _ = _settle_bundle(network, b, cfg, sim_duration_ms=t1,
                                 skip_tr=cfg.optimizer_bold_skip_tr, next_stage="eib")
        m, sd, vals = se_of(network, b, cfg, SEEDS[:2])
        scan.append((k, b, m))
        print(f"{k:>7.4f}{(c0*k).mean():>11.4f}{m:>11.4f}{sd:>8.4f}   "
              + " ".join(f"{v:.4f}" for v in vals), flush=True)
        jax.clear_caches()

    kbest, bbest, sbest = min(scan, key=lambda t: abs(t[2] - TARGET_SE))
    print(f"\n선택: k={kbest:.2f}  (S_e {sbest:.4f}, 목표와 {abs(sbest-TARGET_SE):+.4f})")

    m, sd, vals = se_of(network, bbest, cfg, SEEDS)
    fc = np.asarray(compute_simulated_fc(network, bbest, cfg, sim_duration_ms=t1,
                                         skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
    x, y = FC_emp[iu], fc[iu]
    print(f"\n{'조건':<30}{'corr':>9}{'SC>0':>9}{'SC==0':>9}{'rmse':>9}"
          f"{'c_ei':>8}{'S_e(6실현)':>12}")
    print("-" * 88)
    print(f"{'B 변형 w, S_e 매칭 (k=%.2f)' % kbest:<30}"
          f"{np.corrcoef(x,y)[0,1]:>+9.4f}"
          f"{np.corrcoef(x[mt],y[mt])[0,1]:>+9.4f}"
          f"{np.corrcoef(x[~mt],y[~mt])[0,1]:>+9.4f}"
          f"{np.sqrt(np.mean((y-x)**2)):>9.4f}"
          f"{np.asarray(bbest.params.c_ei).mean():>8.4f}{m:>12.4f}")
    print(f"{'B 변형 w, 매칭 전 (FIC 그대로)':<30}"
          f"{0.6784:>+9.4f}{0.7221:>+9.4f}{0.6212:>+9.4f}{0.1852:>9.4f}"
          f"{1.7479:>8.4f}{0.2779:>12.4f}")
    print(f"{'A 원본 w + FIC':<30}{0.6332:>+9.4f}{0.7701:>+9.4f}{0.4232:>+9.4f}"
          f"{0.1624:>9.4f}{1.8990:>8.4f}{0.2265:>12.4f}")
    print(f"{'full pipeline (원본 w, opt c_ei)':<30}{0.7524:>+9.4f}"
          f"{'—':>9}{'—':>9}{'—':>9}{1.8857:>8.4f}{0.2628:>12.4f}")
    print(f"\n6실현 S_e: " + " ".join(f"{v:.4f}" for v in vals) + f"  (sd {sd:.4f})")


if __name__ == "__main__":
    main()
