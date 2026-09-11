#!/usr/bin/env python3
"""probe_se_spread.py — mean_S_e 측정값이 조건 차이인가 측정 잡음인가.

A / B / full-pipeline 세 bundle 에 대해
  (1) 노이즈 실현 6 개 × 60s  -> 실현 간 산포
  (2) 연속 300s 를 30s 씩 10 구간 -> 시간 내 가지 전환 폭
을 재서, 조건 간 차이(0.227 vs 0.292 vs 0.258)와 비교한다.

실행: python3 probe_se_spread.py
"""
import numpy as np
import jax
import jax.numpy as jnp

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from stage_trace import load_bundle, plain_cache_dir, newest
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

IDX = 4
SEEDS = [42, 1, 7, 123, 2024, 31337]


def se_run(network, bundle, cfg, dur_ms, seed=None):
    """시간·노드 평균 S_e. seed 지정 시 노이즈 실현 교체. 30s 구간별 평균도 반환."""
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    model, state = bundle.to_tvb_state(network, solver, t1=int(dur_ms),
                                       dt=cfg.integration_dt_ms)
    if seed is not None:
        ns = jnp.asarray(state._internal.noise_samples)
        _, sk = jax.random.split(jax.random.PRNGKey(int(seed)))
        state._internal.noise_samples = jax.random.normal(sk, ns.shape, ns.dtype)
    ys = np.asarray(model(state).data[:, 0, :])          # [t, node]
    step = int(30_000 / cfg.integration_dt_ms)
    chunks = [ys[i:i + step].mean() for i in range(0, len(ys) - step + 1, step)]
    return float(ys.mean()), np.array(chunks, dtype=np.float64)


def main():
    p = M.prepare_pd_data(IDX, 0.02)
    cfg = M.make_config(p, IDX, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_sespread"
    data = load_data(cfg)
    network, *_ = build_network(cfg, data)

    grad = newest(__import__("glob").glob(
        f"{plain_cache_dir('100878')}/grad_*.pkl"))
    conds = [("A 원본 w + FIC", "output_ppmi_pd/_pfic_idx4/A_orig.pkl", 0.2274),
             ("B 변형 w + FIC", "output_ppmi_pd/_pfic_idx4/B_perf.pkl", 0.2919),
             ("full pipeline", grad, 0.2583)]

    print(f"{'조건':<18}{'앞서 보고':>10}{'6실현 평균':>11}{'sd':>8}"
          f"{'실현 범위':>18}{'300s 30s구간 범위':>20}")
    print("-" * 88)
    for label, path, prev in conds:
        b, _ = load_bundle(path)
        vals = [se_run(network, b, cfg, 60_000, s)[0] for s in SEEDS]
        v = np.array(vals)
        _, chunks = se_run(network, b, cfg, 300_000, None)
        print(f"{label:<18}{prev:>10.4f}{v.mean():>11.4f}{v.std():>8.4f}"
              f"{'[%.4f, %.4f]' % (v.min(), v.max()):>18}"
              f"{'[%.4f, %.4f]' % (chunks.min(), chunks.max()):>20}")
        print(f"{'':18}실현별: " + " ".join(f"{x:.4f}" for x in vals))
        print(f"{'':18}30s구간: " + " ".join(f"{x:.3f}" for x in chunks))
        jax.clear_caches()

    print("\n판정 기준: 조건 간 차이 |0.2919−0.2274| = 0.0645 가 위 산포보다 "
          "충분히 큰가?")


if __name__ == "__main__":
    main()
