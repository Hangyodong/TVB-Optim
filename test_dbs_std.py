#!/usr/bin/env python3
"""test_dbs_std.py — STD DBS 전달 검증.

unit: TM 재귀 vs 해석해, 주파수 포화 곡선, 배열 생성.
integration (--full): mouse idx0 실제 RWW 경로(part4 주입부 그대로)로
  130Hz-STD vs 20Hz-STD vs 130Hz-biphasic 의 ΔS_e 비교.
실행: python3 test_dbs_std.py [--full]
"""
import argparse
import sys
import types

import numpy as np

from dbs_std import build_std_drive, std_release_train, std_steady_release


def unit_tests():
    U, tr = 0.5, 800.0
    # 1) 재귀 정상상태 == 해석해
    for f in (10.0, 20.0, 60.0, 130.0, 180.0):
        rel = std_release_train(2000, 1000.0 / f, U, tr)
        ss = std_steady_release(f, U, tr)
        assert abs(rel[-1] - ss) < 1e-9, (f, rel[-1], ss)
        assert np.all(np.diff(rel) <= 1e-12), "depression은 단조 감소여야"
    # 2) 주파수 포화: 초당 전달(f·rel_ss)이 130/20 에서 선형(6.5배) 대신 ~1배
    r20 = 20 * std_steady_release(20, U, tr)
    r130 = 130 * std_steady_release(130, U, tr)
    ratio = r130 / r20
    assert ratio < 1.5, ratio
    print(f"[unit] 초당 전달 비 130Hz/20Hz = {ratio:.2f} (STD 없으면 6.5) — 포화 OK")
    # 3) 배열: 타깃 노드만, 평균 구동 = amp·rel_ss/U·τ_syn/period 예측치와 일치
    cfg = types.SimpleNamespace(integration_dt_ms=1.0, dbs_pulse_amplitude=1.0,
                                dbs_std_U=U, dbs_std_tau_rec_ms=tr,
                                dbs_std_tau_fac_ms=0.0, dbs_std_tau_syn_ms=5.0)
    period, npul, onset, total = 8, 5000, 1000, 41000
    arr = build_std_drive(3, 10, onset, total, period, npul, cfg)
    assert arr.shape == (total, 10)
    assert np.all(arr[:, [c for c in range(10) if c != 3]] == 0)
    assert np.all(arr[:onset] == 0)
    ss = std_steady_release(1000.0 / period, U, tr)
    # 커널 적분 τ_syn/(1-e^{-dt/τ}) ≈ τ_syn+dt/2; 정확값으로 예측
    gain = 1.0 / (1.0 - np.exp(-1.0 / 5.0))
    pred = (ss / U) * gain / period
    got = arr[total // 2:, 3].mean()
    assert abs(got - pred) / pred < 0.05, (got, pred)
    print(f"[unit] 평균 구동 실측 {got:.5f} vs 예측 {pred:.5f} — OK")
    print("[unit] ALL PASS")


def run_sim(ctx, bundle, stim_arr, total_ms):
    import jax
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics import prepare
    from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun
    from part4_dbs import _make_stimulated_dynamics

    network = ctx["network"]
    original = network.dynamics.dynamics
    fn = _make_stimulated_dynamics(original, jnp.asarray(stim_arr, jnp.float32),
                                   ctx["cfg"], "true_p_t")
    network.dynamics.dynamics = types.MethodType(fn, network.dynamics)
    try:
        bundle.apply_to_network(network)
        solver = BoundedSolver(Heun(), low=0.0, high=1.0)
        model, st = prepare(network, solver, t1=int(total_ms),
                            dt=ctx["cfg"].integration_dt_ms)
        st = bundle.apply_to_state(st)
        out = jax.block_until_ready(model(st))
    finally:
        network.dynamics.dynamics = original
    return np.asarray(out.data)  # [steps, 2(state), nodes] + aux 아님: data=[t, var, node]


def integration_test():
    from dataclasses import replace
    import fopt_mouse as F
    from part4_dbs import _build_stim_array, _compute_derived_parameters
    from pipeline_contracts import ParamSet, StateBundle

    a_ = types.SimpleNamespace(idx=0, se_target=0.161, noise_level=0.01, window_tr=600)
    ctx = F.build_ctx(a_)
    z = np.load("output_mouse116/_fopt/idx0_ns01_sim.npz")
    bundle = StateBundle.from_warmup(
        warmup_result=ctx["warmup_result"], bold_monitor_template=ctx["bold_monitor"],
        initial_params=ParamSet(c_ei=z["c_ei"].astype(np.float32),
                                wLRE=z["wLRE"].astype(np.float32),
                                wFFI=z["wFFI"].astype(np.float32), c_ei_frozen=True),
        internal_state=ctx["internal"], delay_history=ctx["delay_hist"], stage="warmup")

    node = 0
    pre_ms, stim_ms = 20_000.0, 20_000.0
    results = {}
    for name, freq, std_on in [("std130", 130.0, True), ("std20", 20.0, True),
                               ("biph130", 130.0, False), ("biph20", 20.0, False)]:
        cfg = replace(ctx["cfg"], dbs_pulse_amplitude=1.0,
                      dbs_stimulation_frequency_hz=freq,
                      dbs_pre_stimulation_duration_ms=pre_ms,
                      dbs_stimulation_duration_ms=stim_ms,
                      dbs_std_enable=std_on)
        derived = _compute_derived_parameters(cfg)
        pre_steps = int(round(pre_ms / cfg.integration_dt_ms))
        total_steps = pre_steps + derived["n_pulses"] * derived["period_steps"]
        arr, _ = _build_stim_array(cfg, derived, node, ctx["n"], pre_steps, total_steps)
        ctx2 = dict(ctx); ctx2["cfg"] = cfg
        data = run_sim(ctx2, bundle, arr, total_steps * cfg.integration_dt_ms)
        se = data[:, 0, node]
        pre = float(se[pre_steps // 2:pre_steps].mean())       # pre 후반 10s
        dur = float(se[pre_steps + (total_steps - pre_steps) // 2:].mean())  # stim 후반
        mean_drive = float(arr[pre_steps:, node].mean())
        results[name] = (pre, dur, dur - pre, mean_drive)
        print(f"[sim] {name:8s} mean_drive={mean_drive:.5f}  "
              f"S_e pre={pre:.4f} during={dur:.4f}  ΔS_e={dur-pre:+.4f}", flush=True)

    d130, d20 = results["std130"][2], results["std20"][2]
    b130, b20 = results["biph130"][2], results["biph20"][2]
    assert d130 > 0.002, f"STD130 과활성화 미검출: {d130}"
    assert d20 > 0.002, f"STD20 과활성화 미검출: {d20}"
    std_ratio, biph_ratio = d130 / d20, b130 / b20
    # 핵심 주장: STD는 주파수 포화(≈1), 원 biphasic은 펄스수에 민감(≫1)
    assert std_ratio < 1.5, f"STD 포화 기대와 불일치: {std_ratio}"
    assert biph_ratio > 2.0, f"biphasic은 f-민감해야: {biph_ratio}"
    assert biph_ratio > 2.0 * std_ratio, (std_ratio, biph_ratio)
    assert results["std130"][1] < 0.9, "STD130 S_e 포화(validity 밖)"
    print(f"[sim] ΔS_e 비 130/20: STD={std_ratio:.2f} (포화) vs biphasic={biph_ratio:.2f} "
          f"— 주파수의 비선형 반영 확인")
    if results["biph130"][1] > 0.6:
        print(f"[sim] 경고: biphasic amp=1.0 은 S_e={results['biph130'][1]:.2f} "
              f"— 정류로 과대 구동, validity 밖. STD 경로 권장 근거.")
    print("[integration] ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="GPU integration test 포함")
    a = ap.parse_args()
    unit_tests()
    if a.full:
        integration_test()
    else:
        print("(integration test: --full 로 실행)")
