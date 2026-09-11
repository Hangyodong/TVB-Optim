"""test_modelb.py — Model B 모듈 self-check (plain assert, pytest 불필요).

실행: python test_modelb.py
"""
import numpy as np


DKPD25_SUBCORTEX = [
    "L_red_nucleus", "R_red_nucleus", "L_substantia_nigra", "R_substantia_nigra",
    "L_subthalamic_nucleus", "R_subthalamic_nucleus", "L_caudate", "R_caudate",
    "L_putamen", "R_putamen", "L_globus_pallidus_externa", "R_globus_pallidus_externa",
    "L_globus_pallidus_interna", "R_globus_pallidus_interna", "L_thalamus", "R_thalamus",
]


def test_tm_q_trace():
    from dbs_tm import tm_q_trace
    from dbs_std import std_steady_release

    dt, period, n_pulses, onset = 1.0, 8, 4000, 100   # 125Hz, 32s
    total = onset + n_pulses * period + 50

    # E: depressing (τ_fac=0)
    q = tm_q_trace(total, onset, period, n_pulses, dt, U=0.5, tau_rec_ms=800.0)
    assert q.shape == (total,)
    assert 0.0 <= q.min() and q.max() <= 1.0, "q=u·x 범위 이탈"
    assert np.all(q[:onset] == 0.0), "온셋 전 0"
    assert np.all(q[onset + n_pulses * period:] == 0.0), "자극 창 뒤 0"
    assert abs(q[onset] - 0.5) < 1e-12, f"첫 펄스 release=U·1, got {q[onset]}"
    # 정상상태: 마지막 100 펄스 시점의 release ≈ 해석해 (dbs_std.std_steady_release)
    pulse_steps = onset + np.arange(n_pulses) * period
    tail = q[pulse_steps[-100:]]
    q_ss = std_steady_release(1000.0 / (period * dt), 0.5, 800.0)
    assert abs(tail.mean() - q_ss) / q_ss < 0.02, f"steady release {tail.mean():.5f} vs 해석해 {q_ss:.5f}"
    # depression: 펄스 시점 release 는 단조 감소(초반)
    head = q[pulse_steps[:20]]
    assert np.all(np.diff(head) < 0), "depressing 인데 release 증가"

    # I: facilitating — 초반 release 상승
    qf = tm_q_trace(total, onset, period, n_pulses, dt,
                    U=0.15, tau_rec_ms=138.0, tau_fac_ms=670.0)
    assert 0.0 <= qf.min() and qf.max() <= 1.0, "q=u·x 범위 이탈"
    headf = qf[pulse_steps[:5]]
    assert headf[1] > headf[0], f"facilitating 인데 release 하락: {headf[:3]}"
    # 방출→facilitation 순서 규약 검증: 두 번째 상승 > 첫 번째 상승
    assert headf[2] > headf[1], f"방출→facilitation 순서 위반: {headf[:3]}"
    # 정상상태도 해석해와 일치
    qf_ss = std_steady_release(1000.0 / (period * dt), 0.15, 138.0, 670.0)
    tailf = qf[pulse_steps[-100:]]
    assert abs(tailf.mean() - qf_ss) / qf_ss < 0.02
    # 펄스 사이 회복: 펄스 직후보다 다음 펄스 직전 q 가 크다
    k = pulse_steps[2000]
    assert q[k + period - 1] > q[k + 1], "펄스 사이 x 회복 안 됨"
    print("test_tm_q_trace OK")


def test_vta_weights():
    from types import SimpleNamespace
    from vta import vta_radius_mm, modelb_weights

    # 반경 공식 sanity (계획 문서에서 검산한 값)
    r2 = vta_radius_mm(2.0, 60.0, 0.2, 60.0, 65.0)
    assert abs(r2 - 2.52) < 0.05, f"r(2mA,60us)={r2}"
    assert vta_radius_mm(3.0, 60.0, 0.2, 60.0, 65.0) > r2 > vta_radius_mm(1.0, 60.0, 0.2, 60.0, 65.0)
    # PW↑ → E_th↓ → 반경↑
    assert vta_radius_mm(2.0, 90.0, 0.2, 60.0, 65.0) > r2

    cfg = SimpleNamespace(vta_contact_index=1, vta_contact_spacing_mm=2.0,
                          vta_lead_tilt_deg=20.0, vta_sigma_s_per_m=0.2,
                          vta_e_rheobase_v_per_m=60.0, vta_chronaxie_us=65.0)
    atlas = "data/DK+PD25/DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz"
    STN_L = 70   # node index (라벨 71)

    w15 = modelb_weights(atlas, STN_L, 1.5, 60.0, cfg)
    w30 = modelb_weights(atlas, STN_L, 3.0, 60.0, cfg)
    assert w15.shape == (82,) and w30.shape == (82,)
    assert w15[STN_L] == 1.0 and w30[STN_L] == 1.0, "§9: w_t=1"
    assert np.all(w15 >= 0) and np.all(w15 <= 1) and np.all(w30 <= 1), "§9: 0≤w≤1"
    # §12: amplitude↑ → spillover ROI 수·크기 증가(감소 금지)
    others = np.arange(82) != STN_L
    assert np.all(w30[others] >= w15[others] - 1e-12), "amplitude↑ 인데 spillover 감소"
    assert w30[others].sum() > w15[others].sum(), "amplitude↑ 인데 spillover 총량 불변"
    print(f"  w(1.5mA) nonzero={np.count_nonzero(w15)}, w(3.0mA) nonzero={np.count_nonzero(w30)}")
    # 아틀라스 섬 아티팩트 회귀: STN_L 3mA spillover 에 precuneus(23) 금지, SN(68) 은 살아야 함
    assert w30[23] == 0.0, f"precuneus 섬 아티팩트 재발: w={w30[23]}"
    assert w30[68] > 0.0, "실제 이웃 SN spillover 가 사라짐"

    # 캐시 왕복
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        a = modelb_weights(atlas, STN_L, 2.0, 60.0, cfg, cache_dir=td)
        assert os.path.exists(os.path.join(td, "vta_w_n70_a2_pw60_c1_s0.2_e60_ch65_t20_sp2.npz"))
        b = modelb_weights(atlas, STN_L, 2.0, 60.0, cfg, cache_dir=td)  # 캐시 hit
        assert np.array_equal(a, b)

        # σ 변경 → 캐시 미스 검증
        cfg2 = SimpleNamespace(**{**cfg.__dict__, "vta_sigma_s_per_m": 0.1})
        c = modelb_weights(atlas, STN_L, 2.0, 60.0, cfg2, cache_dir=td)
        assert not np.array_equal(a, c), "σ 변경이 캐시를 무효화하지 않음"
    print("test_vta_weights OK")


def test_config_and_split():
    from config import Config
    cfg = Config()
    assert cfg.dbs_modelb_enable is False, "기본 off — 기존 경로 보존"
    assert cfg.dbs_amplitude_ma == 2.0 and cfg.dbs_pulse_width_us == 60.0
    assert cfg.dbs_tm_tau_fac_e_ms == 0.0 and cfg.dbs_tm_tau_fac_i_ms > 0.0

    from data_loader import derive_cortex_subcortex_indices
    cortex_stub = [f"L_ctx{i}" for i in range(33)] + [f"R_ctx{i}" for i in range(33)]
    labels = cortex_stub + DKPD25_SUBCORTEX
    ci, si = derive_cortex_subcortex_indices(labels)
    assert len(ci) == 66 and len(si) == 16, f"cortex {len(ci)} / subcortex {len(si)}"
    assert list(si) == list(range(66, 82))
    print("test_config_and_split OK")


def test_modelb_dynamics_math():
    """model_b 주입식: rate=0 이면 기존 true_p_t(stim=0)와 동일, rate>0 이면
    dS_e 증가분 = (1-S_e)·γ_e·rate, dS_i 증가분 = γ_i·rate."""
    import jax.numpy as jnp
    from types import SimpleNamespace
    from part4_dbs import _make_stimulated_dynamics
    from config import Config
    from model import ReducedWongWangEIB

    cfg = Config()
    n = 3
    # dynamics 가 받는 params 는 model 의 Bunch (pipeline_contracts.ParamSet 아님!)
    params = ReducedWongWangEIB.DEFAULT_PARAMS
    state = jnp.stack([jnp.full((n,), 0.2), jnp.full((n,), 0.1)], axis=0)
    coupling = SimpleNamespace(coupling=jnp.zeros((2, n)))
    w = jnp.array([1.0, 0.25, 0.0])
    T = 10
    zero = jnp.zeros(T)
    rate = jnp.full(T, 20.0)   # 20 Hz

    fn0 = _make_stimulated_dynamics(None, None, cfg, "model_b",
                                    modelb=dict(w=w, rate_e=zero, rate_i=zero))
    fn1 = _make_stimulated_dynamics(None, None, cfg, "model_b",
                                    modelb=dict(w=w, rate_e=rate, rate_i=rate))
    # 기존 모드 회귀: stim 배열 0 인 true_p_t 와 rate 0 인 model_b 는 동일 미분
    stim0 = jnp.zeros((T, n))
    fn_legacy = _make_stimulated_dynamics(None, stim0, cfg, "true_p_t")
    d0, aux0 = fn0(None, 5.0, state, params, coupling, None)
    dl, _ = fn_legacy(None, 5.0, state, params, coupling, None)
    assert np.allclose(np.asarray(d0), np.asarray(dl), atol=1e-7), "rate=0 model_b ≠ legacy"

    d1, aux1 = fn1(None, 5.0, state, params, coupling, None)
    ge, gi = float(params.gamma_e), float(params.gamma_i)
    S_e, S_i = 0.2, 0.1
    exp_de = (1.0 - S_e) * ge * 20.0 * np.asarray(w)      # ΔdS_e = (1-S_e)γ_e·w·rate
    exp_di = gi * 20.0 * np.asarray(w)                     # ΔdS_i = γ_i·w·rate
    got_de = np.asarray(d1[0] - d0[0])
    got_di = np.asarray(d1[1] - d0[1])
    assert np.allclose(got_de, exp_de, rtol=1e-5), f"{got_de} vs {exp_de}"
    assert np.allclose(got_di, exp_di, rtol=1e-5), f"{got_di} vs {exp_di}"
    assert got_de[2] == 0.0 and got_di[2] == 0.0, "w=0 노드에 주입됨"
    print("test_modelb_dynamics_math OK")


if __name__ == "__main__":
    test_tm_q_trace()
    test_vta_weights()
    test_config_and_split()
    test_modelb_dynamics_math()
    print("ALL OK")
