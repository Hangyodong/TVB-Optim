"""dbs_std.py — DBS 펄스열의 Tsodyks-Markram 단기억제(STD) 시냅스 전달.

전극 펄스(→축삭 발화 1회) → STD 시냅스 방출 u_n·x_n → exp 커널 유효 전류.
고주파일수록 depression 으로 per-pulse 방출이 줄어 초당 전달 전하가 포화
→ 자극 주파수/진폭/펄스수가 유효 구동에 비선형으로 반영된다.
biphasic 원파형과 달리 단극성(시냅스 전류) — 부호는 dbs_pulse_amplitude 로 결정.
"""
import numpy as np
from scipy.signal import lfilter


def std_release_train(n_pulses, period_ms, U, tau_rec_ms, tau_fac_ms=0.0):
    """펄스별 방출량 u_n·x_n (TM 이산 재귀). 첫 펄스 = U·1."""
    rel = np.empty(n_pulses, dtype=np.float64)
    x, u = 1.0, U
    er = np.exp(-period_ms / tau_rec_ms)
    ef = np.exp(-period_ms / tau_fac_ms) if tau_fac_ms > 0 else 0.0
    for n in range(n_pulses):
        rel[n] = u * x
        x = 1.0 - (1.0 - x * (1.0 - u)) * er
        u = U + u * (1.0 - U) * ef
    return rel


def std_steady_release(freq_hz, U, tau_rec_ms, tau_fac_ms=0.0):
    """정상상태 방출량 해석해 (검증용)."""
    e = np.exp(-1000.0 / (freq_hz * tau_rec_ms))
    if tau_fac_ms > 0:
        u = U / (1.0 - (1.0 - U) * np.exp(-1000.0 / (freq_hz * tau_fac_ms)))
    else:
        u = U
    x = (1.0 - e) / (1.0 - (1.0 - u) * e)
    return u * x


def build_std_drive(target_node_index, n_nodes, onset_step, total_steps,
                    period_steps, n_pulses, cfg):
    """STD 통과 유효 전류 배열 [total_steps, n_nodes].

    첫 펄스 peak = cfg.dbs_pulse_amplitude (방출량을 U 로 정규화), 이후 depression 반영.
    커널: exp 감쇠 τ_syn (겹치면 합산 → 고주파에서 tonic, 저주파에서 pulsatile).
    """
    dt = cfg.integration_dt_ms
    rel = std_release_train(n_pulses, period_steps * dt, cfg.dbs_std_U,
                            cfg.dbs_std_tau_rec_ms, cfg.dbs_std_tau_fac_ms)
    impulses = np.zeros(total_steps, dtype=np.float64)
    steps = onset_step + np.arange(n_pulses) * period_steps
    keep = steps < total_steps
    impulses[steps[keep]] = cfg.dbs_pulse_amplitude * rel[keep] / cfg.dbs_std_U
    decay = np.exp(-dt / cfg.dbs_std_tau_syn_ms)
    drive = lfilter([1.0], [1.0, -decay], impulses)
    out = np.zeros((total_steps, n_nodes), dtype=np.float32)
    out[:, target_node_index] = drive.astype(np.float32)
    return out
