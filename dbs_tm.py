"""dbs_tm.py — Model B: DBS 펄스열 → Tsodyks-Markram E/I q(t) trace (문서 §15-§17).

q(t) = u(t)·x(t). 펄스 도착 시 release = u·x → x -= release, u += U(1-u)(facilitation);
펄스 사이 x→1 (τ_rec), u→U (τ_fac) 지수 완화. 방출·갱신 순서는 dbs_std.std_release_train
과 동일 규약(release 먼저, facilitation 나중) — std_steady_release 해석해와 일치한다.

part4 model_b 모드가 H_DBS(t) = w_i · f · q(t) [Hz] 로 RWW gating 에 가산한다 (§18-§19).
자극 창 밖(t < onset_step, t ≥ onset_step + n_pulses·period_steps)은 0.
"""
import numpy as np


def tm_q_trace(total_steps, onset_step, period_steps, n_pulses, dt_ms,
               U, tau_rec_ms, tau_fac_ms=0.0):
    """TM q(t)=u·x trace, shape (total_steps,) float64. 자극 창 밖 0.

    ponytail: 780k step 파이썬 루프 ≈1s — 조합당 1회 선계산이라 충분. 느려지면
    펄스 구간별 닫힌형(지수 완화)으로 벡터화.
    """
    total_steps = int(total_steps)
    onset_step = int(onset_step)
    period_steps = int(period_steps)
    n_pulses = int(n_pulses)
    q = np.zeros(total_steps, dtype=np.float64)
    er = np.exp(-dt_ms / tau_rec_ms)
    fac = tau_fac_ms > 0.0
    ef = np.exp(-dt_ms / tau_fac_ms) if fac else 0.0
    x, u = 1.0, U
    end = min(total_steps, onset_step + n_pulses * period_steps)
    for t in range(onset_step, end):
        q[t] = u * x
        if (t - onset_step) % period_steps == 0:      # 펄스 도착 (q[t]=release=u·x)
            x -= u * x
            if fac:
                u += U * (1.0 - u)
        x = 1.0 - (1.0 - x) * er                      # dt 완화
        if fac:
            u = U + (u - U) * ef
    return q
