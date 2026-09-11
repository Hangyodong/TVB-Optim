#!/usr/bin/env python3
"""unclip_test.py — wFFI 하한 clip 을 푼 가중치가 모델에서 더 나은지 직접 확인.

Part2/3 는 매 업데이트마다 clip(w, 0, w_max) 를 걸어서(part2_eib.py:585) wFFI 의
5~8% 엣지가 0 에 붙어 있다. 그 엣지들은 아핀관계 wFFI = a + b*FC_emp (b<0) 가 음수를
요구하는 자리다. 여기서는 "그 음수를 그대로 넣으면 FC 재현이 좋아지는가" 를 본다.

조건
  A  원본 (clip 상태)                                   ← 재현 기준선
  B  clip 엣지만 아핀 외삽 음수로 복원                    ← 하한 제거의 직접 효과
  C  B + c_ei 스칼라 k 로 mean_S_e 를 A 에 맞춤           ← 작동점 이동분 제거

가중치가 학습된 FC 를 그대로 쓰기 위해 MAT_PATH 를 구 .mat 으로 되돌린다
(final .mat 은 같은 subject 라도 FC 가 미세하게 달라 캐시 태그가 어긋난다).

실행: python3 unclip_test.py [idx]
"""
import glob
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

import main_ppmi_pd as M

M.MAT_PATH = "data/AALv3/FC_AAL_ComBat_all_163.mat"   # 가중치 학습 시점 FC

from data_loader import load_data                      # noqa: E402
from model import build_network                        # noqa: E402
from part3_gradient import _settle_bundle, compute_simulated_fc  # noqa: E402
from pipeline_contracts import ParamSet                # noqa: E402
from stage_trace import load_bundle                    # noqa: E402
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun  # noqa: E402

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025",
          9: "101070", 10: "101124", 11: "101146", 12: "101174"}
SEEDS = [42, 1, 7]
KGRID = [1.002, 1.005, 1.010]


def best_bundle_path(sub):
    """load_ref 와 같은 규칙: inputs/FC.csv 해시 일치 + post_grad_fc_corr 최대."""
    import hashlib
    fc = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")[:8, :8]
    hs = {hashlib.sha1(np.asarray(fc, t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc = None, -9.0
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        d = os.path.basename(os.path.dirname(f))
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest")):
            continue
        if not any(h in d for h in hs):
            continue
        with open(f, "rb") as fh:
            o = pickle.load(fh)
        g = lambda x, k, d=None: x.get(k, d) if isinstance(x, dict) else getattr(x, k, d)
        md = g(g(o, "bundle", o), "metadata", {}) or {}
        c = float(md.get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            best, bc = f, c
    return best, bc


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
    return float(np.mean(out)), float(np.std(out))


def main():
    sub = SUBDIR[IDX]
    p = M.prepare_pd_data(IDX, 0.02)
    cfg = M.make_config(p, IDX, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_unclip"
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, *_ = build_network(cfg, data)
    mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    iu = np.triu_indices(n, 1)
    mt = mask.astype(bool)[iu]
    t1 = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

    path, refcorr = best_bundle_path(sub)
    print(f"idx {IDX} ({sub})  bundle={os.path.basename(os.path.dirname(path))}"
          f"  post_grad_corr={refcorr:.4f}", flush=True)
    b0, _ = load_bundle(path)
    c0 = np.asarray(b0.params.c_ei, np.float32)
    wl = np.asarray(b0.params.wLRE, np.float64)
    wf = np.asarray(b0.params.wFFI, np.float64)

    # 아핀 회귀는 SC>0 상삼각에서 (분석 스크립트와 동일 표본)
    scm = mask.astype(bool)
    x, y = FC_emp[iu][mt], wf[iu][mt]
    (b, a), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
    clip_edge = scm & (wf <= 1e-6) & ~np.eye(n, dtype=bool)
    wf_un = wf.copy()
    wf_un[clip_edge] = a + b * FC_emp[clip_edge]          # 음수 허용
    wf_un = 0.5 * (wf_un + wf_un.T) * scm
    npos = int((clip_edge[iu]).sum())
    print(f"아핀: wFFI = {a:+.4f} {b:+.4f}*FC   clip 엣지 {npos} "
          f"({npos/mt.sum()*100:.2f}% of SC>0)   복원값 [{wf_un[clip_edge].min():+.3f}, "
          f"{wf_un[clip_edge].max():+.3f}]  평균 {wf_un[clip_edge].mean():+.3f}", flush=True)

    def evaluate(tag, wLRE, wFFI, c_ei):
        # sanitize 를 안 쓴다 — clip 을 건너뛰는 게 이 실험의 요점.
        ps = ParamSet(c_ei=np.asarray(c_ei, np.float32),
                      wLRE=np.asarray(wLRE, np.float32),
                      wFFI=np.asarray(wFFI, np.float32), c_ei_frozen=False)
        bb = b0.advance(new_params=ps)
        bb, _, _ = _settle_bundle(network, bb, cfg, sim_duration_ms=t1,
                                  skip_tr=cfg.optimizer_bold_skip_tr, next_stage="eib")
        se, sd = se_of(network, bb, cfg, SEEDS)
        fc = np.asarray(compute_simulated_fc(network, bb, cfg, sim_duration_ms=t1,
                                             skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
        u, v = FC_emp[iu], fc[iu]
        r = dict(tag=tag, se=se, se_sd=sd,
                 corr=float(np.corrcoef(u, v)[0, 1]),
                 corr_sc=float(np.corrcoef(u[mt], v[mt])[0, 1]),
                 corr_nosc=float(np.corrcoef(u[~mt], v[~mt])[0, 1]),
                 rmse=float(np.sqrt(np.mean((v - u) ** 2))),
                 c_ei=float(np.mean(c_ei)))
        print(f"  [{tag}] S_e={se:.4f}±{sd:.4f}  corr={r['corr']:+.4f} "
              f"(SC>0 {r['corr_sc']:+.4f} / SC==0 {r['corr_nosc']:+.4f})  "
              f"rmse={r['rmse']:.4f}  c_ei={r['c_ei']:.4f}", flush=True)
        jax.clear_caches()
        return r

    print("\n조건 평가", flush=True)
    A = evaluate("A 원본(clip)", wl, wf, c0)
    B = evaluate("B clip 해제", wl, wf_un, c0)

    # 작동점이 밀렸으면 c_ei 로 되돌린 뒤 다시 본다 (S_e 는 c_ei 에 단조감소)
    if abs(B["se"] - A["se"]) > 0.003:
        print(f"\nS_e 가 {B['se']-A['se']:+.4f} 이동 → c_ei 스케일로 매칭", flush=True)
        best = None
        for k in (KGRID if B["se"] > A["se"] else [2 - k for k in KGRID]):
            r = evaluate(f"C k={k:.3f}", wl, wf_un, c0 * k)
            if best is None or abs(r["se"] - A["se"]) < abs(best["se"] - A["se"]):
                best = r
        print(f"\n최적 매칭: {best['tag']}  S_e={best['se']:.4f} (A {A['se']:.4f}) "
              f"corr={best['corr']:+.4f} (A {A['corr']:+.4f})")


if __name__ == "__main__":
    main()
