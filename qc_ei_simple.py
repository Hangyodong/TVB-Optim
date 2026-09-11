#!/usr/bin/env python3
"""qc_ei_simple.py — simple optim(formula_optim_vmap)의 E/I 튜닝 QC.

② 파라미터 안정성: K회 재-fit (CMA seed + x0 jitter + noise 실현 모두 변경)
   → θ CV, 파라미터맵(wLRE/wFFI/c_ei) 쌍별 corr, FC corr 분산.
③ 파라미터 복원: 알려진 θ_true 로 합성 FC 생성(별도 noise) → 같은 절차로 재-fit
   → θ 복원 오차 + 맵 corr.
④ DBS 강건성: ②의 각 해에 동일 STN_L DBS(STD 모드) → ΔFC = FC_during − FC_pre,
   해 쌍별 corr(vec(ΔFC)) 중앙값 = R_DBS.

noise 는 fit 마다 다른 실현을 쓴다: base_state 의 _internal.noise_samples leaf 를
param leaf 와 같은 방식(id 매칭)으로 vmap 인자에 포함시켜 재컴파일 없이 교체.

실행:  python3 qc_ei_simple.py --idx 4 --phase 2   (기본 K=20)
       python3 qc_ei_simple.py --idx 4 --phase 3
       python3 qc_ei_simple.py --idx 4 --phase 4
출력: output_ppmi_pd/_qc/idx<N>_phase{2,3}.json/.npz, phase4는 dbs 하위폴더 + json
"""
import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np

from analytic_fic import analytic_c_ei
from formula_optim import IU, X0
from formula_optim_vmap import build_ctx, make_bundle
from pipeline_contracts import ParamSet, StateBundle
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

QC_OUT = "output_ppmi_pd/_qc"
SE_WIN_MS = 60_000


# ────────────────────────── vmap 평가기 (noise 교체 지원) ──────────────────────────
def make_batch_eval_noise(ctx, chunk=8):
    """(cei[K,N], wLRE[K,N,N], wFFI[K,N,N], noise[K,...]) → corr/rmse/se/fc.
    formula_optim_vmap.make_batch_eval 와 동일하되 noise_samples 도 배치 인자."""
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    cei0, wl0, wf0 = make_bundle(ctx, list(X0) + [ctx["se_target"]])
    bundle0 = StateBundle.from_warmup(
        warmup_result=ctx["warmup_result"], bold_monitor_template=ctx["bold_monitor"],
        initial_params=ParamSet(c_ei=cei0, wLRE=wl0, wFFI=wf0, c_ei_frozen=True),
        internal_state=ctx["internal"], delay_history=ctx["delay_hist"], stage="warmup")
    model, base_state = bundle0.to_tvb_state(
        ctx["network"], solver, t1=ctx["dur"], dt=ctx["cfg"].integration_dt_ms)
    monitor = bundle0.build_bold_monitor(ctx["cfg"])

    leaves, treedef = jax.tree_util.tree_flatten(base_state)
    noise0 = base_state._internal.noise_samples
    tgt_ids = [id(base_state.dynamics.c_ei),
               id(base_state.coupling.coupling.wLRE),
               id(base_state.coupling.coupling.wFFI),
               id(noise0)]
    idxs = [next(i for i, l in enumerate(leaves) if id(l) == t) for t in tgt_ids]
    assert len(set(idxs)) == 4, "param/noise leaf 식별 실패"

    tgt_holder = {"tgt": jnp.asarray(ctx["t32"])}
    iu0, iu1 = jnp.asarray(IU[0]), jnp.asarray(IU[1])
    skip = ctx["skip"]
    se_win = int(SE_WIN_MS / 4.0)
    n = ctx["n"]

    def run_one(cei, wl, wf, noise, tgt):
        ls = list(leaves)
        ls[idxs[0]], ls[idxs[1]], ls[idxs[2]], ls[idxs[3]] = cei, wl, wf, noise
        st = jax.tree_util.tree_unflatten(treedef, ls)
        sim = model(st)
        bold = monitor(sim)
        ys = bold.ys[:, 0, :] if bold.ys.ndim == 3 else bold.ys
        ts = jnp.nan_to_num(ys[skip:])
        ts = ts - ts.mean(0, keepdims=True)
        tsn = ts / jnp.maximum(ts.std(0, keepdims=True), 1e-6)
        fcm = (tsn.T @ tsn) / jnp.maximum(tsn.shape[0] - 1, 1)
        fcm = jnp.clip(fcm, -1.0, 1.0) * (1.0 - jnp.eye(n))
        corr = jnp.corrcoef(fcm.flatten(), tgt.flatten())[0, 1]
        rmse = jnp.sqrt(jnp.mean((fcm[iu0, iu1] - tgt[iu0, iu1]) ** 2))
        se = jnp.mean(sim.data[-se_win:, 0, :])
        return corr, rmse, se, fcm

    vmapped = jax.jit(jax.vmap(run_one, in_axes=(0, 0, 0, 0, None)))
    noise_shape = jnp.asarray(noise0).shape
    noise_dtype = jnp.asarray(noise0).dtype

    def batch_eval(cei_b, wl_b, wf_b, noise_b):
        outs = []
        for s in range(0, len(cei_b), chunk):
            sl = slice(s, s + chunk)
            outs.append(vmapped(jnp.asarray(cei_b[sl]), jnp.asarray(wl_b[sl]),
                                jnp.asarray(wf_b[sl]), jnp.asarray(noise_b[sl]),
                                tgt_holder["tgt"]))
        return [np.concatenate([np.asarray(o[k]) for o in outs]) for k in range(4)]

    return batch_eval, noise_shape, noise_dtype, tgt_holder


def gen_noise(seed, shape, dtype):
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype)


# ────────────────────────── CMA 1회 fit ──────────────────────────
def run_fit(ctx, batch_eval, noise_shape, noise_dtype, fit_seed, a_):
    """seed 로 x0 jitter + CMA seed + noise 실현을 모두 결정. formula_optim_vmap [2]와 동일 목적함수."""
    import cma
    rng = np.random.default_rng(fit_seed)
    se_target = ctx["se_target"]
    x0 = list(np.clip(np.asarray(X0) + rng.normal(0, 0.4, 4), 0, 8)) + \
         [float(np.clip(se_target + rng.normal(0, 0.02), 0.20, 0.32))]
    noise_one = np.asarray(gen_noise(10_000 + fit_seed, noise_shape, noise_dtype))

    def score_of(c, rmse, se):
        pen = a_.se_w * max(0.0, abs(se - se_target) - a_.se_tol) if a_.se_w > 0 else 0.0
        return a_.corr_w * c - a_.rmse_w * rmse - pen

    es = cma.CMAEvolutionStrategy(x0, 1.0, dict(
        bounds=[[0, 0, 0, 0, 0.20], [8, 8, 8, 8, 0.32]],
        CMA_stds=[0.4, 0.4, 0.4, 0.4, 0.02],
        popsize=a_.popsize, maxfevals=a_.maxfev, seed=int(fit_seed), verbose=-9))
    hist = []
    while not es.stop():
        xs = es.ask()
        ps = [make_bundle(ctx, th) for th in xs]
        cei_b = np.stack([p[0] for p in ps]); wl_b = np.stack([p[1] for p in ps])
        wf_b = np.stack([p[2] for p in ps])
        noise_b = np.broadcast_to(noise_one, (len(xs),) + noise_one.shape)
        cs, rs, ses, _ = batch_eval(cei_b, wl_b, wf_b, noise_b)
        scores = [score_of(float(cs[k]), float(rs[k]), float(ses[k])) for k in range(len(xs))]
        for k, x in enumerate(xs):
            hist.append((list(map(float, x)), float(cs[k]), float(ses[k]), scores[k]))
        es.tell(xs, [-s for s in scores])

    k = int(np.argmax([h[3] for h in hist]))
    theta = np.array(hist[k][0])
    cei, wl, wf = make_bundle(ctx, list(theta))
    noise_b = np.broadcast_to(noise_one, (a_.popsize,) + noise_one.shape)
    cs, rs, ses, fcs = batch_eval(np.stack([cei] * a_.popsize), np.stack([wl] * a_.popsize),
                                  np.stack([wf] * a_.popsize), noise_b)
    return dict(seed=int(fit_seed), theta=list(map(float, theta)),
                corr=float(cs[0]), rmse=float(rs[0]), se=float(ses[0]),
                score=hist[k][3], n_evals=len(hist)), \
        dict(c_ei=cei, wLRE=wl, wFFI=wf, fc_sim=np.asarray(fcs[0], np.float32))


# ────────────────────────── 지표 ──────────────────────────
def pairwise_corr(mats, mask_iu=None):
    """맵 리스트의 쌍별 Pearson (상삼각 or 벡터)."""
    vs = []
    for m in mats:
        v = np.asarray(m, np.float64)
        v = v[mask_iu] if mask_iu is not None else v.ravel()
        vs.append(v)
    K = len(vs); out = []
    for i in range(K):
        for j in range(i + 1, K):
            out.append(float(np.corrcoef(vs[i], vs[j])[0, 1]))
    return out


def stability_metrics(results, maps):
    thetas = np.array([r["theta"] for r in results])   # [K, 5]
    names = ["beta_L", "lam_L", "beta_F", "lam_F", "se_star"]
    cv = {}
    for i, nm in enumerate(names):
        mu, sd = float(thetas[:, i].mean()), float(thetas[:, i].std(ddof=1))
        cv[nm] = dict(mean=mu, sd=sd, cv=sd / abs(mu) if abs(mu) > 1e-12 else float("inf"))
    pc = dict(
        wLRE=pairwise_corr([m["wLRE"] for m in maps], IU),
        wFFI=pairwise_corr([m["wFFI"] for m in maps], IU),
        c_ei=pairwise_corr([m["c_ei"] for m in maps]),
        fc_sim=pairwise_corr([m["fc_sim"] for m in maps], IU),
    )
    summ = {k: dict(median=float(np.median(v)), min=float(np.min(v)), mean=float(np.mean(v)))
            for k, v in pc.items()}
    corrs = [r["corr"] for r in results]
    return dict(theta_cv=cv, map_pairwise=summ,
                fc_corr=dict(mean=float(np.mean(corrs)), sd=float(np.std(corrs, ddof=1)),
                             min=float(np.min(corrs)), max=float(np.max(corrs))))


# ────────────────────────── phases ──────────────────────────
def phase2(ctx, a_):
    batch_eval, nshape, ndtype, _ = make_batch_eval_noise(ctx, a_.chunk)
    results, maps = [], []
    t0 = time.perf_counter()
    for k in range(1, a_.K + 1):
        t = time.perf_counter()
        r, m = run_fit(ctx, batch_eval, nshape, ndtype, k, a_)
        results.append(r); maps.append(m)
        print(f"[②] fit {k:>2}/{a_.K}  corr={r['corr']:+.4f}  θ={np.round(r['theta'],3)}"
              f"  ({time.perf_counter()-t:.0f}s)", flush=True)
        np.savez(f"{QC_OUT}/idx{a_.idx}_phase2_maps.npz",
                 **{f"fit{i}_{kk}": v for i, mm in enumerate(maps) for kk, v in mm.items()})
        with open(f"{QC_OUT}/idx{a_.idx}_phase2.json", "w") as f:
            json.dump(dict(idx=a_.idx, K=len(results), results=results,
                           metrics=stability_metrics(results, maps) if len(results) >= 2 else {},
                           elapsed_s=round(time.perf_counter() - t0, 1)), f, indent=1)
    print("[②] 완료:", json.dumps(stability_metrics(results, maps), indent=1)[:800], flush=True)


def phase3(ctx, a_):
    """θ_true = X0+se_target (집단 평균 초기값 = 그럴듯한 참값). 합성 FC → 재-fit."""
    batch_eval, nshape, ndtype, tgt_holder = make_batch_eval_noise(ctx, a_.chunk)
    se_target = ctx["se_target"]
    theta_true = list(X0) + [se_target]
    cei_t, wl_t, wf_t = make_bundle(ctx, theta_true)
    noise_syn = np.asarray(gen_noise(999_999, nshape, ndtype))
    nb = np.broadcast_to(noise_syn, (a_.popsize,) + noise_syn.shape)
    cs, _, ses, fcs = batch_eval(np.stack([cei_t] * a_.popsize), np.stack([wl_t] * a_.popsize),
                                 np.stack([wf_t] * a_.popsize), nb)
    fc_syn = np.asarray(fcs[0], np.float64)
    print(f"[③] 합성 FC 생성: θ_true corr(emp)={float(cs[0]):+.3f} S_e={float(ses[0]):.3f}", flush=True)

    # ctx 를 합성 타깃으로 교체 (make_bundle 이 ctx['fc']/'resid' 를 씀 → 절차 일관)
    Wsc = np.asarray(ctx["Wn"], np.float64)
    Wsc = Wsc / max(float(Wsc.max()), 1e-12)
    m = ctx["mask"].astype(bool)[IU]
    b_ab, a_ab = np.polyfit(Wsc[IU][m], fc_syn[IU][m], 1)
    ctx["fc"] = fc_syn
    ctx["resid"] = fc_syn - (a_ab + b_ab * Wsc)
    ctx["t32"] = np.nan_to_num(np.asarray(fc_syn, np.float32))
    tgt_holder["tgt"] = jnp.asarray(ctx["t32"])   # 컴파일된 평가기의 타깃 교체

    results, maps = [], []
    for k in range(1, a_.K3 + 1):
        r, mm = run_fit(ctx, batch_eval, nshape, ndtype, 100 + k, a_)
        results.append(r); maps.append(mm)
        print(f"[③] refit {k}/{a_.K3}  corr(FC_syn)={r['corr']:+.4f}  θ={np.round(r['theta'],3)}", flush=True)

    thetas = np.array([r["theta"] for r in results])
    tt = np.array(theta_true)
    names = ["beta_L", "lam_L", "beta_F", "lam_F", "se_star"]
    rec = {nm: dict(true=float(tt[i]), est_mean=float(thetas[:, i].mean()),
                    est_sd=float(thetas[:, i].std(ddof=1)),
                    abs_err=float(np.mean(np.abs(thetas[:, i] - tt[i]))))
           for i, nm in enumerate(names)}
    map_rec = dict(
        wLRE=[float(np.corrcoef(wl_t[IU], mm["wLRE"][IU])[0, 1]) for mm in maps],
        wFFI=[float(np.corrcoef(wf_t[IU], mm["wFFI"][IU])[0, 1]) for mm in maps],
        c_ei=[float(np.corrcoef(cei_t, mm["c_ei"])[0, 1]) for mm in maps],
    )
    out = dict(idx=a_.idx, theta_true=theta_true, K3=a_.K3, results=results,
               recovery=rec, map_recovery={k: dict(mean=float(np.mean(v)), min=float(np.min(v)))
                                           for k, v in map_rec.items()})
    with open(f"{QC_OUT}/idx{a_.idx}_phase3.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez(f"{QC_OUT}/idx{a_.idx}_phase3_maps.npz", fc_syn=fc_syn.astype(np.float32),
             wLRE_true=wl_t, wFFI_true=wf_t, c_ei_true=cei_t)
    print("[③] 완료:", json.dumps(dict(recovery=rec, map_recovery=out["map_recovery"]), indent=1), flush=True)


def phase4(ctx, a_):
    """②의 각 해 → 동일 DBS(STN_L, STD) → ΔFC 쌍별 corr. part4 재사용."""
    import types as _types  # noqa
    from model import build_network  # 재-warmup 없이 ctx 것 재사용
    from part4_dbs import run_dbs_stimulation

    with open(f"{QC_OUT}/idx{a_.idx}_phase2.json") as f:
        p2 = json.load(f)
    thetas = [r["theta"] for r in p2["results"]][: a_.K]
    cfg, data = ctx["cfg"], ctx["data"]
    cfg.dbs_std_enable = True
    cfg.dbs_pre_stimulation_duration_ms = float(a_.dbs_window_tr) * cfg.bold_repetition_time_ms
    cfg.dbs_stimulation_duration_ms = float(a_.dbs_window_tr) * cfg.bold_repetition_time_ms
    cfg.dbs_save_neural_csv = False
    targets_all = dict(cfg.dbs_target_regions)
    tsel = {a_.dbs_target: targets_all[a_.dbs_target]}
    cfg.dbs_target_regions = tsel
    print(f"[④] DBS 타깃 {tsel}, STD=on, 창 {a_.dbs_window_tr}TR×2, K={len(thetas)}", flush=True)

    dfcs = []
    for k, th in enumerate(thetas, 1):
        cei, wl, wf = make_bundle(ctx, list(th))
        bundle = StateBundle.from_warmup(
            warmup_result=ctx["warmup_result"], bold_monitor_template=ctx["bold_monitor"],
            initial_params=ParamSet(c_ei=cei, wLRE=wl, wFFI=wf, c_ei_frozen=True),
            internal_state=ctx["internal"], delay_history=ctx["delay_hist"], stage="qc4")
        outdir = f"{QC_OUT}/idx{a_.idx}_dbs/fit{k:02d}"
        cfg.dbs_output_base_dir = outdir
        os.makedirs(outdir, exist_ok=True)
        t = time.perf_counter()
        run_dbs_stimulation(network=ctx["network"], bundle_in=bundle, cfg=cfg, data=data)
        # part4 저장물에서 ΔFC 로드 (pandas csv: 인덱스+헤더)
        import glob as _g
        import pandas as _pd
        fcsv = _g.glob(f"{outdir}/{a_.dbs_target}/**/fc_diff_during_minus_pre.csv",
                       recursive=True)
        if not fcsv:
            raise SystemExit(f"④ fit{k}: fc_diff_during_minus_pre.csv 없음 — {outdir}")
        d = _pd.read_csv(fcsv[0], index_col=0).to_numpy(dtype=np.float64)
        dfcs.append(d)
        print(f"[④] fit {k}/{len(thetas)} ΔFC 확보 ({time.perf_counter()-t:.0f}s)", flush=True)

    pcs = pairwise_corr(dfcs, IU)
    out = dict(idx=a_.idx, K=len(dfcs), target=a_.dbs_target,
               R_DBS_median=float(np.median(pcs)), R_DBS_mean=float(np.mean(pcs)),
               R_DBS_min=float(np.min(pcs)), pairwise=pcs)
    with open(f"{QC_OUT}/idx{a_.idx}_phase4.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez(f"{QC_OUT}/idx{a_.idx}_phase4_dfc.npz", **{f"dfc{k}": d for k, d in enumerate(dfcs)})
    print(f"[④] 완료: R_DBS median={out['R_DBS_median']:+.3f} min={out['R_DBS_min']:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=4)
    ap.add_argument("--phase", type=int, required=True, choices=(2, 3, 4))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--K3", type=int, default=5)
    ap.add_argument("--noise-level", type=float, default=0.02)
    ap.add_argument("--window-tr", type=int, default=240)
    ap.add_argument("--maxfev", type=int, default=64)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--se-w", type=float, default=2.0)
    ap.add_argument("--se-tol", type=float, default=0.03)
    ap.add_argument("--corr-w", type=float, default=0.7)
    ap.add_argument("--rmse-w", type=float, default=0.3)
    ap.add_argument("--sc-from", type=int, default=None)
    ap.add_argument("--dbs-target", type=str, default="STN_L")
    ap.add_argument("--dbs-window-tr", type=int, default=240)
    a_ = ap.parse_args()
    os.makedirs(QC_OUT, exist_ok=True)

    t0 = time.perf_counter()
    ctx = build_ctx(a_)
    print(f"[QC] ctx 준비 {time.perf_counter()-t0:.0f}s  n={ctx['n']}", flush=True)
    {2: phase2, 3: phase3, 4: phase4}[a_.phase](ctx, a_)


if __name__ == "__main__":
    main()
