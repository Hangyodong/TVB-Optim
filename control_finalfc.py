#!/usr/bin/env python3
"""control_finalfc.py — 기존 최적화 가중치를 final FC 기준으로 재채점.

twopoint_fic.py 결과를 "full pipeline corr"(구 FC 기준 저장값)과 바로 비교하면
데이터 교체 효과와 방법 효과가 섞인다. 여기서는 최적화된 c_ei/wLRE/wFFI 를 그대로
final .mat 파이프라인에 넣고 한 번 시뮬해서, 같은 타깃 위에서의 기준선을 만든다.

실행: python3 control_finalfc.py --idx 4
"""
import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import _settle_bundle, compute_simulated_fc
from pipeline_contracts import (
    ParamSet, StateBundle, capture_internal_state, capture_network_delay_history,
)
from tvboptim.observations.observation import fc_corr

IU = np.triu_indices(163, 1)
SUBDIR = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025",
          9: "101070", 10: "101124", 11: "101146", 12: "101174"}
BACKUP, OUT = "output_ppmi_pd/_inputs_pre_final", "output_ppmi_pd/_2pt"


def _get(o, k, d=None):
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def best_params(sub):
    """백업 FC 해시와 맞는 grad 캐시 중 post_grad_fc_corr 최대 → params 전체."""
    FC = np.loadtxt(f"{BACKUP}/{sub}/FC.csv", delimiter=",")
    hs = {hashlib.sha1(np.asarray(FC[:8, :8], t).tobytes()).hexdigest()[:10]
          for t in (np.float32, np.float64)}
    best, bc, btag = None, -9.0, ""
    for f in glob.glob(f"output_ppmi_pd/{sub}/cache/*/grad_*.pkl"):
        d = os.path.basename(os.path.dirname(f))
        if any(x in f for x in ("formulafic", "fwarm", "2x2", "oomtest", "bench", "2pt")):
            continue
        if not any(h in d for h in hs):
            continue
        with open(f, "rb") as fh:
            o = pickle.load(fh)
        b = _get(o, "bundle", o)
        c = float((_get(b, "metadata", {}) or {}).get("post_grad_fc_corr", np.nan))
        if np.isfinite(c) and c > bc:
            p = _get(b, "params")
            best = {k: np.asarray(_get(p, k)) for k in ("c_ei", "wLRE", "wFFI")}
            bc, btag = c, d
    if best is None:
        raise SystemExit(f"[{sub}] 해시 일치 grad 캐시 없음")
    return best, bc, btag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--noise-level", type=float, default=0.02)
    a_ = ap.parse_args()
    idx, sub = a_.idx, SUBDIR[a_.idx]
    os.makedirs(OUT, exist_ok=True)

    prm, old_corr, tag = best_params(sub)
    print(f"[idx {idx} / {sub}] 최적화 가중치 재사용 (캐시 {tag}, 구 FC corr {old_corr:.4f})",
          flush=True)

    p = M.prepare_pd_data(idx, a_.noise_level)
    cfg = M.make_config(p, idx, use_delay=True)
    cfg.cache_version = f"{cfg.cache_version}_ctrl"
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    tgt, mask, cap, n = data["fc_target"], data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

    ps = ParamSet(c_ei=np.asarray(prm["c_ei"], np.float32),
                  wLRE=np.asarray(prm["wLRE"], np.float32),
                  wFFI=np.asarray(prm["wFFI"], np.float32),
                  c_ei_frozen=False).sanitize(mask, cap)
    bundle = StateBundle.from_warmup(
        warmup_result=warmup_result, bold_monitor_template=bold_monitor,
        initial_params=ps, internal_state=capture_internal_state(initial_state),
        delay_history=capture_network_delay_history(network), stage="warmup")
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)
    bundle, _, _ = _settle_bundle(network, bundle, cfg, sim_duration_ms=dur,
                                  skip_tr=cfg.optimizer_bold_skip_tr, next_stage="eib")

    fc_sim = np.asarray(compute_simulated_fc(network, bundle, cfg, sim_duration_ms=dur,
                                             skip_tr=cfg.optimizer_bold_skip_tr), np.float32)
    t32 = np.nan_to_num(np.asarray(tgt, np.float32))
    mt = mask.astype(bool)[IU]
    res = dict(idx=idx, sub=sub, corr=float(fc_corr(fc_sim, t32)),
               corr_sc=float(np.corrcoef(fc_sim[IU][mt], t32[IU][mt])[0, 1]),
               corr_nosc=float(np.corrcoef(fc_sim[IU][~mt], t32[IU][~mt])[0, 1]),
               rmse=float(np.sqrt(np.mean((fc_sim[IU] - t32[IU]) ** 2))),
               old_fc_corr=old_corr, c_ei_mean=float(np.mean(prm["c_ei"])))
    with open(f"{OUT}/ctrl_idx{idx}.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"CONTROL idx={idx} sub={sub} corr={res['corr']:+.4f} "
          f"(SC>0 {res['corr_sc']:+.4f} / SC==0 {res['corr_nosc']:+.4f}) "
          f"rmse={res['rmse']:.4f}  |  구 FC 기준 {old_corr:.4f}", flush=True)


if __name__ == "__main__":
    main()
