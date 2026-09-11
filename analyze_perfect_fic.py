#!/usr/bin/env python3
"""analyze_perfect_fic.py — run_perfect_fic.py 두 조건의 캐시 bundle 을 재시뮬해
SC>0 / SC==0 별 corr 분해 + mean_S_e 측정.

실행: python3 analyze_perfect_fic.py
"""
import glob

import numpy as np
import jax

import main_ppmi_pd as M
from data_loader import load_data
from model import build_network
from part3_gradient import compute_simulated_fc
from formula_2x2 import measure_se
from stage_trace import newest, load_bundle

IDX = 4


def main():
    p = M.prepare_pd_data(IDX, 0.02)
    cfg = M.make_config(p, IDX, use_delay=True)
    base_cv = cfg.cache_version
    data = load_data(cfg)
    np.fill_diagonal(data["fc_target"], 0.0)
    FC_emp = np.nan_to_num(np.asarray(data["fc_target"], np.float32))
    network, *_ = build_network(cfg, data)
    n = data["n_nodes"]
    iu = np.triu_indices(n, 1)
    scb = data["sc_mask"].astype(bool)
    mt = scb[iu]
    dur = int(cfg.optimizer_bold_window_tr * cfg.bold_repetition_time_ms)

    print(f"상삼각 {len(iu[0])}  SC>0 {int(mt.sum())} ({mt.mean()*100:.1f}%)  "
          f"SC==0 {int((~mt).sum())} ({(~mt).mean()*100:.1f}%)")
    print(f"\n{'조건':<22}{'전체 corr':>10}{'SC>0':>9}{'SC==0':>9}{'rmse':>9}"
          f"{'c_ei':>8}{'mean_S_e':>10}")
    print("-" * 78)

    for label, tag in (("A 원본 w + FIC", "orig"), ("B 변형 w (r=±1) + FIC", "perf")):
        f = {"orig": "output_ppmi_pd/_pfic_idx4/A_orig.pkl",
             "perf": "output_ppmi_pd/_pfic_idx4/B_perf.pkl"}[tag]
        if not glob.glob(f):
            print(f"{label:<22}  캐시 없음 — skip")
            continue
        b, _ = load_bundle(f)
        fc = np.asarray(compute_simulated_fc(network, b, cfg, sim_duration_ms=dur,
                                             skip_tr=cfg.optimizer_bold_skip_tr),
                        np.float32)
        x, y = FC_emp[iu], fc[iu]
        c_all = float(np.corrcoef(x, y)[0, 1])
        c_pos = float(np.corrcoef(x[mt], y[mt])[0, 1])
        c_zero = float(np.corrcoef(x[~mt], y[~mt])[0, 1])
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        ce = np.asarray(b.params.c_ei)
        se = measure_se(network, b, cfg)
        print(f"{label:<22}{c_all:>+10.4f}{c_pos:>+9.4f}{c_zero:>+9.4f}"
              f"{rmse:>9.4f}{ce.mean():>8.4f}{se:>10.4f}")
        jax.clear_caches()

    print("\n참고: full pipeline (원본 w + optimized c_ei) 전체 corr = 0.7524, "
          "c_ei 1.8857, mean_S_e 0.2583")


if __name__ == "__main__":
    main()
