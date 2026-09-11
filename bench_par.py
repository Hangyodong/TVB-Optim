#!/usr/bin/env python3
"""bench_par.py — GPU 동시 실행 시 프로세스당 속도 저하 측정.

FIC 내부 루프와 같은 경로(고정 길이 네트워크 시뮬)를 JIT 이후 3회 반복해 프로세스당
소요를 잰다. 여러 개를 동시에 띄우면 경합 비용이 그대로 드러난다.

실행: python3 bench_par.py <tag>
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import numpy as np, jax
import main_ppmi_pd as M
M.MAT_PATH = "data/AALv3/FC_AAL_ComBat_all_163.mat"
from data_loader import load_data
from model import build_network
from pipeline_contracts import (StateBundle, ParamSet, capture_internal_state,
                                capture_network_delay_history)
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Heun

TAG = sys.argv[1] if len(sys.argv) > 1 else "x"
IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 4
t0 = time.perf_counter()
p = M.prepare_pd_data(IDX, 0.02)
cfg = M.make_config(p, IDX, use_delay=True)
cfg.cache_version = f"{cfg.cache_version}_bench"
data = load_data(cfg)
network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)
mask, cap, n = data["sc_mask"], cfg.connectivity_weight_max, data["n_nodes"]
ps = ParamSet(c_ei=np.ones(n, np.float32), wLRE=np.ones((n, n), np.float32),
              wFFI=np.ones((n, n), np.float32), c_ei_frozen=False).sanitize(mask, cap)
b = StateBundle.from_warmup(warmup_result=warmup_result, bold_monitor_template=bold_monitor,
                            initial_params=ps, internal_state=capture_internal_state(initial_state),
                            delay_history=capture_network_delay_history(network), stage="warmup")
t_setup = time.perf_counter() - t0

solver = BoundedSolver(Heun(), low=0.0, high=1.0)
times = []
for k in range(20):
    t = time.perf_counter()
    model, st = b.to_tvb_state(network, solver, t1=30_000, dt=cfg.integration_dt_ms)
    out = np.asarray(model(st).data)
    dt = time.perf_counter() - t
    times.append(dt)
    print(f"[{TAG}] iter{k} {dt:.2f}s  shape={out.shape}", flush=True)
print(f"RESULT {TAG} setup={t_setup:.1f} jit={times[0]:.2f} steady={np.mean(times[5:]):.3f} "
      f"all={','.join(f'{v:.3f}' for v in times)}", flush=True)
