#!/usr/bin/env python3
"""dbs_modelb_sweep.py — Model B(VTA spillover) DBS (target × A × PW × f) sweep 러너.

Part3(grad) 캐시를 직접 로드해 Part4 DBS 를 실행한다(dbs_from_cache.py 의 Model B 판).
FIC/EIB/Part3 를 재계산하지 않는다. grad pkl 에 최적화 params(c_ei/wLRE/wFFI)가 다 있으므로
DBS 는 그것만 있으면 된다. 네트워크(+warmup)는 1회만 짓고 (target,A,PW,f) 조합마다
자극(VTA spillover w × TM q)만 다시 계산한다.

사용:
    python dbs_modelb_sweep.py --subject-idx 0 --targets STN_L,STN_R \
        --amps-ma 1.0,2.0,3.0 --pws-us 60 --freqs 130 [--dry-run]
선행: python main_dkpd25.py --subject-idx 0   (part1-3 grad 캐시 생성)

출력: <outroot>/mb_a<A>_pw<PW>_f<F>/<target>/model_b/fc_pre_during_diff.png (+csv)
--dry-run: 시뮬레이션 없이 (target,A,PW,f) 조합별 VTA spillover w 벡터만 계산·출력한다
           (grad 캐시·네트워크 빌드 불필요 — part1-3 미실행 상태에서도 조합·w 검증 가능).
"""
import argparse
import gc
import glob
import os
import pickle
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("PART3_REMAT_SCAN", "1")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pathlib as _pl

import main_dkpd25 as M
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: None    # part4 는 savefig 로 직접 저장 → show no-op

from data_loader import load_data
from model import build_network
from pipeline_contracts import StateBundle
from part4_dbs import run_dbs_stimulation


def _parse_list(s):
    return [float(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser(description="Model B DBS sweep: target × A × PW × f (파이프라인 재계산 없음)")
    ap.add_argument("--subject-idx", dest="subject_idx", type=int, default=0)
    ap.add_argument("--noise-level", dest="noise_level", type=float, default=0.02)
    ap.add_argument("--amps-ma", dest="amps_ma", type=str, default="2.0",
                    help="진폭 리스트 mA(콤마). 예: 1.0,2.0,3.0")
    ap.add_argument("--pws-us", dest="pws_us", type=str, default="60",
                    help="pulse width 리스트 µs(콤마). 예: 60,90")
    ap.add_argument("--freqs", dest="freqs", type=str, default="130",
                    help="주파수 리스트 Hz(콤마). dt=1ms 양자화: 20,125,143,167,200")
    ap.add_argument("--targets", dest="targets", type=str, default="",
                    help="타깃 필터(콤마, 라벨명). 비우면 전 타깃. 예: STN_L,STN_R")
    ap.add_argument("--dbs-outroot", dest="dbs_outroot", type=str, default=None,
                    help="sweep 출력 루트(미지정 시 <sub>/dbs_modelb_sweep). 조합별 mb_a<A>_pw<PW>_f<F>/ 하위폴더")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="시뮬레이션 없이 조합·VTA spillover w 벡터만 계산·출력 (grad 캐시/네트워크 빌드 불필요)")
    ap.add_argument("--neural-stride", dest="neural_stride", type=int, default=4,
                    help="타깃 노드 neural CSV 다운샘플 stride (dt1ms 기준 4=250Hz). 0=저장 안 함")
    ap.add_argument("--seed", type=int, default=None,
                    help="노이즈 시드(cfg.bundle_rng_seed). 진폭 경향이 단일 실현의 산물이 "
                         "아닌지 보려면 시드 여러 개로 돌려 비교할 것")
    a = ap.parse_args()

    amps = _parse_list(a.amps_ma)
    pws = _parse_list(a.pws_us)
    freqs = _parse_list(a.freqs)

    p = M.prepare_pd_data(a.subject_idx, a.noise_level)
    cfg = M.make_config(p, a.subject_idx)
    cfg.dbs_save_neural_csv = a.neural_stride > 0
    cfg.dbs_neural_csv_stride = max(1, a.neural_stride)
    if a.seed is not None:
        cfg.bundle_rng_seed = a.seed
        print(f"[dbs_modelb_sweep] bundle_rng_seed = {a.seed}")
    outroot = a.dbs_outroot or os.path.join(p["out_dir"], "dbs_modelb_sweep")

    M._FIG["dir"] = _pl.Path(p["out_dir"]) / "figures"
    M._FIG["dir"].mkdir(parents=True, exist_ok=True)

    # cfg 는 make_config 직후 이미 전 타깃 dict 를 갖고 있다(grad 캐시와 무관) → dry-run 도 안전.
    all_targets = dict(cfg.dbs_target_regions)

    tot = len(amps) * len(pws) * len(freqs)
    print(f"[dbs_modelb_sweep] sweep: A{amps} × PW{pws} × f{freqs} = {tot} 조합 → {outroot}")

    # 실 sweep(dry-run 아님)만 데이터/네트워크/grad 캐시가 필요하다.
    # dry-run 은 여기 블록 전체를 건너뛰므로 JAX network 빌드·grad pkl 로드가 발생하지 않는다.
    if not a.dry_run:
        data = load_data(cfg)
        np.fill_diagonal(data["fc_target"], 0.0)

        # 네트워크(+warmup) 1회
        network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

        # grad 캐시 로드 = 최적화 상태
        gfiles = sorted(glob.glob(os.path.join(data["cache_dir"], "grad_*.pkl")), key=os.path.getmtime)
        if not gfiles:
            sys.exit(f"[dbs_modelb_sweep] grad 캐시 없음: {data['cache_dir']}  "
                      f"→ 먼저: python main_dkpd25.py --subject-idx {a.subject_idx}")
        with open(gfiles[-1], "rb") as fh:
            grad = pickle.load(fh)
        bundle_grad = StateBundle.from_dict(grad["bundle"])
        corr = grad["bundle"].get("metadata", {}).get("post_grad_fc_corr")
        print(f"[dbs_modelb_sweep] grad 로드: {os.path.basename(gfiles[-1])}  post_grad_corr={corr}")

        import jax

    n = 0
    for amp in amps:
        for pw in pws:
            for freq in freqs:
                n += 1
                combo = f"mb_a{amp:g}_pw{pw:g}_f{freq:g}"
                combo_dir = os.path.join(outroot, combo)
                cfg.dbs_modelb_enable = True
                cfg.dbs_amplitude_ma = amp
                cfg.dbs_pulse_width_us = pw
                cfg.dbs_stimulation_frequency_hz = freq
                cfg.dbs_output_base_dir = combo_dir
                if a.targets:
                    keep = set(a.targets.split(","))
                    cfg.dbs_target_regions = {k: v for k, v in all_targets.items() if k in keep}
                    assert cfg.dbs_target_regions, f"--targets {a.targets} 매칭 없음: {list(all_targets)}"
                n_t = len(cfg.dbs_target_regions)
                if a.dry_run:
                    from vta import modelb_weights
                    for tl, ti in cfg.dbs_target_regions.items():
                        w = modelb_weights(cfg.vta_atlas_nii, ti, amp, pw, cfg,
                                           cache_dir=os.path.join(combo_dir, "vta_cache"))
                    print(f"[dry-run] {combo}: targets={list(cfg.dbs_target_regions)}")
                    continue
                # resume: 타깃 수만큼 model_b fc diff 파일이 있으면 skip. 경로에 "model_b" 를
                # 명시해 세므로 legacy dbs_from_cache 스윕(true_p_t/ 하위)과 절대 안 섞인다.
                done = len(glob.glob(os.path.join(combo_dir, "*", "model_b", "fc_pre_during_diff.png")))
                if done >= n_t:
                    print(f"[skip] {combo} (완료 {done}/{n_t})")
                    continue
                print(f"\n[{n}/{tot}] {combo}")
                run_dbs_stimulation(network=network, bundle_in=bundle_grad, cfg=cfg, data=data)
                # 조합 간 JIT 캐시(+stim array 상수) 해제 → GPU 메모리 누적/OOM 방지
                jax.clear_caches()
                gc.collect()

    if a.dry_run:
        print(f"\n[dbs_modelb_sweep] dry-run 완료 ({n}/{tot} 조합)")
    else:
        print(f"\n[dbs_modelb_sweep] sweep 완료 ({n}/{tot} 조합) → {outroot}")


if __name__ == "__main__":
    main()
