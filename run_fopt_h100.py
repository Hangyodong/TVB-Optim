#!/usr/bin/env python3
"""run_fopt_h100.py — 전 PD subject formula_optim 병렬 런처 (H100 대상).

subject 마다 별도 프로세스로 formula_optim.py 를 실행 (JAX/CUDA 컨텍스트 격리).
score = 0.7·corr − 0.3·RMSE − S_e페널티 (기본). 완료된 subject(JSON 존재)는 스킵.
GPU 여러 장이면 CUDA_VISIBLE_DEVICES 라운드로빈 배정.

권장: 프로세스당 CPU 2코어 이상 확보 (1코어 노드에서 4병렬 → 평가 5배 감속 실측).
      H100 1장: --par 24~32 (MPS 켜면 nvidia-cuda-mps-control 로 30~40).

실행 예:
  python3 run_fopt_h100.py                      # 전 187명, GPU 자동감지, par 24
  python3 run_fopt_h100.py --par 32 --gpus 0,1  # 2장 라운드로빈
  python3 run_fopt_h100.py --idx 4-12           # 범위 지정
"""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output_ppmi_pd", "_fopt")
N_PD = 187


def parse_idx(spec):
    if spec == "all":
        return list(range(N_PD))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def detect_gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return ["0"]


def run_one(idx, gpu, a, slots=None):
    tag = f"idx{idx}" + (f"_sew{a.se_w:g}" if a.se_w > 0 else "")
    done_json = os.path.join(OUT, f"{tag}.json")
    if a.skip_done and os.path.exists(done_json):
        return idx, "skip", 0.0
    # 프로세스당 VRAM 고정 슬라이스 (실측 ~2.6GB/proc, 동적 할당은 프로세스 간 OOM 경합)
    # + CPU 스레드 캡: JAX/XLA 는 프로세스당 스레드를 ncpu 개 만들어 par 개 돌리면
    #   스레드 과다(par×ncpu)로 문맥교환 붕괴 (16코어 8proc → 평가 12s→85s 실측)
    allowed = sorted(os.sched_getaffinity(0))          # cgroup 허용 코어 실번호
    ncores = max(1, len(allowed) // max(1, a.par))
    env = dict(os.environ,
               XLA_PYTHON_CLIENT_PREALLOCATE="true",
               XLA_PYTHON_CLIENT_MEM_FRACTION=f"{a.mem_frac:.4f}",
               OMP_NUM_THREADS=str(ncores),
               OPENBLAS_NUM_THREADS=str(ncores),
               MKL_NUM_THREADS=str(ncores),
               CUDA_VISIBLE_DEVICES=gpu)
    slot = slots.get() if slots is not None else None
    cmd = [sys.executable, os.path.join(ROOT, "formula_optim.py"),
           "--idx", str(idx), "--maxfev", str(a.maxfev),
           "--corr-w", str(a.corr_w), "--rmse-w", str(a.rmse_w),
           "--se-w", str(a.se_w), "--se-tol", str(a.se_tol)]
    if a.eib_steps > 0:
        cmd += ["--eib-steps", str(a.eib_steps)]
    cores = allowed[slot * ncores:(slot + 1) * ncores] if slot is not None else []
    if cores:                                    # 코어 슬라이스 피닝 (스레드 이주 차단)
        cmd = ["taskset", "-c", ",".join(map(str, cores))] + cmd
    log = os.path.join(OUT, f"run_idx{idx}.log")
    t = time.perf_counter()
    for attempt in (1, 2):                       # 1회 재시도
        with open(log, "a") as fh:
            fh.write(f"\n=== attempt {attempt} {time.strftime('%F %T')} gpu={gpu} ===\n")
            fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        if r.returncode == 0 and os.path.exists(done_json):
            if slots is not None:
                slots.put(slot)
            return idx, "ok", time.perf_counter() - t
    if slots is not None:
        slots.put(slot)
    return idx, "FAIL", time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default="all", help='"all" | "4-12" | "3,7,20-40"')
    ap.add_argument("--par", type=int, default=8)
    ap.add_argument("--gpus", default=None, help="쉼표 구분 GPU id (기본 자동감지)")
    ap.add_argument("--maxfev", type=int, default=64)
    ap.add_argument("--corr-w", type=float, default=0.7)
    ap.add_argument("--rmse-w", type=float, default=0.3)
    ap.add_argument("--se-w", type=float, default=2.0)
    ap.add_argument("--se-tol", type=float, default=0.03)
    ap.add_argument("--eib-steps", type=int, default=0)
    ap.add_argument("--no-skip-done", dest="skip_done", action="store_false", default=True)
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help="프로세스당 GPU 메모리 비율 (0=자동: 0.85×GPU수/par)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    idxs = parse_idx(a.idx)
    gpus = a.gpus.split(",") if a.gpus else detect_gpus()
    if a.mem_frac <= 0:
        a.mem_frac = min(0.9, max(0.02, 0.85 * len(gpus) / a.par))
    ncpu = os.cpu_count() or 1
    if ncpu < 2 * a.par:
        print(f"[WARN] CPU {ncpu}코어 < 2×par({a.par}) — 호스트 병목으로 평가 감속 예상. "
              f"par {max(1, ncpu // 2)} 권장", flush=True)
    print(f"[런처] {len(idxs)}명  par={a.par}  GPU={gpus}  mem_frac={a.mem_frac:.3f}  maxfev={a.maxfev}  "
          f"score={a.corr_w}·corr−{a.rmse_w}·rmse−S_e페널티(w={a.se_w},tol={a.se_tol})", flush=True)

    t0 = time.perf_counter()
    stat = {"ok": 0, "skip": 0, "FAIL": 0}
    slots = queue.Queue()
    for k in range(a.par):
        slots.put(k)
    with ThreadPoolExecutor(max_workers=a.par) as ex:
        futs = {ex.submit(run_one, idx, gpus[i % len(gpus)], a, slots): idx
                for i, idx in enumerate(idxs)}
        for n, f in enumerate(as_completed(futs), 1):
            idx, st, el = f.result()
            stat[st] += 1
            print(f"[{n}/{len(idxs)}] idx{idx}: {st} ({el/60:.1f}분)  "
                  f"누적 {(time.perf_counter()-t0)/60:.0f}분", flush=True)

    # 요약 CSV
    rows = []
    for idx in idxs:
        tag = f"idx{idx}" + (f"_sew{a.se_w:g}" if a.se_w > 0 else "")
        p = os.path.join(OUT, f"{tag}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        rows.append(dict(idx=idx, corr=j.get("corr"), corr_720=j.get("corr_720"),
                         mean_se=j.get("mean_se"), score=j.get("score"),
                         theta=" ".join(f"{v:.3f}" for v in j.get("theta", [])),
                         elapsed_min=round(j.get("elapsed_s", 0) / 60, 1)))
    if rows:
        out_csv = os.path.join(OUT, "summary.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        cs = [r["corr"] for r in rows if r["corr"] is not None]
        print(f"\n[요약] {out_csv}  n={len(rows)}  corr 평균 {sum(cs)/len(cs):.4f} "
              f"(min {min(cs):.4f} / max {max(cs):.4f})", flush=True)
    print(f"[완료] ok={stat['ok']} skip={stat['skip']} FAIL={stat['FAIL']}  "
          f"총 {(time.perf_counter()-t0)/60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
