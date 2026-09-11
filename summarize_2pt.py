#!/usr/bin/env python3
"""summarize_2pt.py — twopoint_fic.py 결과를 full pipeline 과 비교해 표로 출력.

읽는 것: output_ppmi_pd/_2pt/idx<N>.json (twopoint_fic 산출)
        output_ppmi_pd/_logs/2pt_idx<N>.log (가중치 클리핑 비율)
실행: python3 summarize_2pt.py
"""
import glob
import json
import os
import re

import numpy as np

OUT, LOG = "output_ppmi_pd/_2pt", "output_ppmi_pd/_logs"
# 비교용: 기존 최적화(Part3) 가중치의 회귀 계수 — clip 안 걸린 엣지 기준
OLS = {4: (0.9518, 2.5367), 5: (0.9737, 2.5242), 6: (0.9361, 2.2805),
       7: (1.0027, 2.5920), 8: (0.9577, 3.1788), 9: (0.9824, 2.3645),
       10: (0.9570, 1.5215), 11: (0.9579, 2.7780), 12: (0.9561, 2.9144)}
CLIP_OPT = {4: 5.09, 5: 4.99, 6: 4.74, 7: 4.05, 8: 7.57, 9: 6.34, 10: 5.63,
            11: 6.16, 12: 7.49}   # 최적화 결과의 wFFI=0 비율(%)


def clip_pct(idx):
    p = f"{LOG}/2pt_idx{idx}.log"
    if not os.path.exists(p):
        return None, None
    t = open(p, errors="replace").read()
    m = re.search(r"wLRE [\d.]+ \(0 인 엣지 ([\d.]+)%\)\s+wFFI [\d.]+ \(0 ([\d.]+)%\)", t)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def main():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/idx*.json"),
                    key=lambda s: int(re.search(r"idx(\d+)", s).group(1))):
        d = json.load(open(f))
        d["clip_lre"], d["clip_ffi"] = clip_pct(d["idx"])
        rows.append(d)
    if not rows:
        print("결과 없음 — 아직 완료된 idx 가 없다.")
        return

    print("## 계수 비교 (2점 연립방정식 vs 기존 최적화 회귀)\n")
    print("| idx | sub | 2점 a | 2점 b | 회귀 a | 회귀 b | b 오차 | 앵커 같은 엣지 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        oa, ob = OLS[r["idx"]]
        print(f"| {r['idx']} | {r['sub']} | {r['a']:.3f} | {r['b']:.3f} | {oa:.3f} | {ob:.3f} "
              f"| {(r['b']/ob-1)*100:+.1f}% | {'O' if r['anchor']['same_edge'] else 'X'} |")

    # 대조군: 같은 final FC 타깃 위에서 최적화 가중치를 재채점한 값 (있으면 그걸 기준으로)
    for r in rows:
        cf = f"{OUT}/ctrl_idx{r['idx']}.json"
        r["ctrl"] = json.load(open(cf))["corr"] if os.path.exists(cf) else None
        r["base"] = r["ctrl"] if r["ctrl"] is not None else r["full_pipeline_corr"]

    print("\n## FC 재현 (2점+FIC vs 최적화 가중치, 같은 final FC 타깃)\n")
    print("| idx | sub | 2점 corr | 최적화(final FC) | 차이 | 구 FC 저장값 | SC>0 | SC==0 | "
          "rmse | c_ei 평균 | wFFI=0 (2점) | wFFI=0 (최적화) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cs = f"{r['ctrl']:.4f}" if r["ctrl"] is not None else "—"
        print(f"| {r['idx']} | {r['sub']} | {r['corr']:+.4f} | {cs} "
              f"| {r['corr']-r['base']:+.4f} | {r['full_pipeline_corr']:.4f} | {r['corr_sc']:+.4f} "
              f"| {r['corr_nosc']:+.4f} | {r['rmse']:.4f} | {r['c_ei_mean']:.3f} "
              f"| {r['clip_ffi'] if r['clip_ffi'] is not None else '—'}% | {CLIP_OPT[r['idx']]}% |")

    d = np.array([r["corr"] - r["base"] for r in rows])
    c = np.array([r["corr"] for r in rows])
    p = np.array([r["base"] for r in rows])
    print(f"\n평균: 2점 {c.mean():+.4f}  full pipeline {p.mean():.4f}  차이 {d.mean():+.4f} "
          f"(범위 {d.min():+.4f} ~ {d.max():+.4f})   n={len(rows)}")
    print(f"full pipeline 대비 회복률: {(c/p).mean()*100:.1f}%  "
          f"(2점이 더 나은 subject {int((d>0).sum())}/{len(rows)}명)")


if __name__ == "__main__":
    main()
