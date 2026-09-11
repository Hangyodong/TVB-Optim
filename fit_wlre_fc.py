#!/usr/bin/env python3
"""fit_wlre_fc.py — 최적화된 wLRE/wFFI 를 FC_emp 로 회귀 (선형 / 2차 / 3차).

대상: final .mat (FC_AAL_all_163_final.mat) 기준 현 idx 4~12 의 9명.
가중치는 plot_wlre_fc.load_ref (현 inputs/FC.csv 해시와 일치하는 grad 캐시 중
post_grad_fc_corr 최대) 에서 가져온다. 회귀 표본은 SC>0 상삼각 엣지.

실행: python3 fit_wlre_fc.py
"""
import numpy as np

from plot_wlre_fc import IU, load_ref

# 현 idx (final .mat, PD 187) → subject. 구 idx10~13 이 −1 시프트된 결과.
SUBS = {4: "100878", 5: "100889", 6: "100905", 7: "100952", 8: "101025",
        9: "101070", 10: "101124", 11: "101146", 12: "101174"}


def polyfit_r2(x, y, deg):
    c = np.polyfit(x, y, deg)
    r2 = 1 - np.sum((y - np.polyval(c, x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return c, r2


def main():
    rows, pool = [], {"x": [], "wLRE": [], "wFFI": []}
    for idx, sub in SUBS.items():
        ref, corr, tag = load_ref(sub)
        if ref is None:
            print(f"idx {idx} ({sub}): 해시 일치 캐시 없음 → skip")
            continue
        W = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/weight.csv", delimiter=",")
        FCe = np.loadtxt(f"output_ppmi_pd/{sub}/inputs/FC.csv", delimiter=",")
        np.fill_diagonal(FCe, 0.0)
        m = (W > 0)[IU]
        x = FCe[IU][m]
        r = {"idx": idx, "sub": sub, "corr": corr, "n": len(x), "tag": tag,
             "sum_mean": float((ref["wLRE"] + ref["wFFI"])[IU][m].mean()),
             "sum_std": float((ref["wLRE"] + ref["wFFI"])[IU][m].std())}
        pool["x"].append(x)
        for nm in ("wLRE", "wFFI"):
            y = ref[nm][IU][m]
            pool[nm].append(y)
            for d in (1, 2, 3):
                c, r2 = polyfit_r2(x, y, d)
                assert d == 1 or r2 >= r[f"{nm}_r2_{d-1}"] - 1e-9, "고차 R² 역행"
                r[f"{nm}_c_{d}"], r[f"{nm}_r2_{d}"] = c, r2
            r[f"{nm}_clip0"] = float((y <= 1e-6).mean())
        rows.append(r)

    for nm in ("wLRE", "wFFI"):
        print(f"\n=== {nm} = f(FC_emp)  (SC>0 상삼각) " + "=" * 30)
        print(f"{'idx':>3} {'sub':>7} {'n':>6}  "
              f"{'선형 a':>8} {'b':>8} {'R2':>7} | "
              f"{'2차 c0':>8} {'c1':>8} {'c2':>8} {'R2':>7} | {'3차 R2':>7} | {'y=0':>6}")
        for r in rows:
            a1 = r[f"{nm}_c_1"]; a2 = r[f"{nm}_c_2"]
            print(f"{r['idx']:>3} {r['sub']:>7} {r['n']:>6}  "
                  f"{a1[1]:>8.4f} {a1[0]:>8.4f} {r[f'{nm}_r2_1']:>7.4f} | "
                  f"{a2[2]:>8.4f} {a2[1]:>8.4f} {a2[0]:>8.4f} {r[f'{nm}_r2_2']:>7.4f} | "
                  f"{r[f'{nm}_r2_3']:>7.4f} | {r[f'{nm}_clip0']*100:>5.1f}%")
        b = np.array([r[f"{nm}_c_1"][0] for r in rows])
        a = np.array([r[f"{nm}_c_1"][1] for r in rows])
        print(f"    평균±SD  a = {a.mean():+.4f}±{a.std():.4f} (CV {abs(a.std()/a.mean())*100:.1f}%)"
              f"   b = {b.mean():+.4f}±{b.std():.4f} (CV {abs(b.std()/b.mean())*100:.1f}%)")

        xs = np.concatenate(pool["x"]); ys = np.concatenate(pool[nm])
        for d in (1, 2, 3):
            c, r2 = polyfit_r2(xs, ys, d)
            print(f"    pooled(n={len(xs):,}) deg{d}: "
                  + " ".join(f"{v:+.4f}" for v in c[::-1]) + f"   R2={r2:.4f}")

    # subject 마다 기울기가 달라 pooled 회귀는 R² 가 떨어진다 → 기울기를 subject 수준
    # FC 통계로 예측할 수 있는지. 되면 미실행 subject 도 시뮬 없이 가중치를 세울 수 있다.
    print("\n=== 기울기 b 를 subject 수준 FC 통계로 설명 " + "=" * 20)
    fcm = np.array([x.mean() for x in pool["x"]])
    fcs = np.array([x.std() for x in pool["x"]])
    for nm in ("wLRE", "wFFI"):
        b = np.array([r[f"{nm}_c_1"][0] for r in rows])
        for lab, v in (("mean(FC)", fcm), ("SD(FC)", fcs)):
            print(f"  b({nm}) ~ {lab:>9}: r = {np.corrcoef(v, b)[0,1]:+.4f}")

    print("\n=== wLRE + wFFI (엣지별 합) " + "=" * 30)
    for r in rows:
        print(f"idx {r['idx']:>2} ({r['sub']}): {r['sum_mean']:.4f} ± {r['sum_std']:.4f}"
              f"    post_grad corr={r['corr']:.4f}  cache={r['tag']}")


if __name__ == "__main__":
    main()
