#!/usr/bin/env python3
"""
main_dkpd25.py — EI Tuning 파이프라인 (DK+PD25 82노드 PD: FC_raw/SC_weight/SC_length from .mat)

Usage:
    python3 main_dkpd25.py                          # subject idx=0, noise=0.02
    python3 main_dkpd25.py --subject-idx 5          # N번째 subject (0-based, 0~178)
    python3 main_dkpd25.py --noise-level 0.0        # 무잡음
    python3 main_dkpd25.py --fic-only               # FIC만 실행
    python3 main_dkpd25.py --enable-sigma           # Part3.5 σ 튜닝 켜기 (기본 skip)
    python3 main_dkpd25.py --enable-dbs             # Part4 DBS 켜기 (기본 skip)

Part 1(FIC) / 2(EIB) / 3(Gradient) 는 기본 실행, --skip-{fic,eib,gradient} 로 끈다.
Part 3.5(σ 튜닝) / 4(DBS) 는 기본 skip, --enable-{sigma,dbs} 로 켠다.
(--enable-sigma 는 Part3 출력에 의존 → --skip-gradient 와 함께 쓰면 자동 skip)

data/DK+PD25/FC_DKPD25_82_ppmi_all_nomed_qc.mat 의 data struct array(238명 = PD179+HC59)에서
group==PD 필터 후 subject_idx 번째를 추출. 필드: subject(id), group(PD/HC),
FC_raw(Pearson raw corr, ComBat 없음), SC_weight(raw streamline), SC_length(mm),
TR(2.5), labels(1..82 identity).
행렬은 82×82 (DesikanCortexPD25: DK cortex 66 + PD25 subcortex 16; FC 결측 노드 없음 → DROP 불필요).
노드 순서는 .mat 내장 region_names 줄 순서 = .mat 노드 순서(labels 1..82 identity 로 검증).

cortex 66 / subcortex 16 으로 분리된다(data_loader._DKPD25_SUBCORTEX_LABELS 기준):
subcortex = STN/GPe/GPi/Putamen/Caudate/Thalamus/SN/Red_N 좌우 8쌍.

출력은 output_dkpd25/<sub_num>/ 아래 inputs/ figures/ cache/ (+DBS 시 dbs_analysis/) 로 저장된다.
DBS 타깃은 STN_L/R + GPe_L/R + GPi_L/R — PD25 subcortex 라벨에 GPe/GPi 구분이 있어 그대로 쓴다.
"""
import argparse
import os
import sys

# ── 0. JAX 환경변수 (노트북 Cell 1과 동일) ───────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR",   "platform")
# Part3/3.5 full backprop 은 remat 없으면 GPU OOM. scan-body checkpointing 기본 ON.
os.environ.setdefault("PART3_REMAT_SCAN", "1")

import jax
jax.config.update("jax_enable_x64", False)
print(f"backend : {jax.default_backend()}")
print(f"devices : {jax.devices()}")
print(f"jax_enable_x64 : {jax.config.jax_enable_x64}")

# ── 1. 임포트 ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 화면 없이 저장만

import matplotlib.pyplot as _plt
import pathlib as _pl, re as _re

# figure 출력 폴더는 main()에서 sub_num 확정 후 설정 (output_dkpd25/<sub_num>/figures).
_FIG = {"dir": None}
_fig_counter = {"n": 0}

def _slugify(text):
    return _re.sub(r"[^\w가-힣\-]", "_", text.strip())[:60] or "figure"

def _get_label(fig, idx):
    try:
        t = fig._suptitle.get_text()
        if t: return _slugify(t)
    except Exception: pass
    for ax in fig.axes:
        try:
            t = ax.get_title()
            if t: return _slugify(t)
        except Exception: pass
    return f"figure_{idx:03d}"

if not hasattr(_plt, "_main_original_show"):
    _plt._main_original_show = _plt.show

def _patched_show(*args, **kwargs):
    out_dir = _FIG["dir"] or _pl.Path(".")   # main()서 설정 전엔 cwd fallback
    for fn in _plt.get_fignums():
        fig = _plt.figure(fn)
        _fig_counter["n"] += 1
        n = _fig_counter["n"]
        label = _get_label(fig, n)
        out = out_dir / f"{n:03d}_{label}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"  [fig] {out}")
    _plt._main_original_show(*args, **kwargs)

_plt.show = _patched_show


from config              import Config
from data_loader         import derive_cortex_subcortex_indices, load_data
from model               import build_network
from part1_fic           import run_fic
from part2_eib           import run_eib
from part3_gradient      import run_gradient_optimization, run_sigma_optimization
from part4_dbs           import run_dbs_stimulation
from pipeline_contracts  import (
    ParamSet,
    StateBundle,
    capture_internal_state,
    capture_network_delay_history,
)

# ── DK+PD25 .mat 경로 ────────────────────────────────────────────────────
MAT_PATH   = "data/DK+PD25/FC_DKPD25_82_ppmi_all_nomed_qc.mat"
# struct array (1,238) = PD 179 + HC 59 (nomed_qc, dwi_qc 전원 pass).
# 필드: subject/group/TR(2.5)/nvol(240)/FC_raw(82×82 Pearson)/FC_raw_z/SC_weight/SC_length
#       /updrs3(+tremor/rigidity/brady). ComBat 없음 → FC_raw 를 타깃으로 쓴다.
MAT_KEY    = "data"
GROUP_FILTER = "PD"              # PD 코호트만 (subject_idx = PD 필터 후 0~178)
N_NODES_RAW = 82                 # DK cortex 66 + PD25 subcortex 16
DROP_LABELS = ()                 # FC 결측 노드 없음
N_NODES    = N_NODES_RAW - len(DROP_LABELS)
OUTPUT_DIR = "output_dkpd25"
# 라벨은 .mat region_names 에 내장 (별도 txt 없음). atlas label_id = node_index+1 (아래 assert).
ATLAS_NII  = "data/DK+PD25/DesikanCortexPD25_space-MNI152NLin6_res-2x2x2.nii.gz"

# Part4 DBS 타깃 (PD25 subcortex 라벨 이름, lower-case 매칭)
DBS_TARGET_LABELS = {
    "STN_L": "l_subthalamic_nucleus",
    "STN_R": "r_subthalamic_nucleus",
    "GPi_L": "l_globus_pallidus_interna",
    "GPi_R": "r_globus_pallidus_interna",
    "GPe_L": "l_globus_pallidus_externa",
    "GPe_R": "r_globus_pallidus_externa",
}


def _dbs_targets_from_labels(labels: list) -> dict:
    """region label 이름 → 0-based 노드 인덱스 dict. 못 찾은 타깃은 경고 후 제외."""
    lower = [l.strip().lower() for l in labels]
    targets = {}
    for key, name in DBS_TARGET_LABELS.items():
        try:
            targets[key] = lower.index(name)
        except ValueError:
            print(f"  [DBS] 라벨 없음 → 타깃 제외: {key} ('{name}')")
    return targets


# ── 2. 인자 파싱 ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="EI Tuning Pipeline (DK+PD25 82 nodes from .mat)")
    parser.add_argument("--subject-idx", dest="subject_idx", type=int, default=0,
                        help="사용할 subject 0-based index (0 ~ N-1)")
    parser.add_argument("--noise-level", dest="noise_level", type=float, default=0.02,
                        help="가산 잡음 σ (additive_noise_sigma). 0 이면 무잡음")
    parser.add_argument("--skip-fic",      action="store_true", help="FIC 건너뜀")
    parser.add_argument("--skip-eib",      action="store_true", help="EIB 건너뜀")
    parser.add_argument("--skip-gradient", action="store_true", help="Gradient 건너뜀")
    # Part3.5 / Part4 는 기본 skip. --enable-* 로 켠다 (--skip-* 은 하위호환용 no-op 기본값).
    parser.add_argument("--enable-sigma", dest="skip_sigma", action="store_false", default=True,
                        help="Part3.5 σ 튜닝 활성 (기본: skip)")
    parser.add_argument("--skip-sigma",   dest="skip_sigma", action="store_true",
                        help="Part3.5 σ 튜닝 건너뜀 (기본값)")
    parser.add_argument("--enable-dbs",   dest="skip_dbs", action="store_false", default=True,
                        help="Part4 DBS 활성 (기본: skip). 타깃은 라벨에서 자동 산출")
    parser.add_argument("--skip-dbs",     dest="skip_dbs", action="store_true",
                        help="Part4 DBS 건너뜀 (기본값)")
    parser.add_argument("--fic-only",      action="store_true", help="FIC만 실행")
    # ── Part 3.5 — per-node noise σ 튜닝 ──────────────────────────────────
    parser.add_argument("--sigma-shared",  dest="sigma_per_node", action="store_false",
                        default=True, help="공유 스칼라 σ 1개(파일럿). 기본은 per-node")
    parser.add_argument("--sigma-max",     dest="sigma_max", type=float, default=0.10,
                        help="σ BoundedParameter 상한 (기본 0.10)")
    parser.add_argument("--sigma-l2",      dest="sigma_l2", type=float, default=0.01,
                        help="σ spread L2 가중치 (기본 0.01)")
    parser.add_argument("--freeze-c-ei", dest="freeze_c_ei", default=None, action="store_true",
                        help="FIC 후 c_ei 동결. 미지정 시 Config 기본값")
    parser.add_argument("--no-freeze-c-ei", dest="freeze_c_ei", action="store_false",
                        help="FIC 후 c_ei 학습 허용")
    parser.add_argument("--fc-eval-n-seeds", dest="fc_eval_n_seeds", type=int, default=None,
                        help="FC eval을 N개 noise seed로 평균(분산↓). 미지정=Config 기본 1. 예: 5")
    parser.add_argument("--sc-norm", dest="sc_norm", type=str, default=None, choices=["log1p", "max", "log1pm"],
                        help="SC 정규화. max=원본 w/max(안정), log1p=과흥분, log1pm=log1p+입력보존(약한edge복원·안전). 미지정=Config 기본")
    parser.add_argument("--no-delay", dest="use_delay", action="store_false", default=True,
                        help="tract_length 기반 전도지연을 끈다(delay-free ODE/SDE). 기본=지연 반영(DDE)")
    parser.add_argument("--conduction-speed", dest="conduction_speed", type=float, default=None,
                        help="전도속도 mm/ms (기본 3.0). delay=SC_length/speed")
    parser.add_argument("--grad-steps", dest="grad_steps", type=int, default=None,
                        help="Part3 gradient step 수 (기본 250). 100 이후 수렴 → 100 권장")
    # ── Part4 DBS 오버라이드 (--enable-dbs 와 함께) ──────────────────────
    parser.add_argument("--dbs-amplitude", dest="dbs_amplitude", type=float, default=None,
                        help="DBS 진폭(모델 단위, config 기본 1.0). 예: 1.0='1uA', 2.0='2uA'")
    parser.add_argument("--dbs-freq", dest="dbs_freq", type=float, default=None,
                        help="DBS 주파수 Hz(config 기본 130). dt=1ms 양자화: 143→142.9, 130→125")
    parser.add_argument("--dbs-outdir", dest="dbs_outdir", type=str, default=None,
                        help="DBS 출력 폴더(미지정 시 <sub>/dbs_analysis)")
    return parser.parse_args()


# ── 3. PD .mat → CSV 추출 ────────────────────────────────────────────────
def _load_region_labels() -> list:
    """.mat region_names → 이름 리스트(82). labels==1..82 (node_index=label_id-1) 를 검증한다
    — vta.py 의 아틀라스 매핑이 이 가정에 의존한다."""
    import numpy as _np
    import scipy.io as _sio
    m = _sio.loadmat(MAT_PATH)
    names = [str(_np.asarray(x).ravel()[0]).strip() for x in m["region_names"].ravel()]
    labs = _np.asarray(m["labels"]).ravel().astype(int)
    assert list(labs) == list(range(1, len(names) + 1)), \
        "atlas label_id != node_index+1 — VTA 매핑 가정 붕괴"
    return names


def _load_pd_entries():
    """data(238건 = PD 179 + HC 59)에서 group==GROUP_FILTER 인 entry 를 파일 순서대로 반환(PD 179).
    subject_idx 가 이 리스트의 인덱스(0~178). HC 는 인터리브돼 있어 필터로 뽑는다."""
    import numpy as _np
    import scipy.io as _sio
    m = _sio.loadmat(MAT_PATH)
    if MAT_KEY not in m:
        raise KeyError(f"{MAT_KEY} not in {MAT_PATH} (keys={[k for k in m if not k.startswith('__')]})")
    arr = m[MAT_KEY].ravel()
    entries = [r for r in arr if str(_np.asarray(r["group"]).ravel()[0]) == GROUP_FILTER]
    if not entries:
        raise RuntimeError(f"group=='{GROUP_FILTER}' entry 없음 (총 {arr.size}건)")
    return entries


def prepare_pd_data(subject_idx: int, noise_level: float) -> dict:
    """data(FC_DKPD25_82_ppmi_all_nomed_qc.mat) 의 PD subject_idx 번째 데이터(FC_raw/SC_weight/SC_length)를
    output_dkpd25/<sub_num>/inputs/ 에 CSV 로 추출하고 make_config 용 dict 를 반환한다."""
    import numpy as _np

    entries = _load_pd_entries()
    N = len(entries)
    assert 0 <= subject_idx < N, f"subject_idx must be 0..{N-1}, got {subject_idx} ({GROUP_FILTER} {N}명)"

    s = entries[subject_idx]
    # subject 필드는 BIDS 스타일 "sub-100001" (AAL163 의 순수 숫자 문자열과 다름) → 접두어 제거 후 int.
    sub_num = int(str(_np.asarray(s["subject"]).ravel()[0]).strip().removeprefix("sub-"))
    # 단일 파일(data)에서 FC_raw + SC_weight + SC_length 를 모두 읽는다.
    # nan_to_num 은 DROP_LABELS 제거 뒤에 한다 — 먼저 0 으로 덮으면 결측을 진단할 수 없다.
    SC  = _np.asarray(s["SC_weight"], dtype=_np.float64)   # streamline count
    FC  = _np.asarray(s["FC_raw"],    dtype=_np.float64)   # Pearson (ComBat 없음)
    LEN = _np.asarray(s["SC_length"], dtype=_np.float64)   # mm
    n_raw = SC.shape[0]
    assert SC.shape == FC.shape == LEN.shape == (n_raw, n_raw), (
        f"matrix shape mismatch: SC{SC.shape} FC{FC.shape} LEN{LEN.shape}")
    assert n_raw == N_NODES_RAW, f"n_nodes {n_raw} != expected {N_NODES_RAW} (DK+PD25)"

    # ── DROP_LABELS 노드 제거 (라벨 이름 기준 → SC/FC/LEN/labels 동시에) ──────
    labels_raw = _load_region_labels()
    assert len(labels_raw) == n_raw, f"label count {len(labels_raw)} != n_nodes {n_raw}"
    keep = [i for i, l in enumerate(labels_raw) if l not in DROP_LABELS]
    dropped = [(i, labels_raw[i]) for i in range(n_raw) if labels_raw[i] in DROP_LABELS]
    assert len(dropped) == len(DROP_LABELS), (
        f"DROP_LABELS {DROP_LABELS} 중 라벨에서 못 찾은 것이 있다: 찾은 것={dropped}")
    _ix = _np.ix_(keep, keep)
    SC, FC, LEN = SC[_ix], FC[_ix], LEN[_ix]
    labels = [labels_raw[i] for i in keep]
    n = len(keep)
    assert n == N_NODES, f"n_nodes {n} != expected {N_NODES} (DK+PD25, {len(DROP_LABELS)}개 제거 후)"

    # 제거 후에도 남은 FC 결측 경고 (subject 에 따라 특정 노드가 전체 NaN 인 경우가 있다)
    _off = ~_np.eye(n, dtype=bool)
    _n_nan = int(_np.isnan(FC[_off]).sum())
    if _n_nan:
        _dead = [labels[i] for i in range(n) if _np.isnan(FC[i][_off[i]]).all()]
        print(f"  [WARN] FC off-diag NaN {_n_nan}개 → 0.0 으로 대체 (상관 0 타깃이 됨)"
              + (f"  전체결측 노드: {_dead}" if _dead else ""))

    SC  = _np.nan_to_num(SC,  nan=0.0)
    FC  = _np.nan_to_num(FC,  nan=0.0)
    LEN = _np.nan_to_num(LEN, nan=0.0)

    out_dir = os.path.abspath(os.path.join(OUTPUT_DIR, str(sub_num)))
    in_dir  = os.path.join(out_dir, "inputs")
    os.makedirs(in_dir, exist_ok=True)

    sc_csv  = os.path.join(in_dir, "weight.csv")
    len_csv = os.path.join(in_dir, "tract_length.csv")
    fc_csv  = os.path.join(in_dir, "FC.csv")
    reg_txt = os.path.join(in_dir, "region_labels.txt")

    _np.savetxt(sc_csv,  SC,  delimiter=",")
    _np.savetxt(len_csv, LEN, delimiter=",")
    _np.savetxt(fc_csv,  FC,  delimiter=",")

    # labels 는 위에서 DROP 반영된 것(len=n)을 그대로 쓴다.
    with open(reg_txt, "w") as fh:
        for lab in labels:
            fh.write(lab + "\n")
    # cortex/subcortex 판정은 data_loader 와 반드시 같은 기준을 써야 한다(따로 세면 어긋남).
    ctx_idx, sub_idx = derive_cortex_subcortex_indices(labels)
    n_ctx = len(ctx_idx)
    # DBS 타깃은 제거 후 labels 기준 → 삭제 노드 뒤 인덱스가 자동으로 당겨진다.
    dbs_targets = _dbs_targets_from_labels(labels)

    print(f"Dataset: DK+PD25 TR_2.5_PD  (subject idx={subject_idx}/{N}, sub_num={sub_num})")
    print(f"  dropped -> {[f'{l}(raw node {i})' for i, l in dropped]}  → {n_raw} - {len(dropped)} = {n} nodes")
    print(f"  n_nodes={n}  cortex={n_ctx}  subcortex={len(sub_idx)}")
    print(f"  SC     -> {sc_csv}   (max={SC.max():.0f})")
    print(f"  length -> {len_csv}  (max={LEN.max():.1f}mm)")
    print(f"  FC     -> {fc_csv}   (range=[{FC.min():.2f},{FC.max():.2f}])")
    print(f"  labels -> {reg_txt}  (src=.mat region_names)")
    print(f"  out_dir-> {out_dir}")
    print(f"  DBS targets (0-based) -> {dbs_targets}")

    return dict(
        sc_csv=sc_csv, length_csv=len_csv, fc_csv=fc_csv, region_txt=reg_txt,
        sub_num=sub_num, out_dir=out_dir,
        dbs_targets=dbs_targets,
        tract_conduction_speed=3.0,
        additive_noise_sigma=noise_level,
        wc_c_ei_init=1.0,   # RWW 표준 J_i (10.0이면 S_e가 0 근처로 짓눌려 FIC 불가)
        fic_target_firing_rate_hz=2.0,   # 진단용 Hz (FIC 제어는 fic_target_se=S_e gating)
        # Bold HRF (human 계열, mouse-32s 커널 차용 — main_ppmi 동일)
        bold_hrf_k1=5.6, bold_hrf_V0=0.02, bold_hrf_tau_s=0.8, bold_hrf_tau_f=0.4,
        bold_hrf_scaling=1.0 / 3.0, bold_hrf_duration_ms=32_000.0,
    )


# ── 4. Config 생성 (TR=2.5 스케일, main_ppmi 값 차용) ────────────────────
def make_config(p: dict, subject_idx: int, use_delay: bool = True,
                conduction_speed: float | None = None) -> Config:
    cfg = Config(
        # ── 데이터 경로 ──────────────────────────────────────────
        region_txt   = p["region_txt"],
        sc_csv       = p["sc_csv"],
        length_csv   = p["length_csv"],
        fc_csv       = p["fc_csv"],
        # ── 캐시: output_dkpd25/<sub_num>/cache 로 라우팅 ────────
        # cache_run_label 이 절대경로면 data_loader 의 set_cache_path(join(cache_root, label/tag))
        # 에서 절대경로가 root 를 무시 → 캐시가 out_dir/cache/<cache_tag>/ 에 저장됨.
        cache_run_label = os.path.join(p["out_dir"], "cache"),
        # dkpd25 태그로 기존 schaefer/aal163 캐시(v_pd100_*/v_pdaal163_*)와 분리
        cache_version   = f"v_dkpd25_tr25_s{subject_idx}",   # DK+PD25 FC_raw (ComBat 없음)

        # ── 시뮬레이션 공통 ──────────────────────────────────────
        integration_dt_ms                   = 1.0,
        warmup_duration_ms                  = 600_000,   # 600s
        bold_hrf_k1          = p["bold_hrf_k1"],
        bold_hrf_V0          = p["bold_hrf_V0"],
        bold_hrf_tau_s       = p["bold_hrf_tau_s"],
        bold_hrf_tau_f       = p["bold_hrf_tau_f"],
        bold_hrf_scaling     = p["bold_hrf_scaling"],
        bold_hrf_duration_ms = p["bold_hrf_duration_ms"],

        bold_repetition_time_ms             = 2500.0,   # PD 실측 TR=2.5s
        tract_conduction_speed              = p["tract_conduction_speed"],
        additive_noise_sigma                = p["additive_noise_sigma"],

        # ── Part 1 — FIC ─────────────────────────────────────────
        fic_target_firing_rate_hz           = p["fic_target_firing_rate_hz"],
        fic_learning_rate                   = 0.5,     # EI_Tuning 참조값 (init c_ei=1.0 기준)
        fic_max_iterations                  = 2000,
        fic_early_stop_patience             = 500,
        fic_step_duration_ms                = 2_500,   # =1 TR @2.5s (total_tr=int(2500/2500)=1)
        fic_step_skip_tr                    = 0,
        fic_posthoc_duration_ms             = 600_000, # 240 TR × 2.5s = 600s
        fic_posthoc_skip_tr                 = 48,       # 20% of 240
        # 원본식(EI_Tuning cell36-37): c_ei freeze 안 함. Part3 backprop 이 c_ei 도 최적화하되
        # loss 의 activity 항(optimizer_activity_weight, 아래)이 mean_S_e→0.25 로 묶어
        # c_ei 를 생리적 balance 에 soft 고정 → degeneracy drift(0.2대) 방지. 원본 안전장치 복원.
        freeze_c_ei_after_fic               = False,

        # ── Part 2 — EIB ─────────────────────────────────────────
        eib_max_iterations                  = 10000,
        eib_internal_fic_learning_rate      = 0.1,     # 원본 cell28 (0.05→0.1)
        eib_max_weight_learning_rate        = 0.005,   # 원본 cell28 (0.002→0.005)
        eib_bold_window_samples             = 240,     # 600s @ TR=2.5s
        eib_snapshot_save_interval          = 50,
        connectivity_weight_max             = 10.0,    # 원본 unbounded(inf) 취지 — 2.0 캡 해제(max-norm 약한 base서 wLRE가 FC 만들게). ※재실행 후 S_e 재포화 안 하는지 확인
        eib_posthoc_duration_ms             = 600_000, # 240 TR × 2.5s = 600s
        eib_posthoc_skip_tr                 = 48,

        # ── Part 3 — Full Gradient ───────────────────────────────
        optimizer_learning_rate             = 0.0001,
        optimizer_max_steps                 = 250,
        optimizer_chunk_steps               = 10,
        optimizer_bold_window_tr            = 240,     # EIB(240)와 동일 측정선 = 600s
        optimizer_bold_skip_tr              = 48,      # 20% of 240, 유효 192 TR

        # ── EIB score 계산용 ──────────────────────────────────────
        pd_fit_region_count                 = 14,
        full_brain_fc_loss_weight           = 1.00,
        correlation_loss_weight             = 0.80,
        rmse_loss_weight                    = 0.20,

        # ── Patch 9: Gradient 3-term loss weights ────────────────
        optimizer_global_corr_weight        = 0.80,
        optimizer_nodewise_corr_weight      = 0.40,    # part3 미사용
        optimizer_rmse_weight               = 0.20,
        optimizer_rmse_block                = True,
        optimizer_activity_weight           = 1.0,   # 원본 cell36 activity_loss(=fc_loss 동일 가중). c_ei를 S_e 0.25에 묶어 drift 차단

        # ── 캐시 경량화 ──────────────────────────────────────────
        # neural trace 50-stride 다운샘플(plot 전용) → post_*_neural 557MB→~11MB.
        # delay_history 는 pipeline_contracts 가 자동으로 max_delay tail 만 저장(557MB→<1MB).
        neural_cache_stride                 = 50,

        # ── Subcortex FC fitting emphasis ────────────────────────
        fc_block_share_cortex               = 0.50,
        fc_block_share_cross                = 0.40,
        fc_block_share_subsub               = 0.10,
        corr_block_weight_cc                = 0.40,
        corr_block_weight_cross             = 0.40,
        corr_block_weight_subsub            = 0.20,

        # ── Part 4 — DBS (기본 skip, --enable-dbs 로 활성) ────────
        # 타깃은 라벨 이름 매칭으로 산출 (config.py 기본값은 mouse 인덱스라 사용 금지).
        dbs_target_regions                  = p["dbs_targets"],
        # 미설정 시 config 기본 "./dbs_analysis" → cwd 오염. sub_num 폴더로 라우팅.
        dbs_output_base_dir                 = os.path.join(p["out_dir"], "dbs_analysis"),
        # pre(transient/baseline)·during 자극 지속시간 = 시뮬레이션 시간(240 TR × 2500ms = 600s).
        # config 기본(pre 720s / during 60s)을 덮어씀 → during-FC 를 full 창(240 TR)으로 측정
        # (구 60s=16 TR rank-deficient 문제 회피). 자극 조건(진폭·주파수·biphasic)은 config 기본 유지.
        dbs_pre_stimulation_duration_ms     = 600_000,
        dbs_stimulation_duration_ms         = 600_000,

        wc_c_ei_init                        = p["wc_c_ei_init"],
        vta_atlas_nii                       = ATLAS_NII,
    )
    if conduction_speed is not None:
        cfg.tract_conduction_speed = conduction_speed
    cfg.use_delay = use_delay
    if use_delay:
        # delay-free 캐시와 섞이면 안 된다(속도가 바뀌면 delay 도 바뀜) → cache_version 분리.
        # main() 뿐 아니라 dbs_from_cache 같은 caller 도 같은 태그를 얻어야 캐시를 찾는다.
        cfg.cache_version = f"{cfg.cache_version}_delay{cfg.tract_conduction_speed:g}"
    return cfg


# ── 5. 메인 실행 ─────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print(f"  EI Tuning Pipeline — DK+PD25 TR_2.5_PD ({N_NODES} nodes) "
          f"(subject_idx={args.subject_idx})")
    print("=" * 60)

    if args.fic_only:
        args.skip_eib = args.skip_gradient = args.skip_sigma = args.skip_dbs = True
    if args.skip_gradient:
        args.skip_sigma = True   # σ 튜닝은 Part3 출력 의존

    # PD .mat → CSV 추출
    p = prepare_pd_data(args.subject_idx, args.noise_level)

    # figure 출력 폴더를 sub_num 폴더로 라우팅 (load_data 의 data-matrix figure부터 캡처)
    _FIG["dir"] = _pl.Path(p["out_dir"]) / "figures"
    _FIG["dir"].mkdir(parents=True, exist_ok=True)
    print(f"[FIG] 출력 폴더: {_FIG['dir']}")

    # Config
    cfg = make_config(p, args.subject_idx, args.use_delay, args.conduction_speed)
    if args.freeze_c_ei is not None:
        cfg.freeze_c_ei_after_fic = args.freeze_c_ei
    print(f"[cfg] freeze_c_ei_after_fic={cfg.freeze_c_ei_after_fic}")
    if args.fc_eval_n_seeds is not None:
        cfg.fc_eval_n_seeds = args.fc_eval_n_seeds
    print(f"[cfg] fc_eval_n_seeds={cfg.fc_eval_n_seeds}")
    if args.sc_norm is not None:
        cfg.sc_norm = args.sc_norm
    print(f"[cfg] sc_norm={cfg.sc_norm}")
    if args.grad_steps is not None:
        cfg.optimizer_max_steps = args.grad_steps
        print(f"[cfg] optimizer_max_steps={cfg.optimizer_max_steps}")
    print(f"[cfg] use_delay={cfg.use_delay}  "
          f"tract_conduction_speed={cfg.tract_conduction_speed} mm/ms")
    # DBS 오버라이드 (--enable-dbs 와 함께 진폭/주파수/출력폴더 지정)
    if args.dbs_amplitude is not None:
        cfg.dbs_pulse_amplitude = args.dbs_amplitude
    if args.dbs_freq is not None:
        cfg.dbs_stimulation_frequency_hz = args.dbs_freq
    if args.dbs_outdir is not None:
        cfg.dbs_output_base_dir = args.dbs_outdir
    cfg.print_summary()

    # Data loading
    print("\n[0] Loading data...")
    data = load_data(cfg)

    # fc_target 대각 0
    import numpy as _np_diag
    _np_diag.fill_diagonal(data["fc_target"], 0.0)
    print("[DATA] fc_target diagonal → 0")

    # Network build + warmup
    print("\n[0] Building network + warmup...")
    network, initial_state, bold_monitor, warmup_result = build_network(cfg, data)

    initial_params = ParamSet.default(
        data["n_nodes"], c_ei_init=cfg.wc_c_ei_init
    ).sanitize(data["sc_mask"], cfg.connectivity_weight_max)

    bundle_init = StateBundle.from_warmup(
        warmup_result         = warmup_result,
        bold_monitor_template = bold_monitor,
        initial_params        = initial_params,
        internal_state        = capture_internal_state(initial_state),
        delay_history         = capture_network_delay_history(network),
        stage                 = "warmup",
    )
    print(bundle_init)

    # Part 1: FIC
    if not args.skip_fic:
        print("\n[1] Running FIC...")
        bundle_fic = run_fic(network=network, bundle_in=bundle_init, cfg=cfg, data=data)
        print(f"[FIC] c_ei_frozen={bundle_fic.params.c_ei_frozen}  "
              f"mean c_ei={bundle_fic.params.c_ei.mean():.4f}")
        print(bundle_fic)
    else:
        print("\n[1] FIC skipped (--skip-fic)")
        bundle_fic = bundle_init

    # Part 2: EIB
    if not args.skip_eib:
        print("\n[2] Running EIB...")
        bundle_eib = run_eib(network=network, bundle_in=bundle_fic, cfg=cfg, data=data)
        print(f"[EIB] stage={bundle_eib.stage}  c_ei_frozen={bundle_eib.params.c_ei_frozen}")
        print(bundle_eib)
    else:
        print("\n[2] EIB skipped (--skip-eib)")
        bundle_eib = bundle_fic

    # Part 3: Full Gradient
    if not args.skip_gradient:
        print("\n[3] Running Gradient Optimization...")
        bundle_grad = run_gradient_optimization(
            network=network, bundle_in=bundle_eib, warmup_bundle=bundle_init,
            cfg=cfg, data=data,
        )
        print(f"[Part3] stage={bundle_grad.stage}  c_ei_frozen={bundle_grad.params.c_ei_frozen}")
        print(bundle_grad)
    else:
        print("\n[3] Gradient skipped (--skip-gradient)")
        bundle_grad = bundle_eib

    # Part 3.5: per-node noise σ 튜닝
    if not args.skip_sigma:
        print("\n[3.5] Running per-node noise σ optimization...")
        bundle_sigma = run_sigma_optimization(
            network=network, bundle_in=bundle_grad, cfg=cfg, data=data,
            sigma_per_node=args.sigma_per_node, sigma_max=args.sigma_max,
            sigma_l2_weight=args.sigma_l2,
        )
        meta = bundle_sigma.metadata
        print(f"[Part3.5] stage={bundle_sigma.stage}  "
              f"post_sigma_fc_corr={meta.get('post_sigma_fc_corr')}  "
              f"post_sigma_fc_rmse={meta.get('post_sigma_fc_rmse')}")
        print(bundle_sigma)
        bundle_grad = bundle_sigma
    else:
        print("\n[3.5] σ optimization skipped (기본값 — 켜려면 --enable-sigma)")

    # Part 4: DBS (기본 skip, --enable-dbs 로 활성)
    if not args.skip_dbs:
        if not cfg.dbs_target_regions:
            raise SystemExit(
                "[4] --enable-dbs 인데 DBS 타깃이 비었다. "
                f"라벨(.mat region_names)에 "
                f"{list(DBS_TARGET_LABELS.values())} 이름이 있는지 확인할 것.")
        print(f"\n[4] Running DBS Stimulation... targets={cfg.dbs_target_regions}")
        run_dbs_stimulation(network=network, bundle_in=bundle_grad, cfg=cfg, data=data)
        print(f"[4] DBS outputs → {cfg.dbs_output_base_dir}")
    else:
        print("\n[4] DBS skipped (기본값 — 켜려면 --enable-dbs)")

    print("\n" + "=" * 60)
    print(f"  Pipeline complete. → {p['out_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
