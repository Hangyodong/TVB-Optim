"""vta.py — Model B 공간 가지: canonical 전극 + 해석적 point-source VTA + spillover w (§5-§14).

VTA = {r : |E(r;A)| ≥ E_th(PW)},  |E(r)| = I/(4πσr²) (등방 point source)
E_th(PW) = E_rh·(1 + chronaxie/PW)  (strength-duration)
→ 반경 r = sqrt(I / (4πσ·E_th))  (구형 VTA — 문서 §6 isotropic baseline 과 일관)

w_t = 1,  w_i = min(1, |VTA∩ROI_i| / |VTA∩ROI_t|),  비겹침 0  (§9)

아틀라스: DK+PD25 82라벨 NIfTI(MNI152NLin6 2mm). node_index = label_id - 1 (로더에서 검증).
ponytail: FEM/이방성/전극 형상 대신 point-source 구 — 문서 §6 "canonical" 취지.
σ·E_rh·chronaxie 가 보정 knob (r(2mA,60µs)≈2.5mm 재현 기본값).
"""
import os
import numpy as np
import nibabel as nib
from scipy import ndimage


def _keep_largest_component(vol):
    """라벨별 최대 연결성분만 유지 — 2mm NN 리샘플링이 만든 고립 섬(예: precuneus 4복셀이
    STN 옆에 출현) 제거. §9 공식은 불변; 아틀라스 정제만."""
    out = vol.copy()
    for lab in np.unique(vol):
        if lab <= 0:
            continue
        cc, n = ndimage.label(vol == lab)
        if n <= 1:
            continue
        # 각 연결성분의 크기 계산
        try:
            sizes = ndimage.sum_labels(np.ones_like(cc), cc, index=np.arange(1, n + 1))
        except AttributeError:
            # 구버전 scipy: ndimage.sum 사용
            sizes = np.array([ndimage.sum(np.ones_like(cc), cc, i) for i in range(1, n + 1)])
        keep = 1 + int(np.argmax(sizes))
        out[(vol == lab) & (cc != keep)] = 0
    return out


def load_atlas(nii_path):
    """→ (라벨 volume int32 [i,j,k], affine float64 4×4)"""
    img = nib.load(nii_path)
    vol = np.asarray(img.dataobj).astype(np.int32)
    vol = _keep_largest_component(vol)
    return vol, np.asarray(img.affine, dtype=np.float64)


def roi_centroid_mm(vol, affine, label_id):
    """라벨 복셀 무게중심 → world(mm) 좌표 (§5)"""
    ijk = np.argwhere(vol == label_id)
    if ijk.size == 0:
        raise ValueError(f"atlas 에 label {label_id} 없음")
    return (affine @ np.r_[ijk.mean(axis=0), 1.0])[:3]


def vta_radius_mm(amp_ma, pw_us, sigma_s_per_m, e_rheobase_v_per_m, chronaxie_us):
    """§6-§7: A[mA], PW[µs] → 구형 VTA 반경[mm]"""
    e_th = e_rheobase_v_per_m * (1.0 + chronaxie_us / pw_us)          # V/m
    r_m = np.sqrt(amp_ma * 1e-3 / (4.0 * np.pi * sigma_s_per_m * e_th))
    return float(1000.0 * r_m)


def contact_center_mm(centroid_mm, contact_index, tilt_deg, spacing_mm):
    """active contact 중심(§5: contact_index=1 이 centroid). canonical 직선 lead,
    시상면 내 lateral tilt 만 반영 — 반구별 ±x. contact 는 lead 축을 따라 spacing 간격."""
    th = np.deg2rad(tilt_deg)
    sign = 1.0 if centroid_mm[0] >= 0 else -1.0
    axis = np.array([sign * np.sin(th), 0.0, np.cos(th)])
    return np.asarray(centroid_mm, dtype=np.float64) + (contact_index - 1) * spacing_mm * axis


def spillover_weights(vol, affine, target_label_id, center_mm, radius_mm):
    """§9 spatial rule → w shape (vol.max(),), index = label_id-1 = node_index."""
    n_labels = int(vol.max())
    ijk = np.argwhere(vol > 0)
    xyz = (affine @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T
    d2 = np.einsum("ij,ij->i", xyz - center_mm, xyz - center_mm)
    lab_in = vol[tuple(ijk[d2 <= radius_mm ** 2].T)]
    counts = np.bincount(lab_in, minlength=n_labels + 1).astype(np.float64)  # index=label_id
    tgt = counts[target_label_id]
    if tgt <= 0:
        # 반경이 너무 작거나 contact shift 로 target 을 벗어남 → target-only 자극으로 폴백
        print(f"[VTA] 경고: |VTA∩target|=0 (label {target_label_id}, r={radius_mm:.2f}mm) → target-only w")
        w = np.zeros(n_labels, dtype=np.float64)
        w[target_label_id - 1] = 1.0
        return w
    w = np.minimum(1.0, counts[1:] / tgt)
    w[target_label_id - 1] = 1.0
    return w


def modelb_weights(atlas_nii, target_node_index, amp_ma, pw_us, cfg, cache_dir=None):
    """(target, A, PW, contact) → w [n_labels] (§24 캐시: vta_w_<tag>.npz)."""
    # 보정 knob(σ/E_rh/chronaxie/tilt/spacing) 전부 태그에 포함 — 누락 시 stale 캐시 무경고 로드
    # (과거 noise-σ cache_tag 사고와 동일 계열).
    tag = (f"n{target_node_index}_a{amp_ma:g}_pw{pw_us:g}_c{cfg.vta_contact_index}"
           f"_s{cfg.vta_sigma_s_per_m:g}_e{cfg.vta_e_rheobase_v_per_m:g}"
           f"_ch{cfg.vta_chronaxie_us:g}_t{cfg.vta_lead_tilt_deg:g}"
           f"_sp{cfg.vta_contact_spacing_mm:g}")
    fp = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fp = os.path.join(cache_dir, f"vta_w_{tag}.npz")
        if os.path.exists(fp):
            return np.load(fp)["w"]
    vol, affine = load_atlas(atlas_nii)
    label_id = int(target_node_index) + 1              # 로더가 identity 매핑을 assert 함
    centroid = roi_centroid_mm(vol, affine, label_id)
    center = contact_center_mm(centroid, cfg.vta_contact_index,
                               cfg.vta_lead_tilt_deg, cfg.vta_contact_spacing_mm)
    r = vta_radius_mm(amp_ma, pw_us, cfg.vta_sigma_s_per_m,
                      cfg.vta_e_rheobase_v_per_m, cfg.vta_chronaxie_us)
    w = spillover_weights(vol, affine, label_id, center, r)
    nz = {int(i): round(float(w[i]), 3) for i in np.nonzero(w)[0]}
    print(f"[VTA] node{target_node_index}(label{label_id}) A={amp_ma}mA PW={pw_us}µs "
          f"r={r:.2f}mm center=({center[0]:+.1f},{center[1]:+.1f},{center[2]:+.1f}) → w>0: {nz}")
    if fp:
        np.savez(fp, w=w, radius_mm=r, center_mm=center)
    return w
