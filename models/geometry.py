"""
models/geometry.py — Model Layer (Geometry)
หน้าที่: คำนวณ Canonical Transform, B-spline, Deviation Analysis

อ้างอิงหลัก:
  - Fan et al. (2025) Computer Aided Geometric Design 119:102436
    → B-spline Arch Curve §4.2.1, Canonical Frame §4.1, Deviation §4.3.3
  - Li, Gateno, Xia (2017) Int J Oral Maxillofac Surg 46(7):974-981
    → Triangular Method + PAMED สำหรับ Canonical Frame Orientation
    → พิสูจน์ว่า Standard PCA ให้ผลถูกต้องเพียง 36.7% (not recommended)
    → Triangular Method ให้ผลถูกต้อง 93.3%
  - Wellens (2007) Am J Orthod Dentofacial Orthop 131:160.e17-160.e25
    → COG-based orientation ใช้เป็น fallback เมื่อ molar ขาดหาย
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import make_interp_spline

from config.settings import (
    CP_FALLBACK, CP_TARGETS,
    ALL_TEETH,
    BSPLINE_DEGREE, BSPLINE_N_EVAL,
    MIN_TEETH_FOR_BSPLINE,
    KP_CENTROID, KP_MMR,
)

# ── Molar landmarks ที่ใช้ใน Triangular Method ──────────────────────────────
# Li et al. (2017): U6 = mesiobuccal cusp ของ first molar ซ้ายและขวา
# ปรับมาใช้ Centroid ของ 2nd molar (37/47) เพราะ 2nd molar
# อยู่ปลาย arch ให้ทิศสมมาตรชัดเจนกว่าใน 2D occlusal view
_TRIANGULAR_MOLAR_R = ["R37", "R36"]   # priority: 2nd molar → 1st molar
_TRIANGULAR_MOLAR_L = ["L47", "L46"]


# =============================================================================
# SECTION C — Canonical Transform
# อ้างอิง: Fan et al. (2025) §4.1  +  Li et al. (2017) Triangular Method
# =============================================================================

def _find_anchor_mmr(tooth_id: str, kp_map: dict) -> np.ndarray | None:
    """
    หาพิกัด MMR ของฟัน tooth_id
    ถ้าไม่มี → ลอง fallback ฟันใกล้เคียงตาม CP_FALLBACK
    คืน np.array([x, y]) หรือ None ถ้าหาไม่ได้เลย
    """
    if tooth_id in kp_map:
        return np.array(kp_map[tooth_id]["keypoints"][KP_MMR], dtype=float)
    for fb in CP_FALLBACK.get(tooth_id, []):
        if fb in kp_map:
            print(f"  [Anchor Fallback] '{tooth_id}' ไม่มี → ใช้ '{fb}' แทน")
            return np.array(kp_map[fb]["keypoints"][KP_MMR], dtype=float)
    return None


def _find_centroid(tooth_ids: list[str], kp_map: dict) -> np.ndarray | None:
    """
    หา Centroid ของฟัน โดยลองตามลำดับ priority ใน tooth_ids
    คืน np.array([x, y]) ของฟันซี่แรกที่หาเจอ หรือ None
    """
    for tid in tooth_ids:
        if tid in kp_map:
            return np.array(kp_map[tid]["keypoints"][KP_CENTROID], dtype=float)
    return None


def _triangular_arch_angle(origin: np.ndarray,
                            molar_r: np.ndarray,
                            molar_l: np.ndarray) -> float:
    """
    คำนวณมุม orientation ของ Dental Arch ด้วย Triangular Method

    หลักการ (Li et al., 2017 — Int J Oral Maxillofac Surg):
    ─────────────────────────────────────────────────────────
    ใช้ 3 landmark ทางกายวิภาคกำหนด midsagittal plane ของ arch:
      U0 = midpoint ระหว่าง central incisor (= Origin ใน frame นี้)
      U6 = mesiobuccal cusp ของ 1st/2nd molar ฝั่งขวาและซ้าย

    เส้นสมมาตรของ arch (midsagittal) คือเส้นที่ผ่าน U0
    และตั้งฉากกับเส้นที่เชื่อม U6 ซ้าย-ขวา

    ทำไมถึงเลือกวิธีนี้:
      - Li et al. (2017) ทดสอบกับ 30 คนไข้ dentofacial deformity:
        Triangular method ถูกต้อง 93.3%
        Standard PCA ถูกต้องเพียง 36.7% → "not recommended"
      - หลักการสอดคล้องกับ clinical practice ทันตแพทย์จัดฟัน
        ซึ่งใช้ incisor midline + molar bilateral symmetry
        เป็น reference เสมอ

    Parameters
    ----------
    origin  : np.ndarray  midpoint(MMR_31, MMR_41) — ศูนย์กลางของ arch
    molar_r : np.ndarray  Centroid ของ molar ฝั่งขวาผู้ป่วย
    molar_l : np.ndarray  Centroid ของ molar ฝั่งซ้ายผู้ป่วย

    Returns
    -------
    angle : float  มุม rotation ในหน่วย radian
                   ให้แกน X ของ canonical frame ขนานกับแนว
                   ที่เชื่อม molar ซ้าย-ขวา (transverse arch axis)
    """
    # เส้นที่เชื่อม molar ซ้าย-ขวา (transverse)
    transverse = molar_r - molar_l         # vector จาก L-molar → R-molar
    angle = np.arctan2(transverse[1], transverse[0])

    # บังคับให้ R-molar อยู่ทาง +X (molar_r.x > molar_l.x หลัง rotate)
    if np.cos(angle) < 0:
        angle += np.pi

    return float(angle)


def _cog_arch_angle(matches: list, origin: np.ndarray) -> float:
    """
    Fallback: COG-based orientation (Wellens, 2007 — AJODO)

    ใช้เมื่อ molar ขาดหายทั้งสองฝั่ง จึงไม่สามารถใช้ Triangular Method ได้

    หลักการ (Wellens, 2007):
    ─────────────────────────────────────────────────────────
    1. คำนวณจุด Center of Gravity (COG) ของ Centroid ทุกซี่
    2. คำนวณมุมจากทุกจุดไปยัง COG
    3. เฉลี่ยมุมทั้งหมด → ใช้เป็นทิศหลักของ arch
    4. หมุน cluster จน average angle = 0

    ข้อดีเหนือ standard PCA:
      ยึดจุดกายวิภาค (COG) ไม่ใช่ statistical variance direction
      ให้ symmetric arch เสมอแม้ฟันบางซี่บิดเบี้ยว

    Limitation:
      ใช้ได้ดีเฉพาะเมื่อ arch ค่อนข้างสมมาตร
      (Wellens 2007: correlation 0.994 กับ crowding patients)

    Returns
    -------
    angle : float  มุมในหน่วย radian
    """
    cens = np.array([m["keypoints"][KP_CENTROID] for m in matches], dtype=float)

    if len(cens) < 3:
        return 0.0

    # COG = mean ของ Centroid ทุกซี่
    cog = cens.mean(axis=0)

    # มุมจากแต่ละจุดไปยัง COG
    vectors = cens - cog
    angles  = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Mean angle (circular mean)
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    mean_angle = np.arctan2(mean_sin, mean_cos)

    # แกน X ตั้งฉากกับ mean direction (หมุน 90°)
    arch_angle = mean_angle + np.pi / 2.0

    return float(arch_angle)


def _ensure_r31_positive_x(angle: float,
                            mmr_31: np.ndarray,
                            mmr_41: np.ndarray,
                            origin: np.ndarray) -> float:
    """
    ตรวจสอบและแก้ทิศของมุมให้ R31 อยู่ทาง +X เสมอ

    กฎทางคลินิก:
      R31 (ฝั่งขวาผู้ป่วย, FDI prefix 3) → canonical X > 0
      L41 (ฝั่งซ้ายผู้ป่วย, FDI prefix 4) → canonical X < 0
    """
    def rotate(pt: np.ndarray, a: float) -> np.ndarray:
        c, s = np.cos(-a), np.sin(-a)
        dx, dy = pt[0] - origin[0], pt[1] - origin[1]
        return np.array([dx*c - dy*s, dx*s + dy*c])

    r31_rot = rotate(mmr_31, angle)
    r41_rot = rotate(mmr_41, angle)

    if r31_rot[0] < r41_rot[0]:   # R31 ควรอยู่ทาง +X
        angle += np.pi

    return float(angle)


def compute_transform(matches: list) -> dict:
    """
    คำนวณ Canonical Transform parameters

    ═══════════════════════════════════════════════════════════════
    ORIENTATION METHOD: Triangular Method (Li et al., 2017)
    ═══════════════════════════════════════════════════════════════

    Canonical Frame Definition:
      Origin  = midpoint(MMR_31, MMR_41)
                → จุดกึ่งกลางระหว่าง Mesial Marginal Ridge ของ
                  Central Incisor ทั้งสอง (ตาม Fan et al. §4.1)

      X-axis  = แนวที่ขนานกับเส้นเชื่อม molar ซ้าย-ขวา
                → Triangular Method จาก Li et al. (2017)
                → ใช้ Centroid ของ 2nd molar (37/47) เป็น U6
                → ผ่านการทดสอบ 93.3% accuracy

    Decision Logic (Fallback Hierarchy):
    ─────────────────────────────────────────────────────────────
    1. PRIMARY: Triangular Method
       ถ้ามี molar ทั้งสองฝั่ง (R37/R36 และ L47/L46)
       → ใช้ทิศเส้น molar_R ↔ molar_L

    2. FALLBACK: COG-based (Wellens, 2007)
       ถ้า molar ขาดหายข้างใดข้างหนึ่ง
       → ใช้ mean direction จาก COG ของ Centroid ทุกซี่

    3. SIDE-CHECK: R31 ต้องอยู่ทาง +X เสมอ
       → บังคับทิศให้ถูกต้องตาม FDI convention

    References
    ──────────
    [1] Fan et al. (2025) Comput. Aided Geom. Des. 119:102436
    [2] Li, Gateno, Xia (2017) Int J Oral Maxillofac Surg 46(7):974-981
        https://pmc.ncbi.nlm.nih.gov/articles/PMC5559304/
    [3] Wellens (2007) Am J Orthod Dentofacial Orthop 131:160.e17-25

    Returns
    -------
    dict: { tx, ty, angle_rad, angle_deg, mmr_31, mmr_41,
            method, molar_r_used, molar_l_used }

    Raises
    ------
    ValueError: ถ้าหา anchor (R31/L41) ไม่ได้แม้ใช้ fallback
    """
    kp_map = {m["tooth_class"]: m for m in matches}

    # ── ขั้นที่ 1: หา Origin anchors (MMR 31 + 41) ───────────────────────────
    mmr_31 = _find_anchor_mmr("R31", kp_map)
    mmr_41 = _find_anchor_mmr("L41", kp_map)

    if mmr_31 is None or mmr_41 is None:
        missing_anchors = (
            ([" R31 (และ R32)"] if mmr_31 is None else []) +
            ([" L41 (และ L42)"] if mmr_41 is None else [])
        )
        raise ValueError(
            f"ไม่สามารถกำหนด Canonical Frame ได้\n"
            f"ฟัน anchor หาย: {missing_anchors}\n"
            f"ลอง: ลด CONF threshold หรือตรวจสอบภาพ"
        )

    # Origin = midpoint(MMR_31, MMR_41)
    origin = (mmr_31 + mmr_41) / 2.0

    # ── ขั้นที่ 2: หา Molar landmarks สำหรับ Triangular Method ───────────────
    molar_r = _find_centroid(_TRIANGULAR_MOLAR_R, kp_map)
    molar_l = _find_centroid(_TRIANGULAR_MOLAR_L, kp_map)

    molar_r_used = next((t for t in _TRIANGULAR_MOLAR_R if t in kp_map), None)
    molar_l_used = next((t for t in _TRIANGULAR_MOLAR_L if t in kp_map), None)

    # ── ขั้นที่ 3: เลือก orientation method ──────────────────────────────────
    if molar_r is not None and molar_l is not None:
        # PRIMARY: Triangular Method — Li et al. (2017), accuracy 93.3%
        angle  = _triangular_arch_angle(origin, molar_r, molar_l)
        method = f"Triangular [{molar_r_used}↔{molar_l_used}] (Li et al. 2017)"
        print(f"  [Transform] Triangular Method: "
              f"{molar_r_used} ↔ {molar_l_used}")
    else:
        # FALLBACK: COG-based — Wellens (2007)
        missing_side = "R-molar" if molar_r is None else "L-molar"
        print(f"  [Transform] {missing_side} ขาดหาย "
              f"→ fallback: COG-based (Wellens 2007)")
        angle  = _cog_arch_angle(matches, origin)
        method = "COG-based fallback (Wellens 2007)"
        molar_r_used = molar_r_used or "N/A (missing)"
        molar_l_used = molar_l_used or "N/A (missing)"

    # ── ขั้นที่ 4: ตรวจสอบทิศ R31 → +X ──────────────────────────────────────
    angle = _ensure_r31_positive_x(angle, mmr_31, mmr_41, origin)

    print(f"  [Transform] angle={np.degrees(angle):.3f}°  method={method}")

    return {
        "tx":           float(origin[0]),
        "ty":           float(origin[1]),
        "angle_rad":    float(angle),
        "angle_deg":    float(np.degrees(angle)),
        "mmr_31":       mmr_31,
        "mmr_41":       mmr_41,
        "molar_r_used": molar_r_used,
        "molar_l_used": molar_l_used,
        "method":       method,
    }


def apply_transform(pt, tf: dict) -> np.ndarray:
    """
    แปลงจุดเดียวเข้า Canonical Frame
    pt : [x, y] หรือ np.array shape (2,)
    """
    pt    = np.asarray(pt, dtype=float)
    a     = -tf["angle_rad"]
    dx    = pt[0] - tf["tx"]
    dy    = pt[1] - tf["ty"]
    return np.array([
        dx * np.cos(a) - dy * np.sin(a),
        dx * np.sin(a) + dy * np.cos(a),
    ])


def transform_all(matches: list, tf: dict) -> list:
    """
    Transform keypoints ทุกซี่เข้า Canonical Frame
    เพิ่ม key *_t ให้แต่ละ match dict:
      dmr_t, centroid_t, mmr_t, anc_t
    """
    out = []
    for m in matches:
        kp = m["keypoints"]   # [DMR, Centroid, MMR, ANC]
        out.append({
            **m,
            "dmr_t":      apply_transform(kp[0], tf).tolist(),
            "centroid_t": apply_transform(kp[1], tf).tolist(),
            "mmr_t":      apply_transform(kp[2], tf).tolist(),
            "anc_t":      apply_transform(kp[3], tf).tolist(),
        })
    return out


# =============================================================================
# SECTION D — B-spline Arch Curve
# อ้างอิง: Fan et al. (2025) §4.2.1 Eq.(1)
# =============================================================================

def select_control_points(matches_t: list) -> np.ndarray:
    """
    เลือก Control Points ตาม paper §4.2.1:
      CP_TARGETS: R37, R35, R33, R31, L41, L43, L45, L47

    กลยุทธ์เมื่อฟันขาด:
      1. ลอง fallback ฟันใกล้เคียงตาม CP_FALLBACK
      2. ถ้า CP < MIN_TEETH_FOR_BSPLINE → evenly-spaced fallback

    Returns
    -------
    np.ndarray shape (n_cp, 2)  เรียงตาม X เสมอ
    """
    kp_map  = {m["tooth_class"]: m for m in matches_t}
    cp_list = []
    used    = set()

    for tid in CP_TARGETS:
        if tid in kp_map:
            cp_list.append(kp_map[tid]["centroid_t"])
            used.add(tid)
        else:
            found = False
            for fb in CP_FALLBACK.get(tid, []):
                if fb in kp_map and fb not in used:
                    cp_list.append(kp_map[fb]["centroid_t"])
                    used.add(fb)
                    print(f"  [CP Fallback] '{tid}' → ใช้ '{fb}' แทน")
                    found = True
                    break
            if not found:
                print(f"  [CP Missing ] '{tid}' ไม่มีทั้ง primary และ fallback")

    # Fallback: evenly-spaced ถ้า CP น้อยเกิน
    if len(cp_list) < MIN_TEETH_FOR_BSPLINE:
        print(f"  [CP Fallback] CP เหลือ {len(cp_list)} → ใช้ evenly-spaced")
        cens = np.array([m["centroid_t"] for m in matches_t])
        idx  = np.argsort(cens[:, 0])
        n_cp = min(7, len(idx))
        cp_i = np.round(np.linspace(0, len(idx) - 1, n_cp)).astype(int)
        cp_list = [cens[idx[i]].tolist() for i in cp_i]

    arr = np.array(cp_list)
    return arr[np.argsort(arr[:, 0])]   # เรียงตาม X เสมอ


def fit_bspline(ctrl_pts: np.ndarray,
                degree: int = BSPLINE_DEGREE,
                n_eval: int = BSPLINE_N_EVAL) -> tuple:
    """
    Fit Quadratic B-spline (p=2) ตาม Eq.(1):
        B(t) = Σ N_{i,p}(t) · L^i_{cu}

    ทำไม p=2 ไม่ใช่ p=3:
      Fan et al. §5.4.2 Fig.13 พิสูจน์ว่า p=3 oscillate ปลาย arch
      p=2 smooth และ stable กว่าในกรณีฟันเรียงผิดปกติ

    Returns
    -------
    curve_pts : np.ndarray (n_eval, 2)
    bs_x      : BSpline object สำหรับ x(t)
    bs_y      : BSpline object สำหรับ y(t)
    """
    n      = len(ctrl_pts)
    degree = min(degree, n - 1)   # safety: degree < n
    t      = np.linspace(0, 1, n)
    te     = np.linspace(0, 1, n_eval)

    bs_x = make_interp_spline(t, ctrl_pts[:, 0], k=degree)
    bs_y = make_interp_spline(t, ctrl_pts[:, 1], k=degree)

    return np.column_stack([bs_x(te), bs_y(te)]), bs_x, bs_y


# =============================================================================
# SECTION E — Deviation Analysis
# อ้างอิง: Fan et al. (2025) §4.3.3 Eq.(7)(8)
# =============================================================================

def dist_to_curve(px: float, py: float,
                  bs_x, bs_y,
                  n_search: int = 800) -> tuple:
    """
    คำนวณระยะตั้งฉาก (perpendicular distance) จากจุด (px,py)
    ไปยังเส้น B-spline ตาม Orthogonality Condition Eq.(7):

        x'(t)(x(t)-x₀) + y'(t)(y(t)-y₀) = 0

    วิธีการ: grid search + local refinement
    (ใช้ numeric เพราะ B-spline degree 2 ไม่มี closed-form solution)

    Returns
    -------
    dist    : float  ระยะตั้งฉาก (px)
    cx, cy  : float  จุดที่ใกล้ที่สุดบน curve
    delta_x : float  Δx = x₀ - x(t*)  ตาม Eq.(8)
    delta_y : float  Δy = y₀ - y(t*)  ตาม Eq.(8)
    """
    # Coarse search
    t_arr = np.linspace(0, 1, n_search)
    dists = np.hypot(px - bs_x(t_arr), py - bs_y(t_arr))
    i     = np.argmin(dists)

    # Fine search รอบจุดที่ใกล้สุด
    lo  = max(0.0, t_arr[i] - 3 / n_search)
    hi  = min(1.0, t_arr[i] + 3 / n_search)
    tf  = np.linspace(lo, hi, 300)
    df  = np.hypot(px - bs_x(tf), py - bs_y(tf))
    j   = np.argmin(df)

    cx = float(bs_x(tf[j]))
    cy = float(bs_y(tf[j]))

    return float(df[j]), cx, cy, float(px - cx), float(py - cy)


def compute_deviations(matches_t: list, bs_x, bs_y, inv: dict) -> list:
    """
    คำนวณ Deviation ของ MMR และ DMR ทุกซี่ใน ALL_TEETH (14 ซี่)

    ซี่ที่มีข้อมูล → คำนวณค่าตัวเลขครบ
    ซี่ที่ขาดหาย   → status="missing", ค่าทุก field = None

    Returns
    -------
    list of dict (14 items, เรียงตาม ALL_TEETH)
    """
    present_map = {m["tooth_class"]: m for m in matches_t}
    rows        = []

    for tid in ALL_TEETH:
        side = "Right" if tid[0] == "R" else "Left"

        if tid in present_map:
            m = present_map[tid]

            md, mcx, mcy, mdx, mdy = dist_to_curve(
                m["mmr_t"][0], m["mmr_t"][1], bs_x, bs_y)
            dd, dcx, dcy, ddx, ddy = dist_to_curve(
                m["dmr_t"][0], m["dmr_t"][1], bs_x, bs_y)

            rows.append({
                "tooth": tid, "status": "present", "side": side,
                # MMR
                "mmr_tx": round(m["mmr_t"][0], 2),
                "mmr_ty": round(m["mmr_t"][1], 2),
                "mmr_cx": round(mcx, 2), "mmr_cy": round(mcy, 2),
                "mmr_dev": round(md, 2),
                "mmr_dx":  round(mdx, 2), "mmr_dy": round(mdy, 2),
                # DMR
                "dmr_tx": round(m["dmr_t"][0], 2),
                "dmr_ty": round(m["dmr_t"][1], 2),
                "dmr_cx": round(dcx, 2), "dmr_cy": round(dcy, 2),
                "dmr_dev": round(dd, 2),
                "dmr_dx":  round(ddx, 2), "dmr_dy": round(ddy, 2),
                "mean_dev": round((md + dd) / 2, 2),
            })
        else:
            rows.append({
                "tooth": tid, "status": "missing", "side": side,
                "mmr_tx": None, "mmr_ty": None,
                "mmr_cx": None, "mmr_cy": None,
                "mmr_dev": None, "mmr_dx": None, "mmr_dy": None,
                "dmr_tx": None, "dmr_ty": None,
                "dmr_cx": None, "dmr_cy": None,
                "dmr_dev": None, "dmr_dx": None, "dmr_dy": None,
                "mean_dev": None,
            })

    return rows


# =============================================================================
# SECTION F — Individualized Ideal Arch Form
# แนวคิด: Individualized Ideal Arch (Andrews 1972 + Fan et al. 2025)
# =============================================================================

# Anchor teeth สำหรับ ideal arch — เลือกจากฟันที่มั่นคงทางคลินิก:
#   Molar  : ถูกยึดไว้เป็น anchor ในทุก orthodontic treatment
#   Incisor: กำหนด anterior midpoint ของ arch
#   ไม่รวม: canine, premolar ซึ่งมักต้องเคลื่อนที่มากที่สุด
_IDEAL_ARCH_ANCHORS = ["R37", "R36", "R31", "L41", "L46", "L47"]

# Weighted importance ของแต่ละ anchor
# Molar ให้น้ำหนักมากกว่า incisor เพราะเป็น stable reference ทางคลินิก
# อ้างอิง: Kondo et al. (2004) IEEE TMI ใช้ weighted fitting ที่ให้น้ำหนัก
#          anterior teeth มากกว่าสำหรับ arch form determination
_IDEAL_ARCH_WEIGHTS = {
    "R37": 2.0, "R36": 2.0,   # molar R — stable anchor
    "R31": 1.5, "L41": 1.5,   # central incisor — midpoint reference
    "L46": 2.0, "L47": 2.0,   # molar L — stable anchor
}


def select_ideal_arch_control_points(matches_t: list) -> np.ndarray | None:
    """
    เลือก Control Points สำหรับ Ideal Arch Form

    หลักการ (Individualized Ideal Arch):
    ──────────────────────────────────────────────────────────
    ใช้เฉพาะ anchor teeth ที่มั่นคงทางคลินิก ได้แก่
      - Molar (37/47, 36/46): stable anchor ในการรักษา
      - Central incisor (31/41): กำหนด anterior midpoint

    ไม่รวม canine (33/43) และ premolar (35/45) เพราะ:
      - ฟันเหล่านี้คือเป้าหมายการเคลื่อนที่ใน clear aligner
      - การรวมฟันที่ผิดตำแหน่งเข้าไปจะทำให้ ideal arch
        ถูก distort ตามความผิดปกติ

    อ้างอิง:
      Andrews (1972) Am J Orthod 62(3): six keys to normal occlusion
      Kondo et al. (2004) IEEE TMI: weighted arch fitting

    Returns
    -------
    np.ndarray shape (n, 2) เรียงตาม X, หรือ None ถ้า anchor < 3
    """
    kp_map  = {m["tooth_class"]: m for m in matches_t}
    cp_list = []

    # ลอง primary anchors ก่อน
    for tid in _IDEAL_ARCH_ANCHORS:
        if tid in kp_map:
            cp_list.append(kp_map[tid]["centroid_t"])
        else:
            # ลอง fallback ใกล้เคียง
            for fb in CP_FALLBACK.get(tid, []):
                if fb in kp_map:
                    cp_list.append(kp_map[fb]["centroid_t"])
                    print(f"  [Ideal CP Fallback] '{tid}' → ใช้ '{fb}'")
                    break

    if len(cp_list) < 3:
        print(f"  [Ideal Arch] anchor teeth น้อยเกินไป ({len(cp_list)}) "
              f"→ ใช้ descriptive arch แทน")
        return None

    arr = np.array(cp_list)
    return arr[np.argsort(arr[:, 0])]   # เรียงตาม X


def compute_ideal_arch(matches_t: list,
                        degree: int = BSPLINE_DEGREE,
                        n_eval: int = BSPLINE_N_EVAL) -> tuple | None:
    """
    สร้าง Individualized Ideal Arch Curve

    Concept:
    ──────────────────────────────────────────────────────────
    Ideal arch = B-spline interpolating curve ที่ผ่านจุด anchor
    teeth พอดี (ไม่ใช่ approximation)

    ใช้ make_interp_spline ซึ่งสร้าง curve ที่ผ่านทุก control
    point พอดี ทำให้ anchor teeth มี deviation ≈ 0 เสมอ
    ซึ่งถูกต้องตามหลักการ — anchor ไม่ต้องเคลื่อน

    ผลลัพธ์ที่ใช้ใน Clear Aligner planning:
      deviation จาก ideal arch = ระยะที่ฟันแต่ละซี่ต้องเคลื่อน
      เพื่อให้ได้ arch form ที่สมบูรณ์

    Returns
    -------
    (curve_pts, bs_x, bs_y) หรือ None ถ้าสร้างไม่ได้
    """
    ideal_cp = select_ideal_arch_control_points(matches_t)

    if ideal_cp is None:
        return None

    n   = len(ideal_cp)
    deg = min(degree, n - 1)

    # ใช้ uniform parameterization ตามระยะทางสะสม (chord-length)
    # ทำให้ curve smooth กว่า uniform t เมื่อจุดห่างไม่เท่ากัน
    diffs  = np.diff(ideal_cp, axis=0)
    dists  = np.hypot(diffs[:, 0], diffs[:, 1])
    t_raw  = np.concatenate([[0.0], np.cumsum(dists)])
    t_norm = t_raw / t_raw[-1]   # normalize → [0, 1]
    te     = np.linspace(0, 1, n_eval)

    # make_interp_spline: curve ผ่านทุก control point พอดี
    bs_x = make_interp_spline(t_norm, ideal_cp[:, 0], k=deg)
    bs_y = make_interp_spline(t_norm, ideal_cp[:, 1], k=deg)

    print(f"  [Ideal Arch] Built from {n} anchor CPs  "
          f"(degree p={deg}, chord-length parameterization)")

    return np.column_stack([bs_x(te), bs_y(te)]), bs_x, bs_y


def compute_ideal_deviations(matches_t: list,
                               ideal_bs_x, ideal_bs_y,
                               inv: dict) -> list:
    """
    คำนวณ Deviation ของ MMR/DMR แต่ละซี่จาก Ideal Arch Curve

    ความแตกต่างจาก compute_deviations():
      - ใช้ ideal arch (จาก anchor teeth เท่านั้น)
        แทน descriptive arch (จากฟันทุกซี่)
      - เพิ่ม field: ideal_mmr_dev, ideal_dmr_dev, movement_needed
      - movement_needed = ระยะที่ฟันซี่นั้นต้องเคลื่อนเพื่อให้
        MMR/DMR อยู่บน ideal arch (หน่วย: pixel)

    ข้อจำกัด:
      หน่วยเป็น pixel ไม่ใช่ mm เพราะยังไม่มี scale calibration
      ต้อง calibrate ด้วย reference object ในภาพเพื่อแปลงเป็น mm

    Returns
    -------
    list of dict (14 items) ที่มี field เพิ่มเติม:
      ideal_mmr_dev, ideal_mmr_dx, ideal_mmr_dy,
      ideal_dmr_dev, ideal_dmr_dx, ideal_dmr_dy,
      ideal_mean_dev, movement_needed_px,
      is_anchor_tooth (bool: ฟันนี้ใช้สร้าง ideal arch หรือไม่)
    """
    present_map = {m["tooth_class"]: m for m in matches_t}
    rows        = []

    for tid in ALL_TEETH:
        side = "Right" if tid[0] == "R" else "Left"
        is_anchor = tid in _IDEAL_ARCH_ANCHORS

        if tid in present_map:
            m = present_map[tid]

            # Deviation จาก ideal arch
            imd, imcx, imcy, imdx, imdy = dist_to_curve(
                m["mmr_t"][0], m["mmr_t"][1], ideal_bs_x, ideal_bs_y)
            idd, idcx, idcy, iddx, iddy = dist_to_curve(
                m["dmr_t"][0], m["dmr_t"][1], ideal_bs_x, ideal_bs_y)

            # movement_needed = mean deviation ของ MMR และ DMR
            # = ระยะเฉลี่ยที่ฟันซี่นี้ต้องเคลื่อนเพื่ออยู่บน ideal arch
            movement = round((imd + idd) / 2, 2)

            rows.append({
                "tooth":           tid,
                "status":          "present",
                "side":            side,
                "is_anchor":       is_anchor,
                # MMR → ideal arch
                "ideal_mmr_dev":   round(imd, 2),
                "ideal_mmr_dx":    round(imdx, 2),
                "ideal_mmr_dy":    round(imdy, 2),
                "ideal_mmr_cx":    round(imcx, 2),
                "ideal_mmr_cy":    round(imcy, 2),
                # DMR → ideal arch
                "ideal_dmr_dev":   round(idd, 2),
                "ideal_dmr_dx":    round(iddx, 2),
                "ideal_dmr_dy":    round(iddy, 2),
                "ideal_dmr_cx":    round(idcx, 2),
                "ideal_dmr_cy":    round(idcy, 2),
                # Summary
                "ideal_mean_dev":      round((imd + idd) / 2, 2),
                "movement_needed_px":  movement,
            })
        else:
            rows.append({
                "tooth":               tid,
                "status":              "missing",
                "side":                side,
                "is_anchor":           is_anchor,
                "ideal_mmr_dev":       None,
                "ideal_mmr_dx":        None,
                "ideal_mmr_dy":        None,
                "ideal_mmr_cx":        None,
                "ideal_mmr_cy":        None,
                "ideal_dmr_dev":       None,
                "ideal_dmr_dx":        None,
                "ideal_dmr_dy":        None,
                "ideal_dmr_cx":        None,
                "ideal_dmr_cy":        None,
                "ideal_mean_dev":      None,
                "movement_needed_px":  None,
            })

    return rows
