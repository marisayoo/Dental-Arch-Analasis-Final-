"""
controllers/csv_export.py  —  Export ผลลัพธ์ทั้งหมดเป็น CSV
สร้าง 2 ไฟล์:
  1. canonical_keypoints_[name].csv  — พิกัดจุดสำคัญหลังเข้า canonical frame
  2. deviation_full_[name].csv       — deviation ครบทุก field รวม ideal arch
"""
from __future__ import annotations
from config.settings import ALL_TEETH, IDEAL_ARCH_ANCHORS
import csv
import numpy as np
from datetime import datetime
from pathlib import Path


# =============================================================================
# FILE 1 — Canonical Keypoints
# =============================================================================

def export_canonical_keypoints(
    matches_t: list,
    tf: dict,
    inv: dict,
    image_name: str,
    exports_dir: str,
) -> str:
    """
    Export พิกัดจุดสำคัญทุกซี่หลังแปลงเข้า Canonical Frame

    Columns:
      tooth_id, fdi_number, side, status
      centroid_x, centroid_y
      mmr_x, mmr_y
      dmr_x, dmr_y
      anc_x, anc_y

    หมายเหตุ:
      - พิกัดเป็น canonical frame (pixel) ไม่ใช่ image frame
      - Y เป็นค่าลบเพราะ posterior อยู่ลึกกว่า anterior ในแกน canonical
      - R31/L41 ควรมี mmr_y ≈ 0.0 เสมอ (คือ Origin ของ frame)
    """
    from config.settings import ALL_TEETH

    fname = Path(exports_dir) / f"canonical_keypoints_{Path(image_name).stem}.csv"
    fname.parent.mkdir(parents=True, exist_ok=True)

    pm  = {m["tooth_class"]: m for m in matches_t}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fields = [
        "tooth_id", "fdi_number", "side", "status",
        "centroid_x_px", "centroid_y_px",
        "mmr_x_px",      "mmr_y_px",
        "dmr_x_px",      "dmr_y_px",
        "anc_x_px",      "anc_y_px",
    ]

    with open(fname, "w", newline="", encoding="utf-8") as f:
        # metadata header
        meta = [
            "# ============================================================",
            "# Canonical Frame Keypoints",
            "# ============================================================",
            f"# Image       : {image_name}",
            f"# Generated   : {now}",
            f"# Frame       : Canonical coordinate system",
            f"# Origin      : midpoint(MMR_31, MMR_41)  [Fan et al. §4.1]",
            f"# Origin (raw): ({tf['tx']:.2f}, {tf['ty']:.2f}) px",
            f"# Rotation    : {tf['angle_deg']:.4f} degrees",
            f"# Method      : {tf.get('method', 'Triangular')}",
            f"#   → Triangular Method [Li et al. 2017, 93.3% accuracy]",
            f"#   → COG fallback      [Wellens 2007] (if molar missing)",
            f"# Unit        : pixel (canonical frame)",
            f"# Y-axis      : negative = posterior  (display: flip Y)",
            f"# R-side      : x > 0  (patient right, FDI 3x)",
            f"# L-side      : x < 0  (patient left,  FDI 4x)",
            f"# Teeth total : {inv['n_present']} present, {inv['n_missing']} missing",
            "# ============================================================",
            "#",
            "# Keypoints per tooth:",
            "#   MMR = Mesial Marginal Ridge  (KP index 2)",
            "#   DMR = Distal Marginal Ridge  (KP index 0)",
            "#   CEN = Centroid of crown      (KP index 1)",
            "#   ANC = Anchor point           (KP index 3)",
            "#",
            "# Verification: MMR_31 Y ≈ 0.0  and  MMR_41 Y ≈ 0.0",
            f"#   MMR_31 Y = {pm.get('R31', {}).get('mmr_t', [0,0])[1]:.4f} px",
            f"#   MMR_41 Y = {pm.get('L41', {}).get('mmr_t', [0,0])[1]:.4f} px",
            "#",
        ]
        for line in meta:
            f.write(line + "\n")

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for tid in ALL_TEETH:
            side = "Right" if tid[0] == "R" else "Left"
            fdi  = tid[1:]

            if tid in pm:
                m = pm[tid]
                writer.writerow({
                    "tooth_id":      tid,
                    "fdi_number":    fdi,
                    "side":          side,
                    "status":        "present",
                    "centroid_x_px": round(m["centroid_t"][0], 4),
                    "centroid_y_px": round(m["centroid_t"][1], 4),
                    "mmr_x_px":      round(m["mmr_t"][0], 4),
                    "mmr_y_px":      round(m["mmr_t"][1], 4),
                    "dmr_x_px":      round(m["dmr_t"][0], 4),
                    "dmr_y_px":      round(m["dmr_t"][1], 4),
                    "anc_x_px":      round(m["anc_t"][0], 4),
                    "anc_y_px":      round(m["anc_t"][1], 4),
                })
            else:
                writer.writerow({
                    "tooth_id":      tid,
                    "fdi_number":    fdi,
                    "side":          side,
                    "status":        "MISSING",
                    "centroid_x_px": "MISSING", "centroid_y_px": "MISSING",
                    "mmr_x_px":      "MISSING", "mmr_y_px":      "MISSING",
                    "dmr_x_px":      "MISSING", "dmr_y_px":      "MISSING",
                    "anc_x_px":      "MISSING", "anc_y_px":      "MISSING",
                })

        # summary footer
        f.write("\n")
        f.write(f"#SUMMARY,n_present,{inv['n_present']}\n")
        f.write(f"#SUMMARY,n_missing,{inv['n_missing']}\n")
        f.write(f"#SUMMARY,anchor_ok,{inv.get('cp_primary_ok', inv.get('anchor_ok','N/A'))}\n")
        f.write(f"#SUMMARY,angle_deg,{tf['angle_deg']:.4f}\n")
        f.write(f"#SUMMARY,method,{tf.get('method','')}\n")

    print(f"[CSV] Canonical keypoints → {fname.name}")
    return str(fname)


# =============================================================================
# FILE 2 — Full Deviation Report
# =============================================================================

def export_deviation_full(
    matches_t: list,
    dev_rows: list,
    ideal_dev: list,
    tf: dict,
    inv: dict,
    image_name: str,
    exports_dir: str
) -> str:
    """
    Export Deviation Analysis ครบทุก field รวม ideal arch

    Columns per tooth:
      — Position (canonical):  centroid, mmr, dmr
      — Descriptive arch:      mmr_dev, mmr_dx, mmr_dy, dmr_dev, dmr_dx, dmr_dy, mean_dev
      — Ideal arch:            ideal_mmr_dev, ideal_dmr_dev, movement_needed_px, is_anchor
      — Clinical note:         deviation_grade, movement_grade

    deviation_grade:
      none   = 0 px
      low    = 1-5 px
      medium = 5-15 px
      high   = >15 px

    movement_grade:
      anchor  = anchor tooth (stable)
      low     = 0-10 px
      medium  = 10-22 px
      high    = >22 px
    """

    fname = Path(exports_dir) / f"deviation_full_{Path(image_name).stem}.csv"
    fname.parent.mkdir(parents=True, exist_ok=True)

    pm       = {m["tooth_class"]: m for m in matches_t}
    dev_map  = {r["tooth"]: r for r in dev_rows}
    idev_map = {r["tooth"]: r for r in ideal_dev}
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # stats for summary
    pres  = [r for r in dev_rows  if r["status"] == "present"]
    ipres = [r for r in ideal_dev if r["status"] == "present"]
    mmr_d = [r["mmr_dev"] for r in pres]
    dmr_d = [r["dmr_dev"] for r in pres]
    mov_d = [r["movement_needed_px"] for r in ipres if not r.get("is_anchor")]

    fields = [
        # identification
        "tooth_id", "fdi_number", "side", "status", "is_anchor_tooth",
        # canonical coordinates
        "centroid_x_px", "centroid_y_px",
        "mmr_x_px",      "mmr_y_px",
        "dmr_x_px",      "dmr_y_px",
        # deviation → descriptive arch  [Fan et al. §4.3.3]
        "mmr_foot_x_px",   "mmr_foot_y_px",
        "mmr_dist_px",     "mmr_delta_x_px", "mmr_delta_y_px",
        "dmr_foot_x_px",   "dmr_foot_y_px",
        "dmr_dist_px",     "dmr_delta_x_px", "dmr_delta_y_px",
        "mean_dist_px",
        "deviation_grade",
        # deviation → ideal arch  [Andrews 1972]
        "ideal_mmr_dist_px", "ideal_dmr_dist_px",
        "movement_needed_px",
        "movement_grade",
    ]

    def dev_grade(v):
        if v is None:    return "MISSING"
        if v == 0:       return "none"
        if v <= 5:       return "low"
        if v <= 15:      return "medium"
        return "high"

    def mov_grade(v, anchor):
        if anchor:       return "anchor"
        if v is None:    return "MISSING"
        if v <= 10:      return "low"
        if v <= 22:      return "medium"
        return "high"

    with open(fname, "w", newline="", encoding="utf-8") as f:
        meta = [
            "# ============================================================",
            "# Full Deviation Analysis Report",
            "# ============================================================",
            f"# Image       : {image_name}",
            f"# Generated   : {now}",
            f"# Method      : {tf.get('method','Triangular')}",
            f"# Origin (raw): ({tf['tx']:.2f}, {tf['ty']:.2f}) px",
            f"# Angle       : {tf['angle_deg']:.4f}°",
            "#",
            "# === REFERENCES ===",
            "# Canonical Frame : Fan et al. (2025) §4.1",
            "# B-spline arch   : Fan et al. (2025) §4.2.1 Eq.(1)  degree p=2",
            "# Deviation eq.   : Fan et al. (2025) §4.3.3 Eq.(7-8) Orthogonality",
            "# Orientation     : Li, Gateno, Xia (2017) — Triangular Method 93.3%",
            "# Ideal arch      : Andrews (1972) — six keys to normal occlusion",
            "#",
            "# === DEVIATION GRADES ===",
            "# deviation_grade: none(0) | low(1-5) | medium(5-15) | high(>15) px",
            "# movement_grade : anchor | low(0-10) | medium(10-22) | high(>22) px",
            "#",
            "# === CLINICAL NOTE ===",
            "# All values in PIXEL (canonical frame).",
            "# To convert to mm: px × (known_distance_mm / known_distance_px)",
            "# Calibration requires a reference object in the photograph.",
            "#",
            f"# n_present = {inv['n_present']}  n_missing = {inv['n_missing']}",
        ]
        if mmr_d:
            meta += [
                f"# Descriptive arch: MMR μ={np.mean(mmr_d):.2f} σ={np.std(mmr_d):.2f} max={np.max(mmr_d):.2f} px",
                f"# Descriptive arch: DMR μ={np.mean(dmr_d):.2f} σ={np.std(dmr_d):.2f} max={np.max(dmr_d):.2f} px",
            ]
        if mov_d:
            meta += [
                f"# Ideal arch move:  μ={np.mean(mov_d):.2f} σ={np.std(mov_d):.2f} max={np.max(mov_d):.2f} px  (non-anchor only)",
            ]
        meta.append("#")
        for line in meta:
            f.write(line + "\n")

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for tid in ALL_TEETH:
            side     = "Right" if tid[0] == "R" else "Left"
            fdi      = tid[1:]
            is_anch  = tid in IDEAL_ARCH_ANCHORS
            dr       = dev_map.get(tid, {})
            ir       = idev_map.get(tid, {})

            if dr.get("status") == "present":
                m = pm[tid]
                move_val = ir.get("movement_needed_px")
                mean_val = dr.get("mean_dev")

                writer.writerow({
                    "tooth_id":        tid,
                    "fdi_number":      fdi,
                    "side":            side,
                    "status":          "present",
                    "is_anchor_tooth": "YES" if is_anch else "no",
                    # canonical position
                    "centroid_x_px":   round(m["centroid_t"][0], 3),
                    "centroid_y_px":   round(m["centroid_t"][1], 3),
                    "mmr_x_px":        round(m["mmr_t"][0], 3),
                    "mmr_y_px":        round(m["mmr_t"][1], 3),
                    "dmr_x_px":        round(m["dmr_t"][0], 3),
                    "dmr_y_px":        round(m["dmr_t"][1], 3),
                    # descriptive deviation
                    "mmr_foot_x_px":   round(dr.get("mmr_cx", 0), 3),
                    "mmr_foot_y_px":   round(dr.get("mmr_cy", 0), 3),
                    "mmr_dist_px":     round(dr.get("mmr_dev", 0), 3),
                    "mmr_delta_x_px":  round(dr.get("mmr_dx", 0), 3),
                    "mmr_delta_y_px":  round(dr.get("mmr_dy", 0), 3),
                    "dmr_foot_x_px":   round(dr.get("dmr_cx", 0), 3),
                    "dmr_foot_y_px":   round(dr.get("dmr_cy", 0), 3),
                    "dmr_dist_px":     round(dr.get("dmr_dev", 0), 3),
                    "dmr_delta_x_px":  round(dr.get("dmr_dx", 0), 3),
                    "dmr_delta_y_px":  round(dr.get("dmr_dy", 0), 3),
                    "mean_dist_px":    round(mean_val, 3) if mean_val else 0,
                    "deviation_grade": dev_grade(mean_val),
                    # ideal arch
                    "ideal_mmr_dist_px":  ir.get("ideal_mmr_dev", "N/A"),
                    "ideal_dmr_dist_px":  ir.get("ideal_dmr_dev", "N/A"),
                    "movement_needed_px": move_val if move_val is not None else "N/A",
                    "movement_grade":     mov_grade(move_val, is_anch),
                })
            else:
                empty = {k: "MISSING" for k in fields}
                empty.update({
                    "tooth_id":        tid,
                    "fdi_number":      fdi,
                    "side":            side,
                    "status":          "MISSING",
                    "is_anchor_tooth": "YES" if is_anch else "no",
                    "deviation_grade": "MISSING",
                    "movement_grade":  "MISSING",
                })
                writer.writerow(empty)

        # Summary stats rows
        f.write("\n")
        f.write("# === SUMMARY STATISTICS ===\n")
        f.write("# stat,metric,value,unit\n")
        if mmr_d:
            for stat, fn in [("mean",np.mean),("std",np.std),
                              ("max",np.max),  ("min",np.min)]:
                f.write(f"#STAT,{stat},mmr_dist,{fn(mmr_d):.3f},px\n")
                f.write(f"#STAT,{stat},dmr_dist,{fn(dmr_d):.3f},px\n")
        if mov_d:
            for stat, fn in [("mean",np.mean),("std",np.std),
                              ("max",np.max),  ("min",np.min)]:
                f.write(f"#STAT,{stat},movement_needed_non_anchor,{fn(mov_d):.3f},px\n")
        f.write(f"#STAT,count,present,{inv['n_present']},teeth\n")
        f.write(f"#STAT,count,missing,{inv['n_missing']},teeth\n")
        f.write(f"#STAT,transform_angle,{tf['angle_deg']:.4f},degrees\n")
        f.write(f"#STAT,orientation_method,{tf.get('method','')},\n")

    print(f"[CSV] Full deviation report → {fname.name}")
    return str(fname)
