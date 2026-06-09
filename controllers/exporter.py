"""
controllers/exporter.py — Controller Layer (Output)
หน้าที่: Export ผลลัพธ์เป็น CSV และ print ตาราง console
แยกออกจาก Model เพื่อให้ Model ไม่รู้จัก I/O เลย
"""
from __future__ import annotations
import csv
from datetime import datetime

import numpy as np

from config.settings import DEVIATION_CSV_FIELDS


# =============================================================================
# Console Output
# =============================================================================
def print_inventory_report(inv: dict, img_name: str = "") -> None:
    """แสดง Tooth Inventory Report บน console"""
    print("\n" + "=" * 62)
    print(f"  Tooth Inventory  —  {img_name}")
    print("=" * 62)
    print(f"  ฟันที่ตรวจพบ   : {inv['n_present']:>2} ซี่  {inv['present']}")
    print(f"  ฟันที่ขาดหาย   : {inv['n_missing']:>2} ซี่  {inv['missing']}")
    if inv["invalid"]:
        print(f"  ID ไม่รู้จัก   : {inv['invalid']}")
    print(f"  Anchor (31,41) : "
          f"{'✅ ครบ' if inv['cp_primary_ok'] else '❌ ขาด'}")
    if inv["warnings"]:
        print()
        for w in inv["warnings"]:
            print(f"  ⚠  {w}")
    print("=" * 62)


def print_deviation_table(dev_rows: list) -> None:
    """แสดงตาราง Deviation Analysis บน console"""
    h = (f"{'Tooth':<6} | {'Status':<8} | "
         f"{'MMR dist':>9} {'MMR Δx':>8} {'MMR Δy':>8} | "
         f"{'DMR dist':>9} {'DMR Δx':>8} {'DMR Δy':>8}")
    sep = "-" * len(h)
    print("\n" + "=" * len(h))
    print("  Deviation Analysis  (canonical frame, px)")
    print("=" * len(h))
    print(h)
    print(sep)

    for r in dev_rows:
        if r["status"] == "present":
            print(
                f"{r['tooth']:<6} | {'present':<8} | "
                f"{r['mmr_dev']:>9.1f} {r['mmr_dx']:>8.1f} "
                f"{r['mmr_dy']:>8.1f} | "
                f"{r['dmr_dev']:>9.1f} {r['dmr_dx']:>8.1f} "
                f"{r['dmr_dy']:>8.1f}"
            )
        else:
            print(
                f"{r['tooth']:<6} | {'MISSING':<8} | "
                f"{'—':>9} {'—':>8} {'—':>8} | "
                f"{'—':>9} {'—':>8} {'—':>8}"
            )

    present = [r for r in dev_rows if r["status"] == "present"]
    if present:
        mmr_all = [r["mmr_dev"] for r in present]
        dmr_all = [r["dmr_dev"] for r in present]
        print(sep)
        print(
            f"{'MEAN':<6} | {f'n={len(present)}':<8} | "
            f"{np.mean(mmr_all):>9.1f} {'':>8} {'':>8} | "
            f"{np.mean(dmr_all):>9.1f}"
        )
        print(
            f"{'STD':<6} | {'':<8} | "
            f"{np.std(mmr_all):>9.1f} {'':>8} {'':>8} | "
            f"{np.std(dmr_all):>9.1f}"
        )
    print("=" * len(h))


# =============================================================================
# CSV Export: deviation results
# =============================================================================
def export_deviation_csv(dev_rows: list, inv: dict, tf: dict,
                          image_name: str, save_path: str) -> None:
    """
    Export Deviation Analysis เป็น CSV

    โครงสร้างไฟล์:
      - Metadata header (# comment lines)
      - Inventory summary (# comment lines)
      - Header row (fieldnames)
      - Data rows: present = ตัวเลข, missing = "MISSING"
      - Blank separator
      - Summary stats: MEAN, STD, MAX, MIN
    """
    present_rows = [r for r in dev_rows if r["status"] == "present"]
    missing_rows = [r for r in dev_rows if r["status"] == "missing"]
    mmr_vals = [r["mmr_dev"] for r in present_rows]
    dmr_vals = [r["dmr_dev"] for r in present_rows]
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(save_path, "w", newline="", encoding="utf-8") as f:

        # ── Metadata comments ─────────────────────────────────────────────
        meta_lines = [
            "# ============================================================",
            "# Deviation Analysis Report",
            "# ============================================================",
            f"# Image           : {image_name}",
            f"# Generated at    : {now}",
            f"# Coordinate frame: Origin = midpoint(MMR_31, MMR_41)",
            f"# X-axis          : PCA (all centroids) + MMR_31/41 refinement",
            f"# Origin (raw px) : ({tf['tx']:.2f}, {tf['ty']:.2f})",
            f"# Rotation angle  : {tf['angle_deg']:.4f} degrees",
            f"# Rotation method : {tf.get('method', 'PCA+MMR31/41')}",
            f"# B-spline        : degree p=2 (Fan et al. 2025 §4.2.1)",
            f"# Deviation eq.   : Orthogonality Condition (Fan et al. §4.3.3)",
            f"# Unit            : px (pixel, canonical coordinate)",
            "#",
            f"# --- Tooth Inventory ---",
            f"# Present  ({inv['n_present']:>2}) : {inv['present']}",
            f"# Missing  ({inv['n_missing']:>2}) : {inv['missing']}",
        ]
        if inv["invalid"]:
            meta_lines.append(f"# Invalid        : {inv['invalid']}")
        meta_lines.append(
            f"# Anchor 31+41   : {'OK' if inv['cp_primary_ok'] else 'MISSING'}"
        )
        for w in inv["warnings"]:
            meta_lines.append(f"# WARNING        : {w}")
        meta_lines.append("#")

        for line in meta_lines:
            f.write(line + "\n")

        writer = csv.DictWriter(f, fieldnames=DEVIATION_CSV_FIELDS)
        writer.writeheader()

        # ── Data rows ─────────────────────────────────────────────────────
        for r in dev_rows:
            if r["status"] == "present":
                writer.writerow({
                    "tooth_id":        r["tooth"],
                    "iso_number":      r["tooth"][1:],
                    "side":            r["side"],
                    "status":          "present",
                    "mmr_tx_px":       r["mmr_tx"],
                    "mmr_ty_px":       r["mmr_ty"],
                    "mmr_closest_x":   r["mmr_cx"],
                    "mmr_closest_y":   r["mmr_cy"],
                    "mmr_dist_px":     r["mmr_dev"],
                    "mmr_delta_x_px":  r["mmr_dx"],
                    "mmr_delta_y_px":  r["mmr_dy"],
                    "dmr_tx_px":       r["dmr_tx"],
                    "dmr_ty_px":       r["dmr_ty"],
                    "dmr_closest_x":   r["dmr_cx"],
                    "dmr_closest_y":   r["dmr_cy"],
                    "dmr_dist_px":     r["dmr_dev"],
                    "dmr_delta_x_px":  r["dmr_dx"],
                    "dmr_delta_y_px":  r["dmr_dy"],
                    "mean_mr_dist_px": r["mean_dev"],
                })
            else:
                empty = {k: "MISSING" for k in DEVIATION_CSV_FIELDS}
                empty.update({
                    "tooth_id":   r["tooth"],
                    "iso_number": r["tooth"][1:],
                    "side":       r["side"],
                    "status":     "MISSING",
                })
                writer.writerow(empty)

        # ── Summary stats ─────────────────────────────────────────────────
        writer.writerow({k: "" for k in DEVIATION_CSV_FIELDS})

        def _stat_row(label: str, fn) -> None:
            mmr_s  = round(float(fn(mmr_vals)), 2) if mmr_vals else ""
            dmr_s  = round(float(fn(dmr_vals)), 2) if dmr_vals else ""
            mean_s = (round((float(fn(mmr_vals)) + float(fn(dmr_vals))) / 2, 2)
                      if mmr_vals and dmr_vals else "")
            writer.writerow({
                "tooth_id":        "SUMMARY",
                "iso_number":      label,
                "side":            "",
                "status":          (f"n_present={len(present_rows)} "
                                    f"n_missing={len(missing_rows)}"),
                "mmr_tx_px": "", "mmr_ty_px": "",
                "mmr_closest_x": "", "mmr_closest_y": "",
                "mmr_dist_px":    mmr_s,
                "mmr_delta_x_px": "", "mmr_delta_y_px": "",
                "dmr_tx_px": "", "dmr_ty_px": "",
                "dmr_closest_x": "", "dmr_closest_y": "",
                "dmr_dist_px":    dmr_s,
                "dmr_delta_x_px": "", "dmr_delta_y_px": "",
                "mean_mr_dist_px": mean_s,
            })

        for label, fn in [
            ("MEAN", np.mean), ("STD", np.std),
            ("MAX", np.max),   ("MIN", np.min),
        ]:
            _stat_row(label, fn)

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n[Export] Deviation CSV → {save_path}")
    print(f"         Present: {len(present_rows)} ซี่  "
          f"|  Missing: {len(missing_rows)} ซี่")
    if mmr_vals:
        print(f"         MMR  mean={np.mean(mmr_vals):.1f} "
              f"±{np.std(mmr_vals):.1f} px  "
              f"max={np.max(mmr_vals):.1f} px")
    if dmr_vals:
        print(f"         DMR  mean={np.mean(dmr_vals):.1f} "
              f"±{np.std(dmr_vals):.1f} px  "
              f"max={np.max(dmr_vals):.1f} px")
