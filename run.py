"""
run.py  —  จุดเริ่มต้นของระบบ  (entry point เดียว)

วิธีใช้งาน:
    python run.py                           → เลือกภาพแบบ interactive
    python run.py --image data/input/x.jpg  → ระบุภาพโดยตรง
    python run.py --csv   data/input/t.csv  → CSV mode (ไม่ใช้ YOLO)
    python run.py --batch                   → วิเคราะห์ทุกภาพใน data/input/

ผลลัพธ์ถูกบันทึกใน:
    output/cases/[ชื่อภาพ]/steps/    — ภาพ step-by-step 7 ขั้น
    output/cases/[ชื่อภาพ]/exports/  — CSV deviation + ideal arch
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import (
    OUTPUT_DIR, CASES_DIR, REPORTS_DIR, MIN_TEETH_FOR_ANALYSIS,
)
from models.tooth_data import check_tooth_inventory
from models.geometry import (
    compute_transform, transform_all,
    select_control_points, fit_bspline, compute_deviations,
    compute_ideal_arch, compute_ideal_deviations,
)
from controllers.data_loader import load_from_csv, load_from_model
from controllers.exporter import (
    print_inventory_report, print_deviation_table,
    export_deviation_csv,
)
from views.step_figures import generate_all_steps
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
INPUT_DIR  = ROOT / "data" / "input"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# =============================================================================
# Pipeline
# =============================================================================

def _export_ideal_csv(ideal_dev, tf, img_name, save_path):
    """บันทึก ideal deviation ลง CSV"""
    import csv
    from datetime import datetime
    fields = [
        "tooth_id", "side", "status", "is_anchor_tooth",
        "ideal_mmr_dev_px", "ideal_mmr_dx_px", "ideal_mmr_dy_px",
        "ideal_dmr_dev_px", "ideal_dmr_dx_px", "ideal_dmr_dy_px",
        "ideal_mean_dev_px", "movement_needed_px",
    ]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        for line in [
            "# Ideal Arch Deviation Report",
            f"# Image    : {img_name}",
            f"# Generated: {now}",
            "# Ideal Arch: Individualized (molar + incisor anchors only)",
            "# Anchor teeth: R37/R36, R31, L41, L46/L47",
            f"# Method   : {tf.get('method', 'Triangular')}",
            "# Unit     : pixel (canonical frame) — no mm calibration yet",
            "# Note     : movement_needed_px = estimated displacement",
            "#            to reach ideal arch position",
            "# Ref      : Andrews (1972); Fan et al. (2025)",
            "#",
        ]:
            f.write(line + "\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ideal_dev:
            if r["status"] == "present":
                writer.writerow({
                    "tooth_id": r["tooth"], "side": r["side"],
                    "status": "present",
                    "is_anchor_tooth": "YES" if r["is_anchor"] else "no",
                    "ideal_mmr_dev_px":  r["ideal_mmr_dev"],
                    "ideal_mmr_dx_px":   r["ideal_mmr_dx"],
                    "ideal_mmr_dy_px":   r["ideal_mmr_dy"],
                    "ideal_dmr_dev_px":  r["ideal_dmr_dev"],
                    "ideal_dmr_dx_px":   r["ideal_dmr_dx"],
                    "ideal_dmr_dy_px":   r["ideal_dmr_dy"],
                    "ideal_mean_dev_px": r["ideal_mean_dev"],
                    "movement_needed_px": r["movement_needed_px"],
                })
            else:
                writer.writerow({
                    "tooth_id": r["tooth"], "side": r["side"],
                    "status": "MISSING",
                    "is_anchor_tooth": "YES" if r["is_anchor"] else "no",
                    **{k: "MISSING" for k in fields[4:]},
                })


def run_pipeline(image_path=None, csv_path=None, save_steps=True):
    """
    รัน analysis pipeline สำหรับ 1 case
    คืน dict ผลลัพธ์ หรือ None ถ้าล้มเหลว
    """
    print("=" * 66)
    print("  AI-Driven Occlusal Surface Extraction  v3")
    print("  Dental Arch Template + Clear Aligner Deviation Analysis")
    print("=" * 66)

    # ensure output dirs
    for d in [OUTPUT_DIR, CASES_DIR, REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if csv_path:
        print(f"\n[Mode] CSV  -> {csv_path}")
        matches    = load_from_csv(str(csv_path))
        image_name = Path(csv_path).stem
        case_dir   = CASES_DIR / image_name
    elif image_path:
        print(f"\n[Mode] YOLO -> {image_path}")
        _, _, matches = load_from_model(str(image_path))
        image_name    = Path(image_path).stem
        case_dir      = CASES_DIR / image_name
    else:
        print("\n[Error] ต้องระบุ image_path หรือ csv_path")
        return None

    steps_dir   = case_dir / "steps"
    exports_dir = case_dir / "exports"
    steps_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Inventory ──────────────────────────────────────────────────────────
    inv = check_tooth_inventory(matches)
    print_inventory_report(inv, image_name)
    if inv["n_present"] < MIN_TEETH_FOR_ANALYSIS:
        print(f"\n[STOP] ฟันน้อยเกินไป ({inv['n_present']} ซี่)")
        return None

    # ── 3. Canonical Transform ────────────────────────────────────────────────
    print("\n[Step 3] Computing canonical frame (Triangular Method)...")
    tf = compute_transform(matches)
    print(f"  Origin : ({tf['tx']:.1f}, {tf['ty']:.1f}) px")
    print(f"  Angle  : {tf['angle_deg']:.3f}°   Method: {tf['method']}")
    mt = transform_all(matches, tf)

    # ── 4. B-spline ───────────────────────────────────────────────────────────
    print("\n[Step 4] Fitting B-spline arch curve (p=2)...")
    ctrl_pts    = select_control_points(mt)
    crv, bx, by = fit_bspline(ctrl_pts, degree=2)
    print(f"  Control pts: {len(ctrl_pts)}   Arch width: {crv[:,0].max()-crv[:,0].min():.0f} px")

    # ── 5. Deviation ──────────────────────────────────────────────────────────
    print("\n[Step 5] Computing deviation from descriptive arch...")
    dev_rows = compute_deviations(mt, bx, by, inv)
    print_deviation_table(dev_rows)
    csv_desc = exports_dir / f"deviation_{image_name}.csv"
    export_deviation_csv(dev_rows, inv, tf, image_name, str(csv_desc))

    # ── 6. Ideal Arch ─────────────────────────────────────────────────────────
    print("\n[Step 6] Computing individualized ideal arch...")
    ideal_result = compute_ideal_arch(mt, degree=2)
    ideal_dev    = []
    ideal_crv    = None

    if ideal_result:
        ideal_crv, ibx, iby = ideal_result
        ideal_dev  = compute_ideal_deviations(mt, ibx, iby, inv)
        pres_ideal = [r for r in ideal_dev if r["status"] == "present"]
        non_anch   = [r for r in pres_ideal if not r.get("is_anchor")]
        mov_vals   = [r["movement_needed_px"] for r in non_anch]

        print(f"\n  {'Tooth':<6} {'Anchor':8} {'MMR→ideal':>10} {'DMR→ideal':>10} {'Move(px)':>10}")
        print("  " + "─" * 52)
        for r in ideal_dev:
            if r["status"] == "present":
                a = "★ anchor" if r["is_anchor"] else ""
                print(f"  {r['tooth']:<6} {a:<8} "
                      f"{r['ideal_mmr_dev']:>10.1f} "
                      f"{r['ideal_dmr_dev']:>10.1f} "
                      f"{r['movement_needed_px']:>10.1f}")
            else:
                print(f"  {r['tooth']:<6} {'':8} {'MISSING':>10}")
        print("  " + "─" * 52)
        if mov_vals:
            print(f"  {'MEAN (non-anchor)':<30} {np.mean(mov_vals):>10.1f} px")

        csv_ideal = exports_dir / f"ideal_deviation_{image_name}.csv"
        _export_ideal_csv(ideal_dev, tf, image_name, str(csv_ideal))
        print("\n  ★ = anchor teeth (stable, no movement needed)")
        print("  NOTE: unit is pixel — mm calibration required for clinical use")
    else:
        print("  [Skip] Not enough anchor teeth for ideal arch")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    if save_steps:
        print("\n[Step 7] Generating step-by-step figures...")
        saved = generate_all_steps(
            matches=matches, inv=inv, tf=tf, mt=mt,
            ctrl_pts=ctrl_pts, crv=crv, bx=bx, by=by,
            dev_rows=dev_rows, ideal_crv=ideal_crv, ideal_dev=ideal_dev,
            image_name=image_name, save_dir=steps_dir,
        )
        print(f"\n  Saved {len(saved)} figures → {steps_dir.relative_to(ROOT)}/")

    print("\n" + "=" * 66)
    print("  ✅  เสร็จสมบูรณ์!")
    print(f"  📄  CSV   → {exports_dir.relative_to(ROOT)}/")
    if save_steps:
        print(f"  🖼   ภาพ  → {steps_dir.relative_to(ROOT)}/")
    print("=" * 66 + "\n")

    return {
        "matches": matches, "inv": inv, "tf": tf, "mt": mt,
        "ctrl_pts": ctrl_pts, "crv": crv, "bx": bx, "by": by,
        "dev_rows": dev_rows, "ideal_crv": ideal_crv, "ideal_dev": ideal_dev,
        "image_name": image_name,
    }


# =============================================================================
# Interactive image picker
# =============================================================================

def _scan_images():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in INPUT_DIR.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)


def _pick_image():
    images = _scan_images()
    print()
    print("=" * 60)
    print("  Dental Arch Analysis System")
    print("  AI-Driven Occlusal Surface Extraction")
    print("=" * 60)

    if not images:
        print("""
  [!] ไม่พบภาพใน  data/input/

  วิธีเพิ่มภาพ:
    1. เปิดโฟลเดอร์  data/input/
    2. วางภาพ Lower Occlusal View (.jpg / .png)
    3. รัน  python run.py  อีกครั้ง
""")
        input("  กด Enter เพื่อออก...")
        return None

    print(f"\n  พบ {len(images)} ภาพใน  data/input/\n")
    for i, p in enumerate(images, 1):
        kb = p.stat().st_size // 1024
        print(f"    [{i:>2}]  {p.name:<40}  {kb:>5} KB")
    print()
    print("    [A]  วิเคราะห์ทุกภาพ (batch)")
    print("    [Q]  ออกจากโปรแกรม")
    print()

    n = len(images)
    while True:
        raw = input(f"  เลือก (1–{n} / A / Q): ").strip().upper()
        if raw == "Q":
            return None
        if raw == "A":
            return "ALL"
        try:
            idx = int(raw) - 1
            if 0 <= idx < n:
                return images[idx]
        except ValueError:
            pass
        print(f"    กรุณาพิมพ์ 1–{n}, A หรือ Q")


def _run_one(img_path):
    try:
        return run_pipeline(image_path=str(img_path), save_steps=True) is not None
    except SystemExit:
        return False
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        import traceback; traceback.print_exc()
        return False


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dental Arch Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่าง:
  python run.py                              # interactive mode
  python run.py --image data/input/x.jpg    # ระบุภาพโดยตรง
  python run.py --csv   data/samples/t.csv  # CSV mode (ไม่ใช้ YOLO)
  python run.py --batch                     # วิเคราะห์ทุกภาพใน data/input/
""")
    parser.add_argument("--image",  type=str, metavar="PATH",
                        help="Path ของภาพ (.jpg/.png)")
    parser.add_argument("--csv",    type=str, metavar="PATH",
                        help="Path ของ CSV keypoints")
    parser.add_argument("--batch",  action="store_true",
                        help="วิเคราะห์ทุกภาพใน data/input/")
    parser.add_argument("--no-steps", action="store_true",
                        help="ไม่บันทึกภาพ step-by-step")
    args = parser.parse_args()

    save_steps = not args.no_steps

    if args.image:
        run_pipeline(image_path=args.image, save_steps=save_steps)
        return

    if args.csv:
        run_pipeline(csv_path=args.csv, save_steps=save_steps)
        return

    if args.batch:
        images = _scan_images()
        if not images:
            print("  [!] ไม่พบภาพใน data/input/")
            return
        ok = 0
        print(f"\n  Batch mode: {len(images)} ภาพ\n")
        for img in images:
            print(f"\n  ── {img.name} ──")
            if _run_one(img):
                ok += 1
        print(f"\n  Batch เสร็จ: {ok}/{len(images)} ภาพสำเร็จ")
        return

    # interactive mode
    choice = _pick_image()
    if choice is None:
        sys.exit(0)
    if choice == "ALL":
        images = _scan_images()
        ok = sum(_run_one(img) for img in images)
        print(f"\n  Batch เสร็จ: {ok}/{len(images)} ภาพสำเร็จ")
    else:
        print(f"\n  ► วิเคราะห์:  {choice.name}\n")
        _run_one(choice)


if __name__ == "__main__":
    main()
