"""
main.py — Entry Point  (dental_arch_v3)
รันไฟล์นี้เพื่อเริ่มระบบทั้งหมด

Usage:
    python main.py                              # CSV mode, default sample
    python main.py --csv data/samples/my.csv   # CSV custom file
    python main.py --image photo.jpg           # YOLO mode
    python main.py --steps                     # also save step-by-step figures
"""
import sys, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    MODE_CSV, DATA_DIR, OUTPUT_DIR,CASES_DIR, REPORTS_DIR, MIN_TEETH_FOR_ANALYSIS,
)
from models.tooth_data  import check_tooth_inventory
from models.geometry    import (
    compute_transform, transform_all,
    select_control_points, fit_bspline, compute_deviations,
    compute_ideal_arch, compute_ideal_deviations,
)
from controllers.data_loader import load_from_csv, load_from_model
from controllers.exporter    import (
    print_inventory_report, print_deviation_table,
    export_deviation_csv,
)
from views.step_figures import generate_all_steps


# ── helpers ────────────────────────────────────────────────────────────────────
def _export_ideal_csv(ideal_dev, tf, img_name, save_path):
    import csv
    from datetime import datetime
    fields = ["tooth_id","side","status","is_anchor_tooth",
              "ideal_mmr_dev_px","ideal_mmr_dx_px","ideal_mmr_dy_px",
              "ideal_dmr_dev_px","ideal_dmr_dx_px","ideal_dmr_dy_px",
              "ideal_mean_dev_px","movement_needed_px"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        meta = [
            "# Ideal Arch Deviation Report",
            f"# Image    : {img_name}",
            f"# Generated: {now}",
            "# Ideal Arch: Individualized (molar + incisor anchors only)",
            "# Anchor teeth: R37/R36, R31, L41, L46/L47",
            f"# Method   : {tf.get('method','Triangular')}",
            "# Unit     : pixel (canonical frame) — no mm calibration yet",
            "# Note     : movement_needed_px = estimated displacement",
            "#            to reach ideal arch position",
            "# Ref      : Andrews (1972); Fan et al. (2025)",
            "#",
        ]
        for line in meta: f.write(line + "\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ideal_dev:
            if r["status"] == "present":
                writer.writerow({
                    "tooth_id": r["tooth"], "side": r["side"],
                    "status": "present",
                    "is_anchor_tooth": "YES" if r["is_anchor"] else "no",
                    "ideal_mmr_dev_px": r["ideal_mmr_dev"],
                    "ideal_mmr_dx_px":  r["ideal_mmr_dx"],
                    "ideal_mmr_dy_px":  r["ideal_mmr_dy"],
                    "ideal_dmr_dev_px": r["ideal_dmr_dev"],
                    "ideal_dmr_dx_px":  r["ideal_dmr_dx"],
                    "ideal_dmr_dy_px":  r["ideal_dmr_dy"],
                    "ideal_mean_dev_px":   r["ideal_mean_dev"],
                    "movement_needed_px":  r["movement_needed_px"],
                })
            else:
                writer.writerow({
                    "tooth_id": r["tooth"], "side": r["side"],
                    "status": "MISSING",
                    "is_anchor_tooth": "YES" if r["is_anchor"] else "no",
                    **{k: "MISSING" for k in fields[4:]},
                })


# ── main pipeline ──────────────────────────────────────────────────────────────
def run(csv_path=None, image_path=None, save_steps=False):
    import numpy as np

    print("=" * 66)
    print("  AI-Driven Occlusal Surface Extraction  v3")
    print("  Dental Arch Template + Clear Aligner Deviation Analysis")
    print("=" * 66)

    # ensure output dirs exist
    for d in [OUTPUT_DIR, CASES_DIR, REPORTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if csv_path:
        print(f"\n[Mode] CSV  -> {csv_path}")
        matches    = load_from_csv(str(csv_path))
        img_rgb    = None; result_img = None
        image_name = Path(csv_path).stem
    elif image_path:
        print(f"\n[Mode] YOLO -> {image_path}")
        img_rgb, result_img, matches = load_from_model(str(image_path))
        image_name = Path(image_path).stem

        case_dir = CASES_DIR / image_name
        steps_dir = case_dir / "steps"
        exports_dir = case_dir / "exports"

        steps_dir.mkdir(parents=True, exist_ok=True)
        exports_dir.mkdir(parents=True, exist_ok=True)
    else:
        default_csv = DATA_DIR / "tooth_output.csv"
        print(f"\n[Mode] CSV (default) -> {default_csv}")
        matches    = load_from_csv(str(default_csv))
        img_rgb    = None; result_img = None
        image_name = "sample"

    # ── 2. Inventory ──────────────────────────────────────────────────────────
    inv = check_tooth_inventory(matches)
    print_inventory_report(inv, image_name)
    if inv["n_present"] < MIN_TEETH_FOR_ANALYSIS:
        print(f"\n[STOP] Not enough teeth ({inv['n_present']})")
        sys.exit(1)

    # ── 3. Canonical Transform (Triangular Method) ────────────────────────────
    print("\n[Step 3] Computing canonical frame (Triangular Method)...")
    tf = compute_transform(matches)
    print(f"  Origin    : ({tf['tx']:.1f}, {tf['ty']:.1f}) px")
    print(f"  Angle     : {tf['angle_deg']:.3f}°")
    print(f"  Molar R   : {tf['molar_r_used']}")
    print(f"  Molar L   : {tf['molar_l_used']}")
    print(f"  Method    : {tf['method']}")
    mt = transform_all(matches, tf)

    # ── 4. Control Points + B-spline ─────────────────────────────────────────
    print("\n[Step 4] Fitting B-spline arch curve (p=2)...")
    ctrl_pts         = select_control_points(mt)
    crv, bx, by      = fit_bspline(ctrl_pts, degree=2)
    print(f"  Control pts: {len(ctrl_pts)}")
    print(f"  Curve pts  : 500")
    print(f"  Arch width : {crv[:,0].max()-crv[:,0].min():.0f} px")

    # ── 5. Deviation (descriptive) ────────────────────────────────────────────
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
        ideal_dev = compute_ideal_deviations(mt, ibx, iby, inv)
        pres_ideal = [r for r in ideal_dev if r["status"]=="present"]
        non_anch   = [r for r in pres_ideal if not r.get("is_anchor")]
        mov_vals   = [r["movement_needed_px"] for r in non_anch]

        print(f"\n  {'Tooth':<6} {'Anchor':8} {'MMR->ideal':>11} "
              f"{'DMR->ideal':>11} {'Move(px)':>10}")
        print("  " + "-"*52)
        for r in ideal_dev:
            if r["status"] == "present":
                a = "★ anchor" if r["is_anchor"] else ""
                print(f"  {r['tooth']:<6} {a:<8} "
                      f"{r['ideal_mmr_dev']:>11.1f} "
                      f"{r['ideal_dmr_dev']:>11.1f} "
                      f"{r['movement_needed_px']:>10.1f}")
            else:
                print(f"  {r['tooth']:<6} {'':8} {'MISSING':>11}")
        print("  " + "-"*52)
        if mov_vals:
            import numpy as np
            print(f"  {'MEAN (non-anchor)':<30} "
                  f"{np.mean(mov_vals):>10.1f} px")
        csv_ideal = exports_dir / f"ideal_deviation_{image_name}.csv"
        _export_ideal_csv(ideal_dev, tf, image_name, str(csv_ideal))
        print(f"\n  ★ = anchor teeth (stable, no movement needed)")
        print(f"  movement_needed_px = estimated displacement to ideal position")
        print(f"  NOTE: unit is pixel — mm calibration required for clinical use")
    else:
        print("  [Skip] Not enough anchor teeth for ideal arch")

    # ── 7. Step-by-step figures ───────────────────────────────────────────────
    if save_steps:
        print("\n[Step 7] Generating step-by-step figures...")
        saved = generate_all_steps(
            matches=matches, inv=inv, tf=tf, mt=mt,
            ctrl_pts=ctrl_pts, crv=crv, bx=bx, by=by,
            dev_rows=dev_rows,
            ideal_crv=ideal_crv, ideal_dev=ideal_dev,
            image_name=image_name,
            save_dir=steps_dir,
        )
        print(f"\n  Saved {len(saved)} figures to {steps_dir}/")
        for p in saved:
            print(f"    {p.name}")

    print("\n" + "=" * 66)
    print("  Output files:")
    print(f"    {csv_desc.name}  ->  {exports_dir}/")
    if ideal_result:
        print(f"    {csv_ideal.name}  ->  {exports_dir}/")
    if save_steps:
        print(f"    step1–7 figures  ->  {steps_dir}/")
    print("=" * 66)
    print("\n✅  Done.\n")

    return {
        "matches":   matches, "inv": inv, "tf": tf,
        "mt":        mt,      "ctrl_pts": ctrl_pts,
        "crv":       crv,     "bx": bx, "by": by,
        "dev_rows":  dev_rows,
        "ideal_crv": ideal_crv, "ideal_dev": ideal_dev,
        "image_name": image_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI-Driven Occlusal Surface Extraction v3")
    parser.add_argument("--csv",   type=str, help="Path to CSV file")
    parser.add_argument("--image", type=str, help="Path to image file (YOLO mode)")
    parser.add_argument("--steps", action="store_true",
                        help="Save step-by-step figures")
    args = parser.parse_args()
    run(csv_path  = args.csv,
        image_path= args.image,
        save_steps= args.steps)
