"""
controllers/data_loader.py — Controller Layer (I/O)
หน้าที่: รับ input จากภายนอก (CSV หรือ YOLO model) แล้ว normalize เป็นรูปแบบเดียวกัน
แยก I/O ออกจาก Model เพื่อให้ test ได้ง่าย
"""
from __future__ import annotations
import csv

from config.settings import (
    SEG_MODEL_PATH, POSE_MODEL_PATH, IMAGE_PATH, CONF,
    KP_DMR, KP_CENTROID, KP_MMR, KP_ANC,
)
from models.tooth_data import (
    validate_tooth_id, validate_keypoints, deduplicate_matches, tooth_sort_key
)


# =============================================================================
# Loader: CSV
# =============================================================================
def load_from_csv(csv_path: str = "tooth_output.csv") -> list:
    """
    อ่านข้อมูลจาก tooth_output.csv
    ทุก row ผ่านการ validate ก่อนใช้งาน

    Format ที่คาดหวัง (header):
        tooth_id, mmr_x, mmr_y, dmr_x, dmr_y,
        centroid_x, centroid_y, anc_x, anc_y

    Returns
    -------
    list of dict: [{ "tooth_class": str, "keypoints": [[dmr],[cen],[mmr],[anc]] }, ...]
    เรียงตาม tooth_sort_key (31/41 → 37/47)
    """
    raw     = []
    skipped = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ข้าม comment line
            if row.get("tooth_id", "").startswith("#"):
                continue

            tooth = row["tooth_id"].strip()

            if not validate_tooth_id(tooth):
                skipped.append((tooth, "ไม่ใช่ฟัน Lower arch"))
                continue

            try:
                mmr = (int(row["mmr_x"]),      int(row["mmr_y"]))
                dmr = (int(row["dmr_x"]),      int(row["dmr_y"]))
                cen = (int(row["centroid_x"]), int(row["centroid_y"]))
                anc = (int(row["anc_x"]),      int(row["anc_y"]))
            except (ValueError, KeyError) as e:
                skipped.append((tooth, f"พิกัดผิดรูปแบบ: {e}"))
                continue

            # keypoints อยู่ใน order [DMR, Centroid, MMR, ANC]
            kp_list = [dmr, cen, mmr, anc]

            raw.append({"tooth_class": tooth, "keypoints": kp_list})

    if skipped:
        print("\n[CSV Loader] รายการที่ถูกข้าม:")
        for tid, reason in skipped:
            print(f"  - {tid}: {reason}")

    return deduplicate_matches(raw)


# =============================================================================
# Loader: YOLO Model
# =============================================================================
def load_from_model(image_path: str = IMAGE_PATH) -> tuple:
    """
    รัน YOLO Segmentation + Pose บนภาพ แล้ว match keypoints กับซี่ฟัน
    บันทึกผลเป็น tooth_output.csv ด้วย

    Returns
    -------
    img_rgb    : np.ndarray  (H, W, 3) RGB
    result_img : np.ndarray  (H, W, 3) BGR  (YOLO overlay)
    matches    : list of dict
    """
    # ── import เฉพาะเมื่อจำเป็น (ไม่ต้อง import ถ้า MODE_CSV=True) ──────────
    import cv2
    from ultralytics import YOLO

    seg_model  = YOLO(SEG_MODEL_PATH,  task="segment")
    pose_model = YOLO(POSE_MODEL_PATH, task="pose")

    seg_r  = seg_model.predict(image_path,  conf=CONF)[0]
    pose_r = pose_model.predict(image_path, conf=CONF)[0]

    print(f"[SEG]  detections: {len(seg_r.boxes) if seg_r.boxes else 0}")
    print(f"[POSE] detections: {len(pose_r.boxes) if pose_r.boxes else 0}")

    # ── Draw overlay ──────────────────────────────────────────────────────────
    result_img = seg_r.plot()
    if pose_r.keypoints is not None:
        kpts_all = pose_r.keypoints.xy.cpu().numpy()
        for obj in kpts_all:
            for i, (x, y) in enumerate(obj):
                x, y = int(x), int(y)
                cv2.circle(result_img, (x, y), 4, (255, 0, 0), -1)
                cv2.putText(result_img, str(i), (x + 3, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── Read original image ───────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Match seg-box ↔ pose keypoints ───────────────────────────────────────
    tooth_boxes   = seg_r.boxes.xyxy.cpu().numpy()
    tooth_classes = seg_r.boxes.cls.cpu().numpy().astype(int)
    kpts_all      = pose_r.keypoints.xy.cpu().numpy()
    box_names     = seg_model.names

    def point_in_box(x, y, box):
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    raw = []
    for obj_kpts in kpts_all:
        best_box, best_score = -1, -1
        for box_id, box in enumerate(tooth_boxes):
            count = sum(1 for x, y in obj_kpts if point_in_box(x, y, box))
            if count > best_score:
                best_score, best_box = count, box_id

        if best_score < 2:
            continue

        tooth_id = box_names[tooth_classes[best_box]]
        kp_list  = [(int(x), int(y)) for x, y in obj_kpts]
        # YOLO order: 0=DMR, 1=Centroid, 2=MMR, 3=ANC

        raw.append({"tooth_class": tooth_id, "keypoints": kp_list})

    matches = deduplicate_matches(raw)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    _save_raw_csv(matches, "tooth_output.csv")

    return img_rgb, result_img, matches


# =============================================================================
# CSV Export: raw keypoints
# =============================================================================
def _save_raw_csv(matches: list, save_path: str = "tooth_output.csv") -> None:
    """บันทึก raw keypoints เป็น CSV (tooth_output.csv)"""
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tooth_id",
            "mmr_x", "mmr_y",
            "dmr_x", "dmr_y",
            "centroid_x", "centroid_y",
            "anc_x", "anc_y",
        ])
        for m in matches:
            kp = m["keypoints"]   # [DMR, Centroid, MMR, ANC]
            dmr, centroid, mmr, anc = kp[0], kp[1], kp[2], kp[3]
            writer.writerow([
                m["tooth_class"],
                mmr[0], mmr[1],
                dmr[0], dmr[1],
                centroid[0], centroid[1],
                anc[0], anc[1],
            ])
    print(f"[CSV] tooth_output.csv saved ({len(matches)} teeth)")
