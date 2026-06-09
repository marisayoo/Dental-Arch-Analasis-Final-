"""
models/tooth_data.py — Model Layer
หน้าที่: ตรวจสอบ validate และจัดการข้อมูลฟัน
ไม่มี I/O, ไม่มี visualization — บริสุทธิ์เป็น data logic
"""
from __future__ import annotations
from config.settings import (
    ALL_TEETH, CP_PRIMARY, CP_FALLBACK,
    MIN_TEETH_FOR_ANALYSIS,
)


# =============================================================================
# Sorting
# =============================================================================
def tooth_sort_key(tooth_id: str) -> tuple:
    """
    เรียงฟันจากกลาง (31/41) → ด้านหลัง (37/47)
    ฝั่งขวา (R) ก่อนฝั่งซ้าย (L)
    """
    side = tooth_id[0]
    num  = int(tooth_id[1:])
    return (0 if side == "R" else 1, num % 10)


# =============================================================================
# Validation
# =============================================================================
def validate_tooth_id(tooth_id: str) -> bool:
    """ตรวจว่า tooth_id เป็นฟัน Lower arch ที่รู้จัก (R31-R37, L41-L47)"""
    return tooth_id in ALL_TEETH


def validate_keypoints(kp_list: list) -> bool:
    """
    ตรวจว่า keypoints ไม่เป็น (0,0) ทั้งหมด
    (0,0) = model ไม่ได้ detect จริง
    """
    valid = [(x, y) for x, y in kp_list if not (x == 0 and y == 0)]
    return len(valid) >= 2


# =============================================================================
# Inventory Check
# =============================================================================
def check_tooth_inventory(matches: list) -> dict:
    """
    ตรวจสอบสภาพฟันทั้งหมด เปรียบเทียบกับ ALL_TEETH (14 ซี่)

    Parameters
    ----------
    matches : list of dict
        แต่ละ dict มี key "tooth_class" และ "keypoints"

    Returns
    -------
    dict ที่มี:
        present      – list ของ tooth_ids ที่มีข้อมูล
        missing      – list ของ tooth_ids ที่หายไป
        invalid      – list ของ tooth_ids ที่ไม่รู้จัก
        n_present    – จำนวนซี่ที่มี
        n_missing    – จำนวนซี่ที่หาย
        cp_primary_ok – bool: R31 และ L41 มีครบไหม
        warnings     – list ของ str ข้อความเตือน
    """
    present_ids = {m["tooth_class"] for m in matches}
    warnings    = []

    present = sorted(
        [t for t in present_ids if validate_tooth_id(t)],
        key=tooth_sort_key,
    )
    invalid = sorted([t for t in present_ids if not validate_tooth_id(t)])
    missing = sorted(
        [t for t in ALL_TEETH if t not in present_ids],
        key=tooth_sort_key,
    )

    cp_primary_ok = all(t in present_ids for t in CP_PRIMARY)

    if invalid:
        warnings.append(f"tooth_id ที่ไม่รู้จัก (จะถูกข้าม): {invalid}")

    if not cp_primary_ok:
        missing_cp = [t for t in CP_PRIMARY if t not in present_ids]
        warnings.append(
            f"ฟัน anchor หลักขาดหาย: {missing_cp} "
            f"— จำเป็นสำหรับ Canonical Transform"
        )

    if missing:
        warnings.append(f"ฟันที่ขาดหาย ({len(missing)} ซี่): {missing}")

    if len(present) < MIN_TEETH_FOR_ANALYSIS:
        warnings.append(
            f"จำนวนฟันน้อยเกินไป ({len(present)} ซี่) "
            f"ต้องการอย่างน้อย {MIN_TEETH_FOR_ANALYSIS} ซี่"
        )

    return {
        "present":       present,
        "missing":       missing,
        "invalid":       invalid,
        "n_present":     len(present),
        "n_missing":     len(missing),
        "cp_primary_ok": cp_primary_ok,
        "warnings":      warnings,
    }


# =============================================================================
# Match De-duplication Helper
# =============================================================================
def deduplicate_matches(raw_matches: list) -> list:
    """
    ถ้า tooth_id เดียวกัน detect มาหลายครั้ง → เก็บตัวแรก ทิ้งตัวที่เหลือ
    Returns sorted list
    """
    seen    = {}
    skipped = []
    for m in raw_matches:
        tid = m["tooth_class"]
        if not validate_tooth_id(tid):
            skipped.append((tid, "ไม่ใช่ฟัน Lower arch"))
            continue
        if not validate_keypoints(m["keypoints"]):
            skipped.append((tid, "keypoints เป็น (0,0) ทั้งหมด"))
            continue
        if tid in seen:
            skipped.append((tid, "duplicate — ใช้ตัวแรก"))
            continue
        seen[tid] = m

    if skipped:
        print("\n[Validate] รายการที่ถูกข้าม:")
        for tid, reason in skipped:
            print(f"  - {tid}: {reason}")

    return sorted(seen.values(), key=lambda m: tooth_sort_key(m["tooth_class"]))
