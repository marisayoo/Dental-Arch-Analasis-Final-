"""
config/settings.py
ค่าคงที่และการตั้งค่าทั้งหมดของระบบ
แก้ไขที่นี่ที่เดียวเพื่อเปลี่ยนพฤติกรรมของทั้งระบบ
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data" / "samples"
INPUT_DIR   = ROOT_DIR / "data" / "input"
OUTPUT_DIR  = ROOT_DIR / "output"
CASES_DIR = OUTPUT_DIR / "cases"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ── Model paths ───────────────────────────────────────────────────────────────
SEG_MODEL_PATH  = ROOT_DIR / "tooth_ver1.onnx"
POSE_MODEL_PATH = ROOT_DIR / "keypoint_ver1.onnx"

# ── Run mode ──────────────────────────────────────────────────────────────────
MODE_CSV = True      # True = CSV, False = YOLO model
CONF     = 0.2       # YOLO confidence threshold

# ── ISO Tooth Definitions (FDI) — Lower Arch ──────────────────────────────────
ALL_TEETH_R = [f"R3{i}" for i in range(1, 8)]   # R31-R37
ALL_TEETH_L = [f"L4{i}" for i in range(1, 8)]   # L41-L47
ALL_TEETH   = ALL_TEETH_R + ALL_TEETH_L          # 14 teeth total

# ── YOLO Keypoint indices ──────────────────────────────────────────────────────
KP_DMR      = 0   # Distal Marginal Ridge
KP_CENTROID = 1   # Centroid of tooth
KP_MMR      = 2   # Mesial Marginal Ridge
KP_ANC      = 3   # Anchor point

# ── Triangular Method ─────────────────────────────────────────────────────────
# Reference: Li, Gateno, Xia (2017) Int J Oral Maxillofac Surg 46(7):974-981
# U6 = mesiobuccal cusp of 1st/2nd molar (priority order)
TRIANGULAR_MOLAR_R = ["R37", "R36"]
TRIANGULAR_MOLAR_L = ["L47", "L46"]

# ── Ideal Arch Anchors ────────────────────────────────────────────────────────
# Reference: Andrews (1972) Am J Orthod 62(3): six keys to normal occlusion
# Molar = stable anchor, Incisor = arch midpoint reference
IDEAL_ARCH_ANCHORS = ["R37", "R36", "R31", "L41", "L46", "L47"]

# ── B-spline ──────────────────────────────────────────────────────────────────
# Reference: Fan et al. (2025) §4.2.1  degree p=2 preferred over p=3 (§5.4.2)
BSPLINE_DEGREE = 2
BSPLINE_N_EVAL = 500

# ── Descriptive Control Points ─────────────────────────────────────────────────
CP_TARGETS  = ["R37","R35","R33","R31","L41","L43","L45","L47"]
CP_FALLBACK = {
    "R31": ["R32"], "L41": ["L42"],
    "R33": ["R32","R34"], "L43": ["L42","L44"],
    "R35": ["R34","R36"], "L45": ["L44","L46"],
    "R37": ["R36"],       "L47": ["L46"],
}

# ── Validation thresholds ─────────────────────────────────────────────────────
MIN_TEETH_FOR_ANALYSIS = 4
MIN_TEETH_FOR_BSPLINE  = 4

# ── Figure output ─────────────────────────────────────────────────────────────
FIGURE_DPI    = 180
STEP_FIG_DPI  = 150
REPORT_DPI    = 300

# ── Color palette (consistent across all figures) ─────────────────────────────
COLORS = dict(
    mmr       = "#1A7A4A",
    dmr       = "#B03A2E",
    centroid  = "#2E75B6",
    anc       = "#9B59B6",
    arch_desc = "#D4860A",
    arch_ideal= "#2E75B6",
    cp_desc   = "#7D3C98",
    cp_ideal  = "#C0392B",
    anchor_pt = "#D4860A",
    dev_mmr   = "#0D6E47",
    dev_dmr   = "#8B1A1A",
    grid      = "#F0F0F0",
    panel_bg  = "#F7FAFD",
    navy      = "#1F3864",
    move_high = "#A32D2D",
    move_med  = "#854F0B",
    move_low  = "#0F6E56",
)

# ── Aliases for backward compatibility ───────────────────────────────────────
CP_PRIMARY     = ["R31", "L41"]
KP_MMR         = 2   # already defined above but repeated for clarity

# Backward compat
IMAGE_PATH = DATA_DIR / "sample.jpg"

# CSV export fields
DEVIATION_CSV_FIELDS = [
    "tooth_id", "iso_number", "side", "status",
    "mmr_tx_px", "mmr_ty_px", "mmr_closest_x", "mmr_closest_y",
    "mmr_dist_px", "mmr_delta_x_px", "mmr_delta_y_px",
    "dmr_tx_px", "dmr_ty_px", "dmr_closest_x", "dmr_closest_y",
    "dmr_dist_px", "dmr_delta_x_px", "dmr_delta_y_px",
    "mean_mr_dist_px",
]
