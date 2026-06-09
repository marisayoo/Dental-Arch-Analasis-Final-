"""
views/step_figures.py — View Layer: Step-by-Step Pipeline Figures
สร้างภาพอธิบาย pipeline ทีละ step พร้อมค่าจริง
ออกแบบให้ผู้ใช้เข้าใจได้ทันที ไม่ต้องอ่าน code

Layout rules (ไม่แตะ logic/การคำนวณใดๆ):
  • _step_title() → fig.suptitle() กึ่งกลาง บรรทัดเดียว ด้านบน
  • _side_panel()  → info box วาง figure margin ขวา (นอก axes ทั้งหมด)
  • axes ถูก crop ให้จบที่ AXES_RIGHT = 0.72 เพื่อเปิด margin ขวา 28%
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from config.settings import COLORS, STEP_FIG_DPI

C = COLORS
NAVY = C["navy"]

# ── Layout constants ───────────────────────────────────────────────────────────
AXES_RIGHT  = 0.72   # axes สิ้นสุดที่ 72% ของ figure width
PANEL_X     = 0.745  # info box เริ่มที่ 74.5% (ซ้ายของ panel)
PANEL_Y_TOP = 0.88   # info box top edge (อยู่ใต้ suptitle)


# ── helpers ────────────────────────────────────────────────────────────────────
def _sci(ax, xl="", yl="", equal=True, minor=False):
    ax.set_facecolor(C["panel_bg"])
    ax.set_xlabel(xl, fontsize=8, color="#444", labelpad=3)
    ax.set_ylabel(yl, fontsize=8, color="#444", labelpad=3)
    ax.tick_params(labelsize=7.5, length=3, color="#AAA", labelcolor="#444")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#CCCCCC"); ax.spines[s].set_linewidth(0.8)
    ax.grid(color=C["grid"], lw=0.6, zorder=0)
    ax.axhline(0, color="#DDDDDD", lw=0.8, zorder=1)
    ax.axvline(0, color="#DDDDDD", lw=0.8, zorder=1)
    if equal: ax.set_aspect("equal", adjustable="datalim")
    if minor:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax.grid(which="minor", color="#F6F6F6", lw=0.35, zorder=0)


def _step_title(fig, n: int, title: str, color=None):
    """หัวข้อ STEP — fig.suptitle() กึ่งกลาง บรรทัดเดียว ด้านบน ไม่ซ้อน axes"""
    c = color or NAVY
    fig.suptitle(
        f"STEP {n}  —  {title}",
        fontsize=11, fontweight="bold", color=c,
        x=AXES_RIGHT / 2,   # กึ่งกลางของ axes area (ไม่รวม panel ขวา)
        y=0.97, ha="center", va="top",
    )


def _side_panel(fig, lines: list[str], ec=None, fc="#EBF3FB", fs=7.5):
    """
    Info box วางใน figure margin ขวา — นอก axes ทั้งหมด
    ใช้ fig.text() ไม่ใช่ ax.text() จึงไม่มีทางทับ plot content
    """
    ec = ec or NAVY
    fig.text(
        PANEL_X, PANEL_Y_TOP,
        "\n".join(lines),
        fontsize=fs, va="top", ha="left", color="#111",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=fc, edgecolor=ec,
            alpha=0.97, lw=0.9,
        ),
        transform=fig.transFigure,
        wrap=False,
    )


def _ra(ax, foot, pt, sz=5, col="#333"):
    """Right-angle marker at foot of perpendicular"""
    vm = np.array([pt[0]-foot[0], pt[1]-foot[1]])
    vl = np.linalg.norm(vm)
    if vl < 0.5: return
    vn = vm / vl * sz
    pn = np.array([-vn[1], vn[0]])
    sq = np.array([[foot[0], foot[1]], [foot[0]+vn[0], foot[1]+vn[1]],
                   [foot[0]+vn[0]+pn[0], foot[1]+vn[1]+pn[1]],
                   [foot[0]+pn[0], foot[1]+pn[1]]])
    ax.plot(np.append(sq[:, 0], sq[0, 0]),
            np.append(sq[:, 1], sq[0, 1]),
            "-", color=col, lw=0.9, alpha=0.8, zorder=6)


def fy(v): return float(v) * -1   # Y-flip: posterior down, anterior up


def _make_fig_ax(figsize, equal=True, minor=False,
                 xl="", yl=""):
    """
    สร้าง figure + axes โดย axes จบที่ AXES_RIGHT
    เพื่อเปิด margin ขวาสำหรับ _side_panel()
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.subplots_adjust(
        left=0.08, right=AXES_RIGHT,
        top=0.90,  bottom=0.09,
    )
    _sci(ax, xl, yl, equal=equal, minor=minor)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
def generate_all_steps(
    matches, inv, tf, mt, ctrl_pts, crv, bx, by,
    dev_rows, ideal_crv, ideal_dev,
    image_name: str = "sample",
    save_dir: Path | None = None,
) -> list[Path]:
    """
    สร้างภาพ step-by-step ทั้งหมด 7 ขั้น
    แต่ละ step บันทึกเป็นไฟล์แยก
    Returns list ของ Path ที่บันทึก
    """
    if save_dir is None:
        raise ValueError("save_dir is required")
    sd = Path(save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    saved = []

    pm    = {m["tooth_class"]: m for m in mt}
    crv_f = np.c_[crv[:, 0],      [fy(y) for y in crv[:, 1]]]
    ctrl_f= np.c_[ctrl_pts[:, 0], [fy(y) for y in ctrl_pts[:, 1]]]
    pres  = [r for r in dev_rows  if r["status"] == "present"]
    ipres = [r for r in ideal_dev if r["status"] == "present"]
    mmr_d = [r["mmr_dev"] for r in pres]
    dmr_d = [r["dmr_dev"] for r in pres]

    icrv_f = (np.c_[ideal_crv[:, 0], [fy(y) for y in ideal_crv[:, 1]]]
               if ideal_crv is not None else None)

    def _fill(ax, pts, alpha=0.06):
        verts = np.vstack([pts, [pts[-1, 0], -10], [pts[0, 0], -10]])
        ax.fill(verts[:, 0], verts[:, 1],
                color=C["arch_desc"], alpha=alpha, zorder=0)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Raw keypoints (image space)
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = _make_fig_ax((11, 7),
                           xl="x  (pixel, image space)",
                           yl="y  (pixel, image space)")
    ax.invert_yaxis()
    cmap = plt.cm.tab20(np.linspace(0, 1, 14))
    for i, m in enumerate(matches):
        kp = m["keypoints"]
        cen, mmr_, dmr_ = kp[1], kp[2], kp[0]
        ax.plot([mmr_[0], cen[0], dmr_[0]], [mmr_[1], cen[1], dmr_[1]],
                "-", color="#CCCCCC", lw=0.7, alpha=0.7, zorder=1)
        ax.scatter(*cen, c=[cmap[i]], s=22, zorder=5, edgecolors="white", lw=0.8)
        ax.scatter(*mmr_, c=C["mmr"], s=18, marker="^", zorder=6, alpha=0.9)
        ax.scatter(*dmr_, c=C["dmr"], s=18, marker="v", zorder=6, alpha=0.9)
        ax.annotate(m["tooth_class"][1:], cen, fontsize=6, fontweight="bold",
                    ha="center", xytext=(0, 4), textcoords="offset points",
                    color=NAVY, zorder=8)
    handles = [plt.scatter([], [], c=C["mmr"], s=20, marker="^", label="MMR"),
               plt.scatter([], [], c=C["dmr"], s=20, marker="v", label="DMR"),
               plt.Line2D([0], [0], marker="o", color="w", mfc="#666",
                          ms=6, label="Centroid (color/tooth)")]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)

    _step_title(fig, 1, "Raw YOLO Output — Keypoints in Image Pixel Space")
    _side_panel(fig, [
        "load_from_csv() /",
        "load_from_model()",
        "",
        f"  {len(matches)} teeth detected",
        "  4 keypoints per tooth:",
        "    MMR ▲  DMR ▼",
        "    Centroid ●  ANC ■",
        "",
        "  Unit: pixel",
        "  Y-axis: downward ↓",
    ], ec=C["centroid"], fc="#EBF3FB")

    p = sd / f"step1_raw_keypoints_{image_name}.png"
    plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(p)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Canonical Frame Computation
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = _make_fig_ax((11, 7),
                           xl="x  (pixel, image space)",
                           yl="y  (pixel, image space)")
    ax.invert_yaxis()
    kpm   = {m["tooth_class"]: m for m in matches}
    mmr31 = np.array(kpm["R31"]["keypoints"][2], float)
    mmr41 = np.array(kpm["L41"]["keypoints"][2], float)
    orig  = (mmr31 + mmr41) / 2
    for m in matches:
        ax.scatter(*m["keypoints"][1], c=C["centroid"], s=12, alpha=0.2, zorder=2)
    cens = np.array([m["keypoints"][1] for m in matches], float)
    mc   = cens.mean(axis=0); cc = cens - mc
    cov  = cc.T @ cc / len(cc); ev, evec = np.linalg.eigh(cov)
    pv   = evec[:, np.argmax(ev)] * 200
    ax.annotate("", xy=(mc[0]+pv[0], mc[1]+pv[1]),
                xytext=(mc[0]-pv[0], mc[1]-pv[1]),
                arrowprops=dict(arrowstyle="<->", color="#BBBBBB", lw=1.2))
    ax.text(mc[0]+pv[0]+5, mc[1]+pv[1]-5, "PCA (old, 36.7%)",
            fontsize=7, color="#BBBBBB", style="italic")
    mr = np.array(kpm.get("R37", kpm.get("R36"))["keypoints"][1], float)
    ml = np.array(kpm.get("L47", kpm.get("L46"))["keypoints"][1], float)
    ax.plot([ml[0], mr[0]], [ml[1], mr[1]], "--", color="#2E75B6", lw=2.2, alpha=0.9,
            label="Triangular transverse line")
    ax.scatter(*mr, c="#2E75B6", s=80, zorder=7, marker="D",
               edgecolors="white", lw=1.2, label="Molar R (R37)")
    ax.scatter(*ml, c="#2E75B6", s=80, zorder=7, marker="D",
               edgecolors="white", lw=1.2, label="Molar L (L47)")
    ax.plot([mmr41[0], mmr31[0]], [mmr41[1], mmr31[1]],
            "-", color="#E67E22", lw=2, alpha=0.9)
    for pt, lbl in [(mmr31, "MMR 31"), (mmr41, "MMR 41")]:
        ax.scatter(*pt, c="#E67E22", s=100, zorder=8, marker="*",
                   edgecolors="white", lw=1.2)
        ax.annotate(lbl, pt, fontsize=7, color="#E67E22", fontweight="bold",
                    xytext=(0, 10), textcoords="offset points", ha="center")
    ax.scatter(*orig, c=C["anchor_pt"], s=140, zorder=9, marker="+", linewidths=3.5)
    ax.annotate(f"Origin O\n({orig[0]:.0f}, {orig[1]:.0f})",
                orig, fontsize=7, color=C["anchor_pt"], fontweight="bold",
                xytext=(14, -20), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=C["anchor_pt"], alpha=0.92, lw=0.9))
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    _step_title(fig, 2,
        "compute_transform()  —  Canonical Frame  "
        "(Triangular Method, Li et al. 2017)")
    _side_panel(fig, [
        "Triangular Method",
        "(Li et al. 2017):",
        "",
        "  Origin O =",
        "  mid(MMR31, MMR41)",
        f"  ({orig[0]:.1f}, {orig[1]:.1f}) px",
        "",
        "  X-axis //",
        "  line(R37↔L47)",
        f"  Angle = {tf['angle_deg']:.3f}°",
        "",
        "  PCA (gray) =",
        "  36.7% — not used",
    ], ec=C["centroid"], fc="#EBF3FB")

    p = sd / f"step2_canonical_frame_{image_name}.png"
    plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(p)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Canonical coordinates (Y-flipped)
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = _make_fig_ax((11, 7.5), minor=True,
                           xl="x  (px, canonical)  ← Right (R) | Left (L) ->",
                           yl="y  (px, canonical)  Anterior ↑  |  ↓ Posterior")
    fig.subplots_adjust(left=0.09, right=AXES_RIGHT, top=0.90, bottom=0.13)
    ax.axhspan(-8, 8, color="#FFFDE7", alpha=0.9, zorder=0)
    ax.text(430, 4, "Y = 0  (Origin plane)", fontsize=7,
            color="#B8860B", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFDE7",
                      ec="#B8860B", alpha=0.9, lw=0.7))
    for m in mt:
        cx, cy   = m["centroid_t"][0], fy(m["centroid_t"][1])
        mx_, my_ = m["mmr_t"][0],      fy(m["mmr_t"][1])
        dx_, dy_ = m["dmr_t"][0],      fy(m["dmr_t"][1])
        tid = m["tooth_class"]
        ax.plot([mx_, cx, dx_], [my_, cy, dy_],
                "-", color="#DDDDDD", lw=0.9, alpha=0.9, zorder=2)
        ax.scatter(cx, cy, c=C["centroid"], s=28, zorder=5,
                   edgecolors="white", lw=0.8)
        ax.scatter(mx_, my_, c=C["mmr"], s=24, marker="^", zorder=5)
        ax.scatter(dx_, dy_, c=C["dmr"], s=24, marker="v", zorder=5)
        iso = tid[1:]
        xo  = 16 if tid[0] == "R" else -16
        ax.annotate(iso, (cx, cy), fontsize=7, fontweight="bold",
                    ha="center", xytext=(xo, 0), textcoords="offset points",
                    color=NAVY, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#A9CCE3", alpha=0.92, lw=0.7))
    ax.scatter(0, 0, s=110, c=C["anchor_pt"], marker="+",
               linewidths=3.5, zorder=9)
    ax.annotate("O (0,0)", (0, 0), fontsize=8, color=C["anchor_pt"],
                fontweight="bold", xytext=(14, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=C["anchor_pt"], alpha=0.88, lw=0.7))
    ax.text(0.5, 1.015, "Anterior  (Central Incisors 31/41)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#666", style="italic")
    ax.text(0.5, -0.08, "Posterior  (2nd Molars 37/47)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#666", style="italic")
    mmr31_t = mt[0]["mmr_t"]; mmr41_t = mt[7]["mmr_t"]

    # legend สัญลักษณ์ภายใน axes
    h3 = [plt.scatter([],[],c=C["mmr"],s=22,marker="^",label="MMR (Mesial MR)"),
          plt.scatter([],[],c=C["dmr"],s=22,marker="v",label="DMR (Distal MR)"),
          plt.scatter([],[],c=C["centroid"],s=22,marker="o",label="Centroid"),
          plt.scatter([],[],c=C["anchor_pt"],s=60,marker="+",
                      linewidths=2, label="Origin O (0,0)")]
    ax.legend(handles=h3, fontsize=7.5, loc="upper right",
              framealpha=0.92, ncol=2)

    _step_title(fig, 3, "transform_all()  —  Canonical Frame Applied")
    _side_panel(fig, [
        "transform_all()",
        "",
        "  Adds: mmr_t, dmr_t,",
        "  centroid_t, anc_t",
        "",
        f"  MMR-31 Y =",
        f"  {mmr31_t[1]:.3f} px ≈ 0 ✓",
        f"  MMR-41 Y =",
        f"  {mmr41_t[1]:.3f} px ≈ 0 ✓",
        "",
        "  R3x: x > 0",
        "  L4x: x < 0",
        "  Y-flipped display",
    ], ec="#0D6E47", fc="#E1F5EE")

    p = sd / f"step3_canonical_transformed_{image_name}.png"
    plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(p)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Control Points + B-spline
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = _make_fig_ax((11, 7),
                           xl="x  (px, canonical)",
                           yl="y  Anterior ↑")
    _fill(ax, crv_f)
    ax.plot(crv_f[:, 0], crv_f[:, 1], color=C["arch_desc"], lw=3,
            zorder=4, solid_capstyle="round",
            label="B-spline arch (p=2, 500 pts)")
    ax.plot(ctrl_f[:, 0], ctrl_f[:, 1], "--", color=C["cp_desc"],
            alpha=0.3, lw=1.3, zorder=2)
    for m in mt:
        ax.scatter(m["centroid_t"][0], fy(m["centroid_t"][1]),
                   c=C["centroid"], s=16, alpha=0.35, zorder=2)
    cp_labels = ["CP5(R37)", "CP3(R35)", "CP1(R33)", "CP0(mid)",
                 "CP2(L43)", "CP4(L45)", "CP6(L47)", "CP7"]
    cp_yoff   = [18, -20, 18, -20, 18, -20, 18, 18]
    for i, (cpx, cpy) in enumerate(ctrl_f):
        ax.scatter(cpx, cpy, c=C["cp_desc"], s=110, zorder=6,
                   edgecolors="white", lw=2, marker="D")
        lbl = cp_labels[i] if i < len(cp_labels) else f"CP{i}"
        yo  = cp_yoff[i]   if i < len(cp_yoff)   else 14
        ax.annotate(lbl, (cpx, cpy), fontsize=7, fontweight="bold",
                    color=C["cp_desc"], ha="center",
                    xytext=(0, yo), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=C["cp_desc"], alpha=0.88, lw=0.8))
    for tv, tc2 in zip(np.linspace(0, 1, 7), plt.cm.viridis(np.linspace(0, 1, 7))):
        from scipy.interpolate import make_interp_spline as mis
        n = len(ctrl_pts); deg = min(2, n-1)
        t_arr = np.linspace(0, 1, n)
        bs_x = mis(t_arr, ctrl_pts[:, 0], k=deg)
        bs_y = mis(t_arr, ctrl_pts[:, 1], k=deg)
        xi, yi = float(bs_x(tv)), fy(float(bs_y(tv)))
        ax.scatter(xi, yi, c=[tc2], s=35, zorder=7, edgecolors="white", lw=1)
    ax.scatter(0, 0, s=80, c=C["anchor_pt"], marker="+", linewidths=3, zorder=9)

    # legend สัญลักษณ์ภายใน axes
    h4 = [plt.Line2D([0],[0], color=C["arch_desc"], lw=2.5,
                     label="B-spline arch (p=2, 500 pts)"),
          plt.scatter([],[],c=C["cp_desc"],s=55,marker="D",
                      edgecolors="white",label="Control Points (CP)"),
          plt.scatter([],[],c=C["centroid"],s=18,marker="o",
                      alpha=0.5,label="Centroid (faint)"),
          plt.scatter([],[],c="gray",s=22,marker="o",
                      label="t param (viridis dots)")]
    ax.legend(handles=h4, fontsize=7.5, loc="upper right", framealpha=0.92)

    _step_title(fig, 4,
        "select_control_points() + fit_bspline()  —  "
        "Fan et al. (2025) §4.2.1  degree p=2",
        C["cp_desc"])
    _side_panel(fig, [
        "select_control_points():",
        f"  {len(ctrl_pts)} CPs:",
        "  R37,R35,R33,",
        "  R31+L41,",
        "  L43,L45,L47",
        "",
        "fit_bspline():",
        "  degree p=2",
        "  (Quadratic)",
        "  n_eval = 500 pts",
        "  t ∈ [0,1]",
        "  (viridis dots)",
        "",
        f"  Arch width:",
        f"  {crv_f[:,0].max()-crv_f[:,0].min():.0f} px",
    ], ec=C["cp_desc"], fc="#FAECE7")

    p = sd / f"step4_bspline_arch_{image_name}.png"
    plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(p)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Deviation from descriptive arch (perpendicular lines)
    # ──────────────────────────────────────────────────────────────────────────
    fig, ax = _make_fig_ax((11, 8.5), minor=True,
                           xl="x  (px, canonical)  ← R | L ->",
                           yl="y  Anterior ↑  |  Posterior ↓")
    fig.subplots_adjust(left=0.08, right=AXES_RIGHT, top=0.90, bottom=0.17)
    _fill(ax, crv_f)
    ax.plot(crv_f[:, 0], crv_f[:, 1], color=C["arch_desc"], lw=3.2,
            zorder=4, solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=5, foreground="#FEE8C0", alpha=0.4)],
            label="Descriptive B-spline arch")
    ax.scatter(ctrl_f[:, 0], ctrl_f[:, 1], s=60, color=C["cp_desc"],
               zorder=6, edgecolors="white", lw=1.5, marker="D",
               label=f"Control pts ({len(ctrl_pts)})")
    ax.scatter(0, 0, s=90, c=C["anchor_pt"], marker="+",
               linewidths=3.2, zorder=9)
    for r in pres:
        m = pm[r["tooth"]]
        mx_, my_ = m["mmr_t"][0], fy(m["mmr_t"][1])
        dx_, dy_ = m["dmr_t"][0], fy(m["dmr_t"][1])
        cx_, cy_ = m["centroid_t"][0], fy(m["centroid_t"][1])
        mcx, mcy = r["mmr_cx"], fy(r["mmr_cy"])
        dcx, dcy = r["dmr_cx"], fy(r["dmr_cy"])
        tid = r["tooth"]
        ax.plot([mx_, mcx], [my_, mcy], "-", color=C["dev_mmr"],
                lw=1.4, alpha=0.88, zorder=3, solid_capstyle="round")
        _ra(ax, (mcx, mcy), (mx_, my_), 5, C["dev_mmr"])
        ax.scatter(mcx, mcy, c=C["dev_mmr"], s=18, marker="o",
                   zorder=6, edgecolors="white", lw=0.8)
        ax.scatter(mx_, my_, c=C["mmr"], s=26, marker="^",
                   zorder=7, edgecolors="white", lw=0.9)
        xo = (8 if mx_ > mcx else -8)
        ax.annotate(f"{r['mmr_dev']:.0f}", ((mx_+mcx)/2, (my_+mcy)/2),
                    fontsize=7, color=C["dev_mmr"], fontweight="bold",
                    ha="center", xytext=(xo, 0), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec=C["dev_mmr"], alpha=0.93, lw=0.6))
        ax.plot([dx_, dcx], [dy_, dcy], "-", color=C["dev_dmr"],
                lw=1.4, alpha=0.88, zorder=3, solid_capstyle="round")
        _ra(ax, (dcx, dcy), (dx_, dy_), 5, C["dev_dmr"])
        ax.scatter(dcx, dcy, c=C["dev_dmr"], s=18, marker="o",
                   zorder=6, edgecolors="white", lw=0.8)
        ax.scatter(dx_, dy_, c=C["dmr"], s=26, marker="v",
                   zorder=7, edgecolors="white", lw=0.9)
        xo2 = (-8 if dx_ < dcx else 8)
        ax.annotate(f"{r['dmr_dev']:.0f}", ((dx_+dcx)/2, (dy_+dcy)/2),
                    fontsize=7, color=C["dev_dmr"], fontweight="bold",
                    ha="center", xytext=(xo2, 0), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec=C["dev_dmr"], alpha=0.93, lw=0.6))
        ax.scatter(cx_, cy_, c=C["centroid"], s=18, alpha=0.4, zorder=2)
        ax.annotate(tid[1:], (cx_, cy_), fontsize=7, fontweight="bold",
                    ha="center", color=NAVY,
                    xytext=(14 if tid[0]=="R" else -14, 0),
                    textcoords="offset points")
    ax.text(0.5, 1.015, "Anterior  (31 / 41)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#666", style="italic")
    ax.text(0.5, -0.10, "Posterior  (37 / 47)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#666", style="italic")
    hh = [plt.Line2D([0],[0], color=C["arch_desc"], lw=2.5, label="Descriptive arch"),
          plt.scatter([],[],c=C["mmr"],s=22,marker="^",label="MMR"),
          plt.scatter([],[],c=C["dmr"],s=22,marker="v",label="DMR"),
          plt.scatter([],[],c=C["dev_mmr"],s=18,marker="o",edgecolors="white",label="Foot MMR ⊥"),
          plt.scatter([],[],c=C["dev_dmr"],s=18,marker="o",edgecolors="white",label="Foot DMR ⊥")]
    ax.legend(handles=hh, fontsize=7.5, loc="lower center", ncol=5,
              framealpha=0.95, bbox_to_anchor=(0.5, -0.16))

    _step_title(fig, 5,
        "compute_deviations()  —  Perpendicular Distance: MR→Arch  "
        "(Fan et al. 2025 §4.3.3)",
        C["dev_mmr"])
    _side_panel(fig, [
        "compute_deviations()",
        "[Fan §4.3.3]",
        "",
        "  Line = true ⊥",
        "  □ = right-angle",
        "  ● = foot on arch",
        "  Label = dist (px)",
        "  n = 14 teeth",
        "",
        f"MMR: μ={np.mean(mmr_d):.1f} px",
        f"     σ={np.std(mmr_d):.1f}",
        f"     max={np.max(mmr_d):.1f}",
        "",
        f"DMR: μ={np.mean(dmr_d):.1f} px",
        f"     σ={np.std(dmr_d):.1f}",
        f"     max={np.max(dmr_d):.1f}",
    ], ec=C["dev_mmr"], fc="#E8F8F2")

    p = sd / f"step5_deviation_descriptive_{image_name}.png"
    plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(p)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 — Ideal Arch + Movement Needed
    # ──────────────────────────────────────────────────────────────────────────
    if icrv_f is not None:
        fig, ax = _make_fig_ax((11, 8.5), minor=True,
                               xl="x  (px, canonical)  ← R | L ->",
                               yl="y  Anterior ↑  |  Posterior ↓")
        fig.subplots_adjust(left=0.08, right=AXES_RIGHT, top=0.90, bottom=0.17)
        ax.plot(crv_f[:, 0], crv_f[:, 1], color=C["arch_desc"],
                lw=2, alpha=0.4, zorder=3, ls="--",
                label="Descriptive arch (current)")
        ax.plot(icrv_f[:, 0], icrv_f[:, 1], color=C["arch_ideal"],
                lw=3.2, zorder=5, solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=5,
                                             foreground="#B5D4F4", alpha=0.4)],
                label="Ideal arch (anchor-only)")
        for r in ipres:
            m  = pm[r["tooth"]]
            cx_, cy_ = m["centroid_t"][0], fy(m["centroid_t"][1])
            mx_, my_ = m["mmr_t"][0],      fy(m["mmr_t"][1])
            ax.scatter(mx_, my_, c=C["mmr"], s=26, marker="^",
                       zorder=6, edgecolors="white", lw=0.9)
            ax.scatter(cx_, cy_, c=C["centroid"], s=18, alpha=0.5, zorder=3)
            move = r["movement_needed_px"]
            if r["is_anchor"]:
                ax.scatter(cx_, cy_, c=C["anchor_pt"], s=55, zorder=7,
                           edgecolors="white", lw=1.5, marker="D")
            else:
                col = (C["move_high"] if move > 22 else
                       C["move_med"] if move > 10 else C["move_low"])
                icx, icy = r["ideal_mmr_cx"], fy(r["ideal_mmr_cy"])
                ax.annotate("", xy=(icx, icy), xytext=(mx_, my_),
                            arrowprops=dict(arrowstyle="->", color=col,
                                            lw=1.6, mutation_scale=12))
                ax.annotate(f"{move:.0f}px", (cx_, cy_), fontsize=7.5,
                            fontweight="bold", color=col, ha="center",
                            xytext=(0, 14 if r["tooth"][0]=="R" else -16),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.18",
                                      fc="white", ec=col, alpha=0.93, lw=0.8))
            iso = r["tooth"][1:]
            xo  = 16 if r["tooth"][0]=="R" else -16
            ax.annotate(iso, (cx_, cy_), fontsize=7, fontweight="bold",
                        ha="center", color=NAVY,
                        xytext=(xo, 0), textcoords="offset points")
        ax.scatter(0, 0, s=90, c=C["anchor_pt"], marker="+",
                   linewidths=3.2, zorder=9)
        ax.text(0.5, 1.015, "Anterior  (31 / 41)",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="#666", style="italic")
        ax.text(0.5, -0.10, "Posterior  (37 / 47)",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="#666", style="italic")
        mov_vals = [r["movement_needed_px"] for r in ipres if not r["is_anchor"]]
        hh2 = [plt.Line2D([0],[0], color=C["arch_ideal"], lw=2.5, label="Ideal arch"),
               plt.Line2D([0],[0], color=C["arch_desc"], lw=2, ls="--",
                          alpha=0.5, label="Descriptive arch"),
               plt.scatter([],[],c=C["anchor_pt"],s=45,marker="D",
                           edgecolors="white",label="★ Anchor tooth"),
               plt.scatter([],[],c=C["mmr"],s=22,marker="^",label="MMR")]
        ax.legend(handles=hh2, fontsize=7.5, loc="lower center",
                  ncol=4, framealpha=0.95, bbox_to_anchor=(0.5, -0.16))

        _step_title(fig, 6,
            "compute_ideal_arch() + compute_ideal_deviations()  —  "
            "Andrews (1972) + Fan et al. (2025)",
            C["arch_ideal"])
        _side_panel(fig, [
            "Ideal Arch:",
            "  anchor-only B-spline",
            "",
            "Anchors:",
            "  R37, R36, R31",
            "  L41, L46, L47",
            "",
            "Arrow = move (px)",
            "★ = anchor (stable)",
            "",
            f"Non-anchor mean:",
            f"  {np.mean(mov_vals):.1f} px",
            f"Max move:",
            f"  {np.max(mov_vals):.1f} px",
            "",
            "Unit: px",
            "(mm needs calib.)",
        ], ec=C["arch_ideal"], fc="#EBF3FB")

        p = sd / f"step6_ideal_arch_movement_{image_name}.png"
        plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(); saved.append(p)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 — Side-by-side bar charts
    # ──────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), facecolor="white")
    gs2 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                            left=0.05, right=0.97, top=0.85, bottom=0.13)
    ax_l = fig.add_subplot(gs2[0])
    _sci(ax_l, equal=False)
    all_teeth = [r["tooth"][1:] for r in dev_rows]
    mmr_v = [r["mmr_dev"] if r["status"]=="present" else 0 for r in dev_rows]
    dmr_v = [r["dmr_dev"] if r["status"]=="present" else 0 for r in dev_rows]
    x = np.arange(len(all_teeth)); w = 0.36
    b1 = ax_l.bar(x-w/2, mmr_v, w, color=C["mmr"], alpha=0.83,
                  edgecolor="white", lw=0.5, label="MMR->desc")
    b2 = ax_l.bar(x+w/2, dmr_v, w, color=C["dmr"], alpha=0.83,
                  edgecolor="white", lw=0.5, label="DMR->desc")
    ax_l.axhline(np.mean(mmr_d), color=C["mmr"], lw=1.4, ls="--",
                 alpha=0.8, label=f"MMR mean={np.mean(mmr_d):.1f}px")
    ax_l.axhline(np.mean(dmr_d), color=C["dmr"], lw=1.4, ls="--",
                 alpha=0.8, label=f"DMR mean={np.mean(dmr_d):.1f}px")
    ax_l.set_xticks(x); ax_l.set_xticklabels(all_teeth, fontsize=7.5, rotation=45, ha="right")
    ax_l.set_ylabel("Distance to arch curve (px)", fontsize=9)
    ax_l.set_title("Descriptive Arch Deviation per Tooth",
                   fontsize=10.5, fontweight="bold", color=NAVY)
    ax_l.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax_l.grid(axis="y", alpha=0.2); ax_l.spines[["top","right"]].set_visible(False)
    for bar, val in zip(list(b1)+list(b2), mmr_v+dmr_v):
        if val > 0:
            ax_l.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax_r = fig.add_subplot(gs2[1])
    _sci(ax_r, equal=False)
    mov_v = [r["movement_needed_px"] if r["status"]=="present" else 0
             for r in ideal_dev]
    anch  = [r.get("is_anchor", False) for r in ideal_dev]
    cols  = [C["anchor_pt"] if a else
             (C["move_high"] if v>22 else
              C["move_med"]  if v>10 else C["move_low"])
             for v, a in zip(mov_v, anch)]
    bars  = ax_r.bar(x, mov_v, 0.6, color=cols, alpha=0.85,
                     edgecolor="white", lw=0.5)
    if mov_vals:
        ax_r.axhline(np.mean(mov_vals), color="#444", lw=1.4, ls="--",
                     alpha=0.6, label=f"Non-anchor mean={np.mean(mov_vals):.1f}px")
    ax_r.set_xticks(x); ax_r.set_xticklabels(all_teeth, fontsize=7.5, rotation=45, ha="right")
    ax_r.set_ylabel("Movement needed -> Ideal arch (px)", fontsize=9)
    ax_r.set_title("Ideal Arch: Movement Needed per Tooth",
                   fontsize=10.5, fontweight="bold", color=NAVY)
    ax_r.grid(axis="y", alpha=0.2); ax_r.spines[["top","right"]].set_visible(False)
    for bar, val, a in zip(bars, mov_v, anch):
        if val > 0:
            lbl = f"{val:.0f}" + (" ★" if a else "")
            ax_r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      lbl, ha="center", va="bottom", fontsize=6.5,
                      fontweight="bold" if a else "normal")
    from matplotlib.patches import Patch
    legend_els = [Patch(fc=C["anchor_pt"], label="★ Anchor (stable)"),
                  Patch(fc=C["move_low"],  label="Low (<10px)"),
                  Patch(fc=C["move_med"],  label="Med (10-22px)"),
                  Patch(fc=C["move_high"], label="High (>22px)")]
    ax_r.legend(handles=legend_els, fontsize=7.5, loc="upper left", framealpha=0.92)

    # Step 7 ใช้ suptitle เหมือนกัน ไม่มี side panel (bar chart ใช้พื้นที่เต็ม)
    fig.suptitle(
        f"STEP 7  —  Final Results: Deviation Analysis + Clear Aligner Planning  |  "
        f"Image: {image_name}  |  Triangular Method (Li et al. 2017)",
        fontsize=10.5, fontweight="bold", color=NAVY,
        x=0.5, y=0.97, ha="center", va="top",
    )
    p = sd / f"step7_final_summary_{image_name}.png"
    plt.savefig(p, dpi=STEP_FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(p)

    return saved
