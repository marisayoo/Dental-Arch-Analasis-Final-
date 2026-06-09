# Dental Arch Analysis System  v3

ระบบวิเคราะห์การเรียงตัวของฟันล่างจากภาพถ่าย  
เพื่อวางแผนการจัดฟันใส (Clear Aligner)

---

## โครงสร้างโฟลเดอร์

```
Dental-Arch-Analasis-main/
│
├── run.py                  ← ▶ จุดเริ่มต้น — รันที่นี่เพื่อใช้งาน
├── gui_app.py              ← GUI (PyQt5)
│
├── config/
│   └── settings.py         ← ค่าคงที่และการตั้งค่าทั้งระบบ
│
├── models/                 ← Business Logic (ไม่มี I/O)
│   ├── geometry.py         ← canonical transform, B-spline, deviation
│   └── tooth_data.py       ← validate, inventory, dedup
│
├── controllers/            ← I/O
│   ├── data_loader.py      ← โหลดจาก CSV หรือ YOLO model
│   ├── exporter.py         ← console output + CSV export
│   └── csv_export.py       ← canonical keypoints + full deviation CSV
│
├── views/
│   └── step_figures.py     ← ภาพ step-by-step 7 ขั้น
│
├── data/
│   └── input/              ← ▶ วางภาพที่ต้องการวิเคราะห์ที่นี่
│       └── LO_0625.jpg     ← ภาพตัวอย่าง
│
├── output/                 ← ผลลัพธ์ (สร้างอัตโนมัติ)
│   └── cases/
│       └── [ชื่อภาพ]/
│           ├── steps/      ← ภาพ step1–7
│           └── exports/    ← CSV deviation + ideal arch
│
├── tooth_ver1.onnx         ← YOLO Segmentation model
└── keypoint_ver1.onnx      ← YOLO Pose model
```

---

## วิธีติดตั้ง (ทำครั้งเดียว)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install numpy scipy matplotlib onnxruntime opencv-python PyQt5
```

---

## วิธีใช้งาน

### 1. วางภาพฟัน

วางภาพ Lower Occlusal View ลงใน `data/input/`

### 2. รัน

| คำสั่ง | ผลลัพธ์ |
|--------|---------|
| `python run.py` | เลือกภาพแบบ interactive |
| `python run.py --image data/input/x.jpg` | ระบุภาพโดยตรง |
| `python run.py --batch` | วิเคราะห์ทุกภาพใน data/input/ |
| `python run.py --csv data/samples/t.csv` | CSV mode (ไม่ใช้ YOLO) |
| `python gui_app.py` | เปิด GUI |

### 3. ผลลัพธ์

```
output/cases/[ชื่อภาพ]/
├── steps/
│   ├── step1_raw_keypoints_*.png
│   ├── step2_canonical_frame_*.png
│   ├── step3_canonical_transformed_*.png
│   ├── step4_bspline_arch_*.png
│   ├── step5_deviation_descriptive_*.png
│   ├── step6_ideal_arch_movement_*.png
│   └── step7_final_summary_*.png
└── exports/
    ├── deviation_*.csv
    └── ideal_deviation_*.csv
```

---

## Pipeline (7 ขั้นตอน)

| Step | คำสั่ง | คำอธิบาย |
|------|--------|-----------|
| 1 | `load_from_model()` | YOLO detect keypoints |
| 2 | `compute_transform()` | คำนวณ Canonical Frame (Triangular Method) |
| 3 | `transform_all()` | แปลงพิกัดเข้า Canonical Frame |
| 4 | `fit_bspline()` | Fit B-spline arch curve (p=2) |
| 5 | `compute_deviations()` | วัด deviation ตั้งฉากจาก arch |
| 6 | `compute_ideal_arch()` | สร้าง Ideal Arch จาก anchor teeth |
| 7 | `generate_all_steps()` | สร้างภาพสรุปผล 7 ขั้น |

---

## อ้างอิง

| Paper | ใช้สำหรับ |
|-------|-----------|
| Fan et al. (2025) Comput. Aided Geom. Des. 119:102436 | B-spline, Canonical Frame, Deviation |
| Li, Gateno, Xia (2017) Int J Oral Maxillofac Surg 46(7) | Triangular Method 93.3% accuracy |
| Wellens (2007) Am J Orthod 131:160.e17 | COG fallback orientation |
| Andrews (1972) Am J Orthod 62(3) | Ideal arch concept |
| ISO 3950:2016 | FDI tooth numbering |
"# Dental-Arch-Analasis-Final-" 
