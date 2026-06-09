import sys
import traceback
import torch
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
)

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dental Arch Analysis")
        self.resize(1100, 800)

        self.stack = QStackedWidget()

        self.select_page = self.create_select_page()
        self.result_page = self.create_result_page()

        self.stack.addWidget(self.select_page)
        self.stack.addWidget(self.result_page)

        self.setCentralWidget(self.stack)

        self.selected_image = None

        self.images = []
        self.current_index = 0

    # -------------------------
    # Page 1
    # -------------------------
    def create_select_page(self):
        
        page = QWidget()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setAlignment(Qt.AlignCenter)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(700, 450)
        self.preview_label.setAlignment(Qt.AlignCenter)

        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #999;
                background: white;
            }
        """)

        layout.addWidget(
            self.preview_label,
            alignment=Qt.AlignCenter
        )

        self.path_label = QLabel("")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)

        layout.addWidget(
            self.path_label,
            alignment=Qt.AlignCenter
        )
        layout.addSpacing(30)
        btn_select = QPushButton("เลือกไฟล์ภาพ")
        btn_select.clicked.connect(self.select_file)

        layout.addWidget(
            btn_select,
            alignment=Qt.AlignCenter
        )

        btn_run = QPushButton("เริ่มวิเคราะห์")
        btn_run.clicked.connect(self.run_analysis)

        layout.addWidget(
            btn_run,
            alignment=Qt.AlignCenter
        )

        page.setLayout(layout)

        return page

    # -------------------------
    # Page 2
    # -------------------------
    def create_result_page(self):

        page = QWidget()

        layout = QVBoxLayout()

        back_btn = QPushButton("กลับ")
        back_btn.clicked.connect(
            lambda: self.stack.setCurrentWidget(
                self.select_page
            )
        )

        self.title_label = QLabel("")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.step_label = QLabel("")
        self.step_label.setAlignment(Qt.AlignCenter)

        btn_prev = QPushButton("◀ ก่อนหน้า")
        btn_next = QPushButton("ถัดไป ▶")

        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)

        nav = QHBoxLayout()
        nav.addWidget(btn_prev)
        nav.addWidget(self.step_label)
        nav.addWidget(btn_next)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label)
        layout.addLayout(nav)
        layout.addWidget(back_btn)
        layout.addStretch()

        page.setLayout(layout)

        return page

    # -------------------------
    # Select file
    # -------------------------
    def select_file(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not file_path:
            return

        ext = Path(file_path).suffix.lower()

        allowed = {".png", ".jpg", ".jpeg"}

        if ext not in allowed:
            self.path_label.setText(
                f"ไฟล์ไม่รองรับ ({ext})"
            )
            return

        self.selected_image = file_path

        self.path_label.setText(
            Path(file_path).name
        )

        pixmap = QPixmap(file_path)

        if pixmap.isNull():
            self.path_label.setText(
                "ไม่สามารถโหลดรูปได้"
            )
            return

        pixmap = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.preview_label.setPixmap(pixmap)

    # -------------------------
    # Run pipeline
    # -------------------------
    def run_analysis(self):

        if not self.selected_image:
            return

        from main import run

        run(
            image_path=self.selected_image,
            save_steps=True
        )

        self.load_result_images()
        self.stack.setCurrentWidget(self.result_page)

    # -------------------------
    # Show output images
    # -------------------------
    def load_result_images(self):

        case_name = Path(self.selected_image).stem

        output_dir = (
            Path("output")
            / "cases"
            / case_name
            / "steps"
        )

        if not output_dir.exists():
            return

        self.images = sorted(
            output_dir.glob("*.png")
        )

        self.current_index = 0

        self.show_image()

    def show_image(self):

        if not self.images:
            return

        img_path = self.images[self.current_index]

        self.title_label.setText(img_path.name)

        pixmap = QPixmap(str(img_path))

        pixmap = pixmap.scaled(
            1000,
            700,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)

        self.step_label.setText(
            f"{self.current_index + 1}/{len(self.images)}"
        )

    def next_image(self):

        if not self.images:
            return

        self.current_index = (
            self.current_index + 1
        ) % len(self.images)

        self.show_image()

    def prev_image(self):

        if not self.images:
            return

        self.current_index = (
            self.current_index - 1
        ) % len(self.images)

        self.show_image()

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())