import sys
import subprocess
import logging
import threading
import traceback
import csv
import json
import shutil
import tempfile
import time
from statistics import mean
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QEvent, QTimer, qInstallMessageHandler
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QSplitter,
    QSizePolicy,
    QFrame,
    QDialogButtonBox,
    QPlainTextEdit,
)
from ultralytics import YOLO

# Branding files are optional. The application still opens if they are missing.
PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
PROFILE_FILE = PROJECT_DIR / "profile.json"

# App output is saved here, and can be followed live from the header button.
# This is deliberately separate from the user's images and training results.
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "ocean_studio.log"


class TeeLogStream:
    """Keep existing console output while saving Python stdout/stderr to a file."""

    def __init__(self, original, logfile, lock):
        self.original = original
        self.logfile = logfile
        self.lock = lock
        self.encoding = getattr(original, "encoding", "utf-8") or "utf-8"

    def write(self, text):
        text = str(text)
        with self.lock:
            if self.original is not None:
                try:
                    self.original.write(text)
                except (OSError, ValueError):
                    pass
            try:
                self.logfile.write(text)
                self.logfile.flush()
            except (OSError, ValueError):
                pass
        return len(text)

    def flush(self):
        with self.lock:
            if self.original is not None:
                try:
                    self.original.flush()
                except (OSError, ValueError):
                    pass
            try:
                self.logfile.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return False

    def writable(self):
        return True


def initialize_logging():
    """Capture Python prints, tracebacks and standard Python logging messages."""
    global LOG_FILE
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logfile = LOG_FILE.open("a", encoding="utf-8", buffering=1)
    except OSError:
        # Fallback for a project folder that is read-only.
        LOG_FILE = Path.home() / "ocean_studio.log"
        logfile = LOG_FILE.open("a", encoding="utf-8", buffering=1)

    lock = threading.RLock()
    sys.stdout = TeeLogStream(sys.stdout, logfile, lock)
    sys.stderr = TeeLogStream(sys.stderr, logfile, lock)

    # Package loggers such as Ultralytics use Python's logging module.
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger("ultralytics").addHandler(handler)

    def qt_message_handler(msg_type, context, message):
        # Qt warnings sometimes bypass Python's sys.stderr.
        try:
            sys.stderr.write(f"[Qt {msg_type.name}] {message}\n")
        except Exception:
            pass

    qInstallMessageHandler(qt_message_handler)
    print("\n" + "=" * 68)
    print("Ocean Studio started. Live application log:", LOG_FILE)
    print("=" * 68, flush=True)




def load_profile():
    defaults = {"first_name": "Your first name", "last_name": "Your surname",
                "email": "", "fnum": "",
                "university": "Your university", "program": "Master's thesis",
                "project_title": "Underwater Vision AI"}
    if PROFILE_FILE.is_file():
        try:
            data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in defaults:
                    if isinstance(data.get(key), str):
                        defaults[key] = data[key].strip()
        except (OSError, ValueError):
            pass
    return defaults


def logo_pixmap(filename, width, height):
    file = ASSETS_DIR / filename
    if not file.is_file():
        return None
    pixmap = QPixmap(str(file))
    if pixmap.isNull():
        return None
    return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


def find_background_asset():
    """Return the first usable background image from the assets folder."""
    # Fixed background: no in-app selector. Named background.* takes priority.
    # Legacy background_custom.* remains supported so your existing photo
    # does not disappear when you upgrade from the previous version.
    candidates = [
        "background.png", "background.jpg", "background.jpeg", "background.webp",
        "background_custom.png", "background_custom.jpg", "background_custom.jpeg", "background_custom.webp",
    ]
    for name in candidates:
        file = ASSETS_DIR / name
        if file.is_file():
            return file
    return None


def build_blurred_background_pixmap(width, height):
    """Load a user photo, crop it to the window, blur it, and darken it slightly."""
    if width < 2 or height < 2:
        return None
    file = find_background_asset()
    if file is None:
        return None
    try:
        data = np.fromfile(file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if image is None:
        return None
    src_h, src_w = image.shape[:2]
    scale = max(width / max(1, src_w), height / max(1, src_h))
    resized_w = max(width, int(src_w * scale))
    resized_h = max(height, int(src_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    x0 = max(0, (resized_w - width) // 2)
    y0 = max(0, (resized_h - height) // 2)
    image = image[y0:y0 + height, x0:x0 + width]
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=6, sigmaY=6)
    dark_overlay = np.full_like(image, (10, 18, 28))
    image = cv2.addWeighted(image, 0.56, dark_overlay, 0.44, 0.0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def set_ocean_app_icon(app):
    """Choose icon.png or app_icon.ico. Windows taskbar may prefer .ico."""
    for name in ("app_icon.ico", "app_icon.png"):
        path = ASSETS_DIR / name
        if path.is_file():
            icon = QIcon(str(path))
            if not icon.isNull():
                app.setWindowIcon(icon)
                return



OCEAN_STYLESHEET = r"""
QMainWindow {
    background-color: #06101a;
}
QWidget#root {
    background: transparent;
    color: #e8f4ff;
    font-family: 'Segoe UI', 'Inter', 'Arial';
    font-size: 12px;
}
QLabel { color: #dfebf4; background: transparent; }
QLabel#brandIcon {
    border: 1px solid #1e7893; border-radius: 14px;
    background: #0a3847; color: #65e5e6; font-size: 32px;
}
QLabel#brandTitle { font-size: 24px; font-weight: 800; color: #f4fbff; letter-spacing: 2px; }
QLabel#subTitle { font-size: 10px; color: #819aaf; letter-spacing: 1px; }
QLabel#headerTag {
    background: rgba(15, 43, 43, 218); border: 1px solid #244f57; border-radius: 13px;
    color: #bfece2; padding: 8px 13px; font-weight: 700; font-size: 10px;
}
QLabel#sectionTitle { font-size: 21px; font-weight: 750; color: #f2f9ff; }
QLabel#fieldLabel, QLabel#imageCardTitle, QLabel#cardHeading {
    color: #b8cfdf; font-size: 10px; font-weight: 750; letter-spacing: 1px;
}
QLabel#hintText, QLabel#imageCardHint { color: #7992aa; font-size: 10px; }
QLabel#cardIndex {
    color: #66e1e6; font-weight: 800; background: #123a49;
    border-radius: 7px; padding: 4px 7px;
}
QLabel#valueText { color: #bce7f2; font-weight: 650; }
QLabel#modelName {
    color: #a2f6ef; font-size: 13px; font-weight: 700;
    background: #0b2433; border: 1px solid #265166;
    border-radius: 9px; padding: 10px 11px;
}
QFrame#controlCard, QFrame#imageCard, QFrame#actionBar, QFrame#footer {
    background: rgba(11, 23, 37, 228); border: 1px solid #293c4d; border-radius: 15px;
}
QFrame#imageCard { background: rgba(9, 19, 30, 232); }
QLabel#imageCanvas {
    background: #071422; color: #5d8297; border: 1px dashed #25475b;
    border-radius: 11px; font-size: 12px; font-weight: 650;
}
QLabel#statusText { color: #bad0df; font-size: 11px; }
QLabel#statusLight { color: #55e7bf; font-size: 15px; }
QPushButton {
    color: #dce8ef; background: rgba(20, 37, 52, 228);
    border: 1px solid #335066; border-radius: 9px;
    padding: 9px 13px; min-height: 19px; font-weight: 600;
}
QPushButton:hover { color: white; border-color: #3893a3; background: #1a3b51; }
QPushButton:pressed { background: #103346; }
QPushButton:disabled { color: #62798a; background: #132234; border-color: #243345; }
QPushButton#accentButton, QPushButton#detectButton {
    background: #0e939b; border: 1px solid #20b5bd;
    color: #041922; font-weight: 800;
}
QPushButton#accentButton:hover, QPushButton#detectButton:hover {
    background: #28cad0; border-color: #64f1ee;
}
QPushButton#detectButton { background: #65dbc5; border-color: #72edd8; }
QPushButton#detectButton:hover { background: #a0f0e1; }
QComboBox {
    color: #e4f3f7; background: #10283a; border: 1px solid #34576b;
    border-radius: 9px; padding: 9px 11px; min-height: 19px;
}
QComboBox:hover, QComboBox:focus { border-color: #38b3bf; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #102536; color: #e3f5f7; border: 1px solid #356074;
    selection-background-color: #116b7a; selection-color: white;
}
QSlider::groove:horizontal { height: 5px; border-radius: 2px; background: #254456; }
QSlider::sub-page:horizontal { background: #21c3c4; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #b2ffff; border: 2px solid #0f8a9b;
    width: 15px; margin: -7px 0; border-radius: 9px;
}
QTabWidget#researchTabs::pane {
    background: #0d1e30; border: 1px solid #24394a;
    border-radius: 12px;
}
QTabWidget#researchTabs QTabBar::tab {
    background: #0c1a2a; color: #839bb0;
    border: 1px solid #1c3144; border-bottom: none;
    padding: 10px 17px; margin-right: 4px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    font-weight: 650;
}
QTabWidget#researchTabs QTabBar::tab:selected {
    background: #143345; color: #7feee9; border-color: #2e6c7b;
}
QScrollArea#sidebarScroll, QWidget#sidebar, QWidget#researchPage {
    background: transparent; border: none;
}
QScrollBar:vertical {
    background: #0a1b2a; width: 7px; margin: 3px 0; border: none;
}
QScrollBar::handle:vertical {
    background: #2d5966; border-radius: 3px; min-height: 28px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QSplitter#imageSplitter::handle { background: #07111f; width: 9px; }
QToolTip { color: white; background: #143244; border: 1px solid #3c7f8c; }
QFrame#aboutInfoCard, QFrame#aboutLogoCard {
    background: #10273a; border: 1px solid #29485d; border-radius: 11px;
}
QLabel#aboutSectionLabel { color: #83bbce; font-size: 10px; font-weight: 750; }
QLabel#aboutValue { color: #f2fbff; font-size: 14px; font-weight: 650; }
QLabel#aboutFooter { color: #8eb1c5; font-size: 11px; }
QLabel#workspaceLogoText { color: #53cdd1; font-weight: 800; letter-spacing: 2px; }
QLabel#aboutName { font-size: 17px; color: #f4fbff; font-weight: 800; }
QLabel#aboutProject { color: #6de6dc; font-size: 11px; font-weight: 650; }
QLabel#aboutLogo { background: #091929; border: 1px solid #2a5264;
                   border-radius: 9px; color: #7794a6; font-size: 10px; }
"""

def enhance_image(image, clip_limit=2.0):
    """Improve local contrast using CLAHE on the LAB lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    improved_l = clahe.apply(l)
    merged = cv2.merge((improved_l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def white_balance_image(image, strength=1.0):
    """Gray-world white balance, blended with original using strength 0..1."""
    pixels = image.astype(np.float32)
    means = pixels.mean(axis=(0, 1))
    overall_mean = means.mean()
    scales = overall_mean / np.maximum(means, 1e-6)
    balanced = np.clip(pixels * scales.reshape(1, 1, 3), 0, 255)
    result = pixels * (1.0 - strength) + balanced * strength
    return np.clip(result, 0, 255).astype(np.uint8)


def gamma_correction_image(image, gamma=0.8):
    """Gamma < 1 brightens midtones; gamma > 1 darkens midtones."""
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.float32,
    )
    return cv2.LUT(image, np.clip(table, 0, 255).astype(np.uint8))


def apply_enhancement(image, method, value):
    """Start from the original image every time: no repeated enhancements."""
    if method == "Original (No Enhancement)":
        return image.copy()
    if method == "CLAHE":
        return enhance_image(image, clip_limit=value / 10.0)
    if method == "White Balance":
        return white_balance_image(image, strength=value / 100.0)
    if method == "Gamma Correction":
        return gamma_correction_image(image, gamma=value / 100.0)
    raise ValueError(f"Unknown enhancement method: {method}")


# Slider numbers are integers; decimal settings are converted in apply_enhancement.
METHOD_SETTINGS = {
    "CLAHE": (10, 60, 20),                # clip limit: 1.0 to 6.0; default 2.0
    "White Balance": (0, 100, 100),       # strength: 0% to 100%; default 100%
    "Gamma Correction": (30, 200, 80),   # gamma: 0.30 to 2.00; default 0.80
    "Original (No Enhancement)": (0, 0, 0),
}


COMPARISON_METHODS = (
    "Original (No Enhancement)",
    "CLAHE",
    "White Balance",
    "Gamma Correction",
)


def make_comparison_montage(images, descriptions, cell_width=500, cell_height=330):
    """Make a 2-by-2 PNG-friendly montage at equal sizes without distortion."""
    header_height = 45
    margin = 16
    canvas = np.full(
        (2 * (cell_height + header_height + margin) + margin,
         2 * (cell_width + margin) + margin, 3),
        245, dtype=np.uint8,
    )
    for index, (image, title) in enumerate(zip(images, descriptions)):
        row, col = divmod(index, 2)
        x = margin + col * (cell_width + margin)
        y = margin + row * (cell_height + header_height + margin)
        cv2.putText(canvas, title, (x + 5, y + 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (35, 50, 65), 2, cv2.LINE_AA)
        height, width = image.shape[:2]
        scale = min(cell_width / width, cell_height / height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA
                             if scale < 1 else cv2.INTER_LINEAR)
        y0 = y + header_height + (cell_height - new_size[1]) // 2
        x0 = x + (cell_width - new_size[0]) // 2
        canvas[y0:y0 + new_size[1], x0:x0 + new_size[0]] = resized
    return canvas


class ComparisonDialog(QDialog):
    """Display all four algorithms on the same source image."""

    def __init__(self, original, method_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enhancement Comparison - Same Input Image")
        self.resize(1120, 850)

        self.images = []
        self.descriptions = []
        for method in COMPARISON_METHODS:
            value = method_values[method]
            self.images.append(apply_enhancement(original, method, value))
            if method == "CLAHE":
                title = f"CLAHE (clip limit {value / 10.0:.1f})"
            elif method == "White Balance":
                title = f"White Balance ({value}% strength)"
            elif method == "Gamma Correction":
                title = f"Gamma Correction (gamma {value / 100.0:.2f})"
            else:
                title = "Original (No Enhancement)"
            self.descriptions.append(title)

        layout = QVBoxLayout(self)
        note = QLabel("Same original image in all four panels. Current slider values are used.")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)
        grid = QGridLayout()
        for index, (image, title) in enumerate(zip(self.images, self.descriptions)):
            panel = QVBoxLayout()
            heading = QLabel(title)
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel.addWidget(heading)
            picture = QLabel()
            picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
            picture.setMinimumSize(300, 230)
            picture.setStyleSheet("background: #172b3a; border: 1px solid #456;")
            self.show_preview(picture, image)
            panel.addWidget(picture)
            grid.addLayout(panel, index // 2, index % 2)
        layout.addLayout(grid, 1)

        buttons = QHBoxLayout()
        save_button = QPushButton("Save Comparison PNG")
        save_button.clicked.connect(self.save_comparison)
        buttons.addWidget(save_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    @staticmethod
    def show_preview(label, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, w * channels,
                        QImage.Format.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimage).scaled(
            480, 310, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def save_comparison(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison", "underwater_comparison.png", "PNG Image (*.png)"
        )
        if not path:
            return
        try:
            montage = make_comparison_montage(self.images, self.descriptions)
            success, encoded = cv2.imencode(".png", montage)
            if not success:
                raise ValueError("OpenCV could not encode the comparison image.")
            encoded.tofile(path)
            QMessageBox.information(self, "Saved", "Comparison image saved successfully.")
        except Exception as error:
            QMessageBox.critical(self, "Save Error", str(error))


class DetectionComparisonDialog(QDialog):
    """Run one YOLO model on four versions of exactly the same input image."""

    def __init__(self, original, method_values, model, confidence, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Detection Comparison - Four Image Methods")
        self.resize(1130, 870)
        self.original = original
        self.method_values = method_values.copy()
        self.model = model
        self.confidence = confidence
        self.records = []
        self.images = []
        self.descriptions = []

        layout = QVBoxLayout(self)
        information = QLabel(
            f"Same source image, selected YOLO model and confidence ({confidence:.2f}) for all methods. "
            "Counts are NOT accuracy scores."
        )
        information.setWordWrap(True)
        layout.addWidget(information)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Image method", "Detections", "Avg. confidence", "Time (ms)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(205)
        layout.addWidget(self.table)

        self.message = QLabel("Press Run Comparison to start four YOLO predictions.")
        layout.addWidget(self.message)

        self.grid = QGridLayout()
        self.preview_labels = []
        for index, method in enumerate(COMPARISON_METHODS):
            panel = QVBoxLayout()
            title = QLabel(method)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel.addWidget(title)
            preview = QLabel("Not analyzed yet")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(280, 190)
            preview.setStyleSheet(
                "background: #172b3a; color: white; border: 1px solid #456;"
            )
            panel.addWidget(preview)
            self.preview_labels.append(preview)
            self.grid.addLayout(panel, index // 2, index % 2)
        layout.addLayout(self.grid, 1)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run Comparison")
        self.run_button.clicked.connect(self.run_comparison)
        buttons.addWidget(self.run_button)
        self.csv_button = QPushButton("Export Results CSV")
        self.csv_button.setEnabled(False)
        self.csv_button.clicked.connect(self.save_csv)
        buttons.addWidget(self.csv_button)
        self.png_button = QPushButton("Save Detection Comparison PNG")
        self.png_button.setEnabled(False)
        self.png_button.clicked.connect(self.save_png)
        buttons.addWidget(self.png_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        footnote = QLabel(
            "Timing is wall-clock time per prediction and varies by hardware and model "
            "warm-up. True accuracy requires labeled ground-truth images (a future step)."
        )
        footnote.setWordWrap(True)
        layout.addWidget(footnote)

    def run_comparison(self):
        self.run_button.setEnabled(False)
        self.csv_button.setEnabled(False)
        self.png_button.setEnabled(False)
        self.table.setRowCount(0)
        self.records = []
        self.images = []
        self.descriptions = []
        try:
            for index, method in enumerate(COMPARISON_METHODS):
                self.message.setText(f"Analyzing {index + 1}/4: {method}...")
                QApplication.processEvents()
                value = self.method_values[method]
                image = apply_enhancement(self.original, method, value)
                start = time.perf_counter()
                predictions = self.model.predict(
                    source=image, conf=self.confidence, verbose=False
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                prediction = predictions[0]
                count = len(prediction.boxes)
                mean_confidence = (
                    float(prediction.boxes.conf.mean().item()) if count else None
                )
                annotated = prediction.plot()
                self.images.append(annotated)
                self.descriptions.append(method)
                record = {
                    "method": method,
                    "setting": value,
                    "confidence_threshold": self.confidence,
                    "detections": count,
                    "mean_detection_confidence": mean_confidence,
                    "prediction_time_ms": round(elapsed_ms, 2),
                }
                self.records.append(record)
                self.table.insertRow(index)
                columns = [
                    method,
                    str(count),
                    f"{mean_confidence:.3f}" if mean_confidence is not None else "N/A",
                    f"{elapsed_ms:.1f}",
                ]
                for col, text in enumerate(columns):
                    self.table.setItem(index, col, QTableWidgetItem(text))
                ComparisonDialog.show_preview(self.preview_labels[index], annotated)
            self.message.setText(
                "Comparison complete! Export CSV for your research notes, "
                "or save the four detection images."
            )
            self.csv_button.setEnabled(True)
            self.png_button.setEnabled(True)
        except Exception as error:
            self.message.setText("Comparison interrupted by an error.")
            QMessageBox.critical(self, "Detection Comparison Error", str(error))
        finally:
            self.run_button.setEnabled(True)

    def save_csv(self):
        if len(self.records) != len(COMPARISON_METHODS):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Detection Results", "yolo_comparison.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
            QMessageBox.information(self, "Saved", "Results CSV saved successfully.")
        except Exception as error:
            QMessageBox.critical(self, "CSV Save Error", str(error))

    def save_png(self):
        if len(self.images) != len(COMPARISON_METHODS):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Detection Comparison", "yolo_detection_comparison.png",
            "PNG Image (*.png)"
        )
        if not path:
            return
        try:
            montage = make_comparison_montage(self.images, self.descriptions)
            success, encoded = cv2.imencode(".png", montage)
            if not success:
                raise ValueError("Could not encode the comparison PNG.")
            encoded.tofile(path)
            QMessageBox.information(self, "Saved", "Detection comparison saved.")
        except Exception as error:
            QMessageBox.critical(self, "PNG Save Error", str(error))


# Step 7: Batch comparison (counts and times, not ground-truth accuracy).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BATCH_FIELDS = [
    "filename", "method", "setting", "confidence_threshold", "model",
    "detections", "mean_detection_confidence", "prediction_time_ms", "status",
    "error",
]


class BatchWorker(QThread):
    """Run intensive model inference off the Qt GUI thread.

    Each image gets the same four methods, confidence setting, and YOLO weights.
    Stop is cooperative, checked between methods, not mid-inference.
    """

    progress = Signal(int, int, str)
    row_ready = Signal(dict)
    batch_finished = Signal(int, int, bool)
    batch_failed = Signal(str)

    def __init__(self, paths, method_values, confidence, weights, parent=None):
        super().__init__(parent)
        self.paths = list(paths)
        self.method_values = method_values.copy()
        self.confidence = confidence
        self.weights = str(weights)
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True

    def run(self):
        completed = 0
        failures = 0
        try:
            # Loading inside this thread avoids sharing a model across threads.
            model = YOLO(self.weights)
            total = len(self.paths) * len(COMPARISON_METHODS)
            for path in self.paths:
                if self.cancel_requested:
                    break
                try:
                    raw = np.fromfile(str(path), dtype=np.uint8)
                    original = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if original is None:
                        raise ValueError("File could not be decoded as an image")
                except Exception as exc:
                    # Record each skipped method so failures are visible in the CSV.
                    for method in COMPARISON_METHODS:
                        if self.cancel_requested:
                            break
                        self.row_ready.emit({
                            "filename": path.name, "method": method,
                            "setting": self.method_values[method],
                            "confidence_threshold": self.confidence,
                            "model": self.weights, "detections": "",
                            "mean_detection_confidence": "",
                            "prediction_time_ms": "", "status": "error",
                            "error": str(exc),
                        })
                        completed += 1
                        failures += 1
                        self.progress.emit(completed, total, path.name)
                    continue
                for method in COMPARISON_METHODS:
                    if self.cancel_requested:
                        break
                    value = self.method_values[method]
                    record = {
                        "filename": path.name,
                        "method": method,
                        "setting": value,
                        "confidence_threshold": self.confidence,
                        "model": self.weights,
                        "detections": "",
                        "mean_detection_confidence": "",
                        "prediction_time_ms": "",
                        "status": "ok",
                        "error": "",
                    }
                    try:
                        enhanced = apply_enhancement(original, method, value)
                        start = time.perf_counter()
                        prediction = model.predict(
                            source=enhanced, conf=self.confidence, verbose=False
                        )[0]
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        count = len(prediction.boxes)
                        mean_conf = (float(prediction.boxes.conf.mean().item())
                                     if count else None)
                        record["detections"] = count
                        record["mean_detection_confidence"] = (
                            round(mean_conf, 5) if mean_conf is not None else ""
                        )
                        record["prediction_time_ms"] = round(elapsed_ms, 2)
                    except Exception as exc:
                        record["status"] = "error"
                        record["error"] = str(exc)
                        failures += 1
                    self.row_ready.emit(record)
                    completed += 1
                    self.progress.emit(completed, total, path.name)
            self.batch_finished.emit(completed, failures, self.cancel_requested)
        except Exception as exc:
            self.batch_failed.emit(str(exc))


# Step 8: summarize batch data. The statistics below are descriptive only.
# They are NOT precision, recall, mAP, or ground-truth detection accuracy.
SUMMARY_FIELDS = [
    "method", "successful_images", "failed_rows", "mean_detections",
    "mean_confidence_nonempty", "mean_prediction_time_ms",
]


def summarize_batch(records):
    """Build one aggregate row for each method, ignoring failed predictions."""
    summaries = []
    for method in COMPARISON_METHODS:
        subset = [r for r in records if r.get("method") == method]
        valid = [r for r in subset if r.get("status") == "ok"]
        counts = [float(r["detections"]) for r in valid
                  if r.get("detections") not in ("", None)]
        times = [float(r["prediction_time_ms"]) for r in valid
                 if r.get("prediction_time_ms") not in ("", None)]
        # Only predictions containing >=1 detection have a mean confidence.
        confidences = [float(r["mean_detection_confidence"]) for r in valid
                       if r.get("mean_detection_confidence") not in ("", None)]
        summaries.append({
            "method": method,
            "successful_images": len(valid),
            "failed_rows": len(subset) - len(valid),
            "mean_detections": round(mean(counts), 3) if counts else "",
            "mean_confidence_nonempty": round(mean(confidences), 4) if confidences else "",
            "mean_prediction_time_ms": round(mean(times), 2) if times else "",
        })
    return summaries


class SimpleBarChart(QWidget):
    """Draw a readable chart using Qt only (no additional dependencies)."""

    def __init__(self, title, summaries, field, unit="", parent=None):
        super().__init__(parent)
        self.title = title
        self.summaries = summaries
        self.field = field
        self.unit = unit
        self.setMinimumSize(700, 245)

    def draw_chart(self, painter, width):
        """Draw onto an existing painter, whether on screen or into a PNG."""
        painter.fillRect(0, 0, width, self.minimumHeight(), QColor("#ffffff"))
        painter.setPen(QColor("#1b3342"))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(22, 28, self.title)
        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        values = [float(s[self.field]) for s in self.summaries
                  if s[self.field] not in ("", None)]
        maximum = max(values, default=0.0)
        plot_left = 193
        usable_width = max(10, width - 100 - plot_left)
        colors = ["#176b87", "#158e85", "#db9b39", "#7860af"]
        for index, item in enumerate(self.summaries):
            y = 62 + index * 42
            painter.setPen(QColor("#1b3342"))
            painter.drawText(12, y + 17, item["method"])
            value = item[self.field]
            if value in ("", None):
                painter.drawText(plot_left + 4, y + 17, "N/A")
                continue
            number = float(value)
            bar_width = max(0, round((number / maximum) * usable_width)) if maximum else 0
            painter.fillRect(plot_left, y, bar_width, 23, QColor(colors[index]))
            painter.setPen(QColor("#1b3342"))
            painter.drawText(plot_left + bar_width + 8, y + 17,
                             f"{number:.3f}{self.unit}")

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.save()
            try:
                self.draw_chart(painter, self.width())
            finally:
                painter.restore()
        finally:
            painter.end()


class BatchReportDialog(QDialog):
    """Plot aggregates of in-memory batch rows or of a previously exported CSV."""

    def __init__(self, records, source_label="Batch results", parent=None):
        super().__init__(parent)
        self.records = list(records)
        self.summaries = summarize_batch(self.records)
        self.setWindowTitle("Step 8 - Batch Charts and Summary")
        self.resize(900, 820)
        outer = QVBoxLayout(self)
        heading = QLabel(f"Batch report: {source_label}")
        heading.setWordWrap(True)
        outer.addWidget(heading)
        note = QLabel(
            "These charts describe detections and timing, NOT true accuracy. "
            "Mean confidence is the average of per-image confidence values only "
            "for images containing detections; zero-detection images are excluded "
            "from that confidence average. Compare methods with similar successful "
            "image counts. Time includes prediction-call overhead."
        )
        note.setWordWrap(True)
        outer.addWidget(note)
        self.table = QTableWidget(len(self.summaries), 6)
        self.table.setHorizontalHeaderLabels([
            "Method", "Success", "Errors", "Avg boxes", "Avg confidence*", "Avg ms"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, record in enumerate(self.summaries):
            for col, field in enumerate(SUMMARY_FIELDS):
                value = record[field]
                self.table.setItem(row, col, QTableWidgetItem(
                    "N/A" if value == "" else str(value)
                ))
        self.table.setMaximumHeight(180)
        outer.addWidget(self.table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        chart_layout = QVBoxLayout(container)
        self.charts = [
            SimpleBarChart("Average detections per successfully processed image",
                           self.summaries, "mean_detections"),
            SimpleBarChart("Average YOLO prediction time (milliseconds)",
                           self.summaries, "mean_prediction_time_ms", " ms"),
            SimpleBarChart("Average confidence for images with detections",
                           self.summaries, "mean_confidence_nonempty"),
        ]
        for chart in self.charts:
            chart_layout.addWidget(chart)
        scroll.setWidget(container)
        outer.addWidget(scroll, 1)
        controls = QHBoxLayout()
        self.summary_button = QPushButton("Export Summary CSV")
        self.summary_button.clicked.connect(self.export_summary)
        controls.addWidget(self.summary_button)
        self.charts_button = QPushButton("Save Charts PNG")
        self.charts_button.clicked.connect(self.save_charts)
        controls.addWidget(self.charts_button)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        outer.addLayout(controls)

    def export_summary(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Summary", "underwater_batch_summary.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerows(self.summaries)
            QMessageBox.information(self, "Saved", "Summary CSV saved successfully.")
        except Exception as error:
            QMessageBox.critical(self, "Export Error", str(error))

    def save_charts(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Charts", "underwater_batch_charts.png", "PNG Images (*.png)"
        )
        if not path:
            return
        try:
            width = 820
            height = sum(chart.minimumHeight() for chart in self.charts) + 20
            image = QImage(width, height, QImage.Format.Format_RGB32)
            if image.isNull():
                raise MemoryError("Unable to allocate chart image")
            image.fill(QColor("white"))
            painter = QPainter(image)
            if not painter.isActive():
                raise RuntimeError("Could not start painting the chart image")
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                offset = 10
                for chart in self.charts:
                    painter.save()
                    try:
                        painter.translate(0, offset)
                        chart.draw_chart(painter, width)
                    finally:
                        painter.restore()
                    offset += chart.minimumHeight()
            finally:
                # Must finish painting BEFORE saving or destroying the QImage.
                painter.end()
            if not image.save(path, "PNG"):
                raise IOError("Qt could not save the PNG file")
            QMessageBox.information(self, "Saved", "Charts PNG saved successfully.")
        except Exception as error:
            QMessageBox.critical(self, "Image Save Error", str(error))


def __qt_point(x, y):
    from PySide6.QtCore import QPoint
    return QPoint(x, y)


class BatchComparisonDialog(QDialog):
    """Select a directory and collect one CSV record per image per method."""

    def __init__(self, method_values, confidence, weights, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch YOLO Comparison - Folder of Images")
        self.resize(1080, 650)
        self.method_values = method_values.copy()
        self.confidence = confidence
        self.weights = str(weights)
        self.paths = []
        self.records = []
        self.worker = None
        self.running = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Compare Original, CLAHE, White Balance and Gamma Correction on "
            "EVERY image in a folder. This measures detection counts and timing, "
            "NOT true detection accuracy."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        folder_row = QHBoxLayout()
        self.folder_label = QLabel("No image folder selected")
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_label, 1)
        self.folder_button = QPushButton("Choose Image Folder")
        self.folder_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.folder_button)
        layout.addLayout(folder_row)
        self.settings_label = QLabel(
            f"Model: {Path(self.weights).name}  |  Confidence: {confidence:.2f}  |  "
            "Enhancement settings are captured from the main window."
        )
        self.settings_label.setWordWrap(True)
        layout.addWidget(self.settings_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.message = QLabel("Choose a folder of images, then press Run Batch.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Image", "Method", "Detections", "Mean confidence", "Time (ms)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        controls = QHBoxLayout()
        self.run_button = QPushButton("Run Batch")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_batch)
        controls.addWidget(self.run_button)
        self.cancel_button = QPushButton("Cancel Batch")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)
        controls.addWidget(self.cancel_button)
        self.export_button = QPushButton("Export Batch CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        controls.addWidget(self.export_button)
        self.report_button = QPushButton("Charts & Summary")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.open_report)
        controls.addWidget(self.report_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        controls.addWidget(self.close_button)
        layout.addLayout(controls)
        note = QLabel(
            "Tip: Choose a folder with 3-5 images for your first test. "
            "Only files directly inside the folder are processed. "
            "Inference time includes model execution overhead, can vary between runs, "
            "and the CSV does not provide precision, recall or mAP."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a Folder of Images")
        if not folder:
            return
        root = Path(folder)
        self.paths = sorted(
            (path for path in root.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name.lower(),
        )
        self.folder_label.setText(f"{root}  |  {len(self.paths)} images found")
        self.run_button.setEnabled(bool(self.paths))
        self.progress_bar.setValue(0)
        self.records = []
        self.table.setRowCount(0)
        self.export_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.message.setText(
            f"Found {len(self.paths)} images. "
            f"Run Batch will make {len(self.paths) * len(COMPARISON_METHODS)} predictions."
            if self.paths else "No supported images found in that folder."
        )

    def run_batch(self):
        if not self.paths or self.running or (self.worker is not None and self.worker.isRunning()):
            return
        self.records = []
        self.table.setRowCount(0)
        self.progress_bar.setRange(0, len(self.paths) * len(COMPARISON_METHODS))
        self.progress_bar.setValue(0)
        self.folder_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.running = True
        self.message.setText("Loading YOLO and starting batch analysis...")
        self.worker = BatchWorker(self.paths, self.method_values, self.confidence, self.weights, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.row_ready.connect(self.on_row)
        self.worker.batch_finished.connect(self.on_finished)
        self.worker.batch_failed.connect(self.on_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def on_row(self, record):
        self.records.append(record)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((
            record["filename"], record["method"], record["detections"],
            record["mean_detection_confidence"], record["prediction_time_ms"],
        )):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))

    def on_progress(self, completed, total, filename):
        self.progress_bar.setValue(completed)
        self.message.setText(f"Processed {completed}/{total} methods. Current image: {filename}")

    def finish_ui(self):
        self.running = False
        self.folder_button.setEnabled(True)
        self.run_button.setEnabled(bool(self.paths))
        self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(bool(self.records))
        self.report_button.setEnabled(bool(self.records))

    def on_finished(self, completed, failures, cancelled):
        self.finish_ui()
        status = "Cancelled" if cancelled else "Completed"
        self.message.setText(
            f"{status}: {completed} image-method rows; {failures} failed rows. "
            "Click Export Batch CSV to save the data."
        )

    def on_failed(self, error):
        self.finish_ui()
        self.message.setText("Batch stopped because of an error.")
        QMessageBox.critical(self, "Batch Error", error)

    def open_report(self):
        if not self.records or self.running:
            return
        dialog = BatchReportDialog(self.records, "Current batch", self)
        dialog.exec()

    def cancel_batch(self):
        if self.worker is not None and self.running:
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.message.setText("Stopping after the current prediction finishes...")

    def export_csv(self):
        if not self.records:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch Results", "underwater_batch_results.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=BATCH_FIELDS)
                writer.writeheader()
                writer.writerows(self.records)
            QMessageBox.information(
                self, "Saved", f"Exported {len(self.records)} rows to CSV."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def closeEvent(self, event):
        if self.running or (self.worker is not None and self.worker.isRunning()):
            QMessageBox.information(
                self, "Batch in Progress",
                "Please click Cancel Batch and wait for it to stop before closing."
            )
            event.ignore()
            return
        event.accept()




class GroundTruthEvaluationWorker(QThread):
    """Validate a model against real ground-truth boxes outside the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset_yaml, weights, split, parent=None):
        super().__init__(parent)
        self.dataset_yaml = dataset_yaml
        self.weights = weights
        self.split = split

    def run(self):
        try:
            # Included with Ultralytics. Require class index to mean the SAME
            # object in the dataset as in the trained model.
            import yaml
            with open(self.dataset_yaml, "r", encoding="utf-8") as fh:
                dataset = yaml.safe_load(fh)
            if not isinstance(dataset, dict):
                raise ValueError("Dataset YAML must contain a mapping (key: value).")
            if self.split not in dataset:
                raise ValueError(
                    f"The dataset YAML has no '{self.split}:' entry. "
                    "Choose another split or add its image path."
                )
            names = dataset.get("names")
            if isinstance(names, list):
                dataset_names = {i: str(name) for i, name in enumerate(names)}
            elif isinstance(names, dict):
                dataset_names = {int(i): str(name) for i, name in names.items()}
            else:
                raise ValueError("Dataset YAML must specify 'names' as a list or mapping.")
            if not dataset_names:
                raise ValueError("Dataset YAML contains no class names.")

            model = YOLO(self.weights)
            model_names_raw = model.names
            model_names = (
                {int(i): str(n) for i, n in model_names_raw.items()}
                if isinstance(model_names_raw, dict)
                else {i: str(n) for i, n in enumerate(model_names_raw)}
            )
            mismatch = (
                set(model_names) != set(dataset_names)
                or any(model_names[k].strip().casefold() != dataset_names[k].strip().casefold()
                       for k in dataset_names)
            )
            if mismatch:
                raise ValueError(
                    "Dataset class IDs/names do not match the model's classes!\n\n"
                    "That would make the accuracy scores misleading.\n"
                    "For an underwater-specific dataset, first train/fine-tune a "
                    "YOLO model on those same classes and select its best.pt file.\n\n"
                    f"Dataset classes: {dataset_names}\n"
                    f"Model classes: {model_names}"
                )

            # A low confidence threshold lets validation build PR curves across
            # many thresholds; workers=0 avoids Windows multiprocessing issues.
            # This tests ORIGINAL images only. We will add a controlled
            # enhancement comparison in a later research step.
            metrics = model.val(
                data=self.dataset_yaml,
                split=self.split,
                imgsz=640,
                batch=4,
                conf=0.001,
                iou=0.7,
                workers=0,
                plots=False,
                verbose=False,
                project=str(Path(self.dataset_yaml).parent / "evaluation_runs"),
                name="step9_baseline",
                exist_ok=True,
            )
            box = metrics.box
            result = {
                "dataset_yaml": self.dataset_yaml,
                "weights": self.weights,
                "split": self.split,
                "class_names": dataset_names,
                "precision": float(box.mp),
                "recall": float(box.mr),
                "map50": float(box.map50),
                "map5095": float(box.map),
                "imagesize": 640,
                "confidence_for_validation": 0.001,
                "iou_nms": 0.7,
            }
            # Guard against accidental invalid metrics.
            if not all(np.isfinite(result[key]) for key in
                       ("precision", "recall", "map50", "map5095")):
                raise ValueError("Validation returned non-finite scores; check the labels.")
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class GroundTruthEvaluationDialog(QDialog):
    """Step 9: labeled dataset evaluation of a class-matched YOLO checkpoint."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Step 9 — Ground-Truth YOLO Evaluation")
        self.resize(820, 550)
        self.worker = None
        self.result = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Real accuracy needs a labeled YOLO dataset (images + correct "
            "bounding-box labels + data.yaml). We evaluate ORIGINAL images "
            "first as a baseline. Dataset classes must exactly match the "
            "model's trained classes."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        yaml_row = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        self.yaml_edit.setPlaceholderText("Path to your dataset's data.yaml")
        yaml_row.addWidget(self.yaml_edit, 1)
        select_yaml = QPushButton("Choose data.yaml")
        select_yaml.clicked.connect(self.choose_yaml)
        yaml_row.addWidget(select_yaml)
        layout.addLayout(yaml_row)

        model_row = QHBoxLayout()
        self.weights_edit = QLineEdit("yolo26n.pt")
        self.weights_edit.setToolTip(
            "Use best.pt from training on your underwater classes, "
            "or yolo26n.pt only for a dataset with matching COCO classes."
        )
        model_row.addWidget(QLabel("YOLO model:"))
        model_row.addWidget(self.weights_edit, 1)
        select_weights = QPushButton("Choose .pt")
        select_weights.clicked.connect(self.choose_weights)
        model_row.addWidget(select_weights)
        layout.addLayout(model_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Dataset split:"))
        self.split_choice = QComboBox()
        self.split_choice.addItems(["val", "test"])
        split_row.addWidget(self.split_choice)
        split_row.addStretch()
        layout.addLayout(split_row)

        warning = QLabel(
            "Important: This is NOT accuracy of CLAHE/White Balance/Gamma. "
            "It tests the original dataset images. The Step 8 counts and "
            "confidence graphs are not accuracy metrics."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Evaluate Labeled Dataset")
        self.run_button.clicked.connect(self.start_evaluation)
        actions.addWidget(self.run_button)
        self.export_button = QPushButton("Export Accuracy CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)

        self.status_label = QLabel("Ready. Select a data.yaml and compatible model.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget(4, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Measured value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i, name in enumerate(("Precision (mean)", "Recall (mean)",
                                  "mAP@50", "mAP@50–95")):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem("Not measured"))
        layout.addWidget(self.table)

    def choose_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a YOLO dataset YAML", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.yaml_edit.setText(path)

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model weights", "", "PyTorch weights (*.pt)"
        )
        if path:
            self.weights_edit.setText(path)

    def start_evaluation(self):
        if self.worker is not None and self.worker.isRunning():
            return
        dataset_yaml = self.yaml_edit.text().strip()
        weights = self.weights_edit.text().strip()
        if not Path(dataset_yaml).is_file():
            QMessageBox.warning(self, "Dataset missing", "Choose an existing data.yaml file.")
            return
        if not weights:
            QMessageBox.warning(self, "Model missing", "Choose model weights (.pt).")
            return
        self.result = None
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status_label.setText("Evaluating... this can take a while on CPU.")
        self.worker = GroundTruthEvaluationWorker(
            dataset_yaml, weights, self.split_choice.currentText(), self
        )
        self.worker.completed.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def show_result(self, result):
        self.result = result
        for i, key in enumerate(("precision", "recall", "map50", "map5095")):
            self.table.setItem(i, 1, QTableWidgetItem(f"{result[key]:.4f}"))
        self.status_label.setText(
            "Evaluation completed on original images. Metrics are fractions "
            "from 0 to 1. Export them with the model and dataset information."
        )
        self.export_button.setEnabled(True)
        self.view_charts_button.setEnabled(True)

    def open_charts(self):
        if self.results:
            AccuracyReportDialog(self.results, 'Current evaluation', self).exec()

    def show_error(self, message):
        self.status_label.setText("Evaluation failed. See the explanation.")
        QMessageBox.critical(self, "Dataset Evaluation Error", message)

    def on_worker_finished(self):
        self.run_button.setEnabled(True)
        # QThread object is parent-owned; clean up later when the dialog closes.

    def export_csv(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export labeled evaluation", "step9_accuracy.csv", "CSV (*.csv)"
        )
        if not path:
            return
        fields = ("dataset_yaml", "weights", "split", "precision", "recall",
                  "map50", "map5095", "imagesize",
                  "confidence_for_validation", "iou_nms")
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerow({key: self.result[key] for key in fields})
            self.status_label.setText(f"Accuracy CSV exported to: {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def closeEvent(self, event):
        # Do not destroy a running worker while it is still evaluating.
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, "Evaluation running", "Finish evaluation before closing this window."
            )
            event.ignore()
            return
        super().closeEvent(event)




# ============================================================================
# Step 10: inspect a REAL labeled underwater dataset BEFORE training.
# This checks files and annotations, not model accuracy.
# ============================================================================

DATASET_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_class_names(raw):
    if isinstance(raw, list):
        names = {index: str(name) for index, name in enumerate(raw)}
    elif isinstance(raw, dict):
        names = {int(index): str(name) for index, name in raw.items()}
    else:
        raise ValueError("data.yaml needs 'names:' as a list or dictionary.")
    if not names or set(names) != set(range(len(names))):
        raise ValueError("Class IDs must be consecutive: 0, 1, 2, ...")
    return names


def locate_dataset_split(yaml_path, dataset, split):
    """Try common Ultralytics and Roboflow relative-path conventions."""
    entry = dataset.get(split)
    if entry is None:
        return None
    if not isinstance(entry, str):
        raise ValueError(
            f"'{split}:' should point to an image folder for this beginner checker. "
            "Image lists and multiple folders are not supported here."
        )
    yaml_folder = Path(yaml_path).resolve().parent
    configured_root = Path(str(dataset.get('path', '.')))
    root = configured_root if configured_root.is_absolute() else yaml_folder / configured_root
    relative = Path(entry)
    candidates = []
    if relative.is_absolute():
        candidates.append(relative)
    else:
        candidates += [root / relative, yaml_folder / relative]
        # Some exports use '../train/images' despite storing train/ beside YAML.
        stripped = list(relative.parts)
        while stripped and stripped[0] in ('.', '..'):
            stripped.pop(0)
        if stripped:
            candidates += [yaml_folder.joinpath(*stripped), root.joinpath(*stripped)]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    checked = '\n'.join(f'  - {c.resolve()}' for c in candidates)
    raise FileNotFoundError(
        f"The '{split}' image folder was not found. data.yaml says: {entry}\n"
        f"Places checked:\n{checked}\n\n"
        "Extract the COMPLETE dataset ZIP and keep images/, labels/, and data.yaml together."
    )


def annotation_path_for(image, image_folder):
    """Supports images/train and train/images YOLO directory layouts."""
    parts = list(image.parts)
    image_positions = [i for i, segment in enumerate(parts) if segment.lower() == 'images']
    if not image_positions:
        raise ValueError(
            f"Cannot locate 'images' in {image_folder}. Expected train/images "
            "or images/train."
        )
    parts[image_positions[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


class DatasetAuditWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, yaml_path, parent=None):
        super().__init__(parent)
        self.yaml_path = yaml_path

    def run(self):
        try:
            import yaml
            with open(self.yaml_path, 'r', encoding='utf-8-sig') as fh:
                dataset = yaml.safe_load(fh)
            if not isinstance(dataset, dict):
                raise ValueError('data.yaml is not a valid key/value mapping.')
            names = dataset_class_names(dataset.get('names'))
            if 'nc' in dataset and int(dataset['nc']) != len(names):
                raise ValueError("The 'nc' count disagrees with 'names' in data.yaml.")

            summary = []
            errors = []
            per_class = {name: 0 for name in names}
            total_images = 0
            total_boxes = 0
            total_issues = 0
            for split in ('train', 'val', 'test'):
                folder = locate_dataset_split(self.yaml_path, dataset, split)
                if folder is None:
                    continue
                images = sorted(
                    item for item in folder.rglob('*')
                    if item.is_file() and item.suffix.lower() in DATASET_IMAGE_SUFFIXES
                )
                split_images = len(images)
                split_labels = 0
                split_backgrounds = 0
                split_boxes = 0
                split_issues = 0
                for image in images:
                    annotation = annotation_path_for(image, folder)
                    if not annotation.is_file():
                        # A missing TXT is allowed by YOLO and denotes background.
                        split_backgrounds += 1
                        continue
                    split_labels += 1
                    try:
                        lines = annotation.read_text(encoding='utf-8-sig').splitlines()
                    except (OSError, UnicodeError) as exc:
                        errors.append(f'{annotation}: cannot read label: {exc}')
                        split_issues += 1
                        continue
                    if not any(line.strip() for line in lines):
                        split_backgrounds += 1
                    for lineno, line in enumerate(lines, 1):
                        if not line.strip():
                            continue
                        values = line.split()
                        error = None
                        if len(values) != 5:
                            error = 'expected exactly 5 numbers: class x_center y_center width height'
                        else:
                            try:
                                cls_raw = float(values[0])
                                cls = int(cls_raw)
                                x, y, w, h = map(float, values[1:])
                                if not np.isfinite([cls_raw, x, y, w, h]).all():
                                    error = 'non-finite number'
                                elif cls != cls_raw or cls not in names:
                                    error = 'class ID outside data.yaml names'
                                elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                    error = 'box values must be normalized between 0 and 1'
                                elif x - w / 2 < -1e-5 or x + w / 2 > 1.00001 or y - h / 2 < -1e-5 or y + h / 2 > 1.00001:
                                    error = 'box extends outside image coordinates'
                                else:
                                    per_class[cls] += 1
                                    split_boxes += 1
                            except (ValueError, OverflowError):
                                error = 'not a numeric YOLO annotation'
                        if error:
                            split_issues += 1
                            if len(errors) < 100:
                                errors.append(f'{annotation.name}, line {lineno}: {error} ({annotation})')
                summary.append({
                    'split': split, 'image_folder': str(folder),
                    'images': split_images, 'label_files': split_labels,
                    'background_images': split_backgrounds, 'boxes': split_boxes,
                    'issues': split_issues,
                })
                total_images += split_images
                total_boxes += split_boxes
                total_issues += split_issues
                if split_images == 0:
                    total_issues += 1
                    errors.append(f"'{split}' split contains no supported images: {folder}")
            if not summary:
                raise ValueError('No train, val, or test folders are defined in data.yaml.')
            if 'train' not in {row['split'] for row in summary}:
                errors.append('No train split found: add a training split before fine-tuning.')
            if 'val' not in {row['split'] for row in summary}:
                errors.append('No val split found: a validation split is recommended.')
            self.completed.emit({
                'yaml': str(Path(self.yaml_path).resolve()), 'names': names,
                'summary': summary, 'per_class': per_class, 'errors': errors,
                'images': total_images, 'boxes': total_boxes, 'issues': total_issues,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetAuditDialog(QDialog):
    """Scan a downloaded underwater dataset and export an honest data audit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 10 — Check Underwater Dataset')
        self.resize(900, 680)
        self.worker = None
        self.audit = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            '1. Download and extract a labeled underwater dataset in YOLO format. '
            '2. Select its data.yaml. 3. Click Check Dataset. '
            'This checks annotations, NOT model accuracy. '
            'Missing label files may represent legitimate background images.'
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        source = QLabel(
            'Starter dataset: Roboflow Aquarium (YOLO format). '
            'Remember that aquarium images are not the same as open-water images.'
        )
        source.setWordWrap(True)
        layout.addWidget(source)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText('Choose the extracted dataset/data.yaml')
        row.addWidget(self.path_edit, 1)
        choose = QPushButton('Choose data.yaml')
        choose.clicked.connect(self.choose_yaml)
        row.addWidget(choose)
        layout.addLayout(row)
        buttons = QHBoxLayout()
        self.check_button = QPushButton('Check Dataset')
        self.check_button.clicked.connect(self.start_audit)
        buttons.addWidget(self.check_button)
        self.save_button = QPushButton('Export Dataset Audit CSV')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.export_audit)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self.status = QLabel('Ready. Select the data.yaml file from your extracted dataset.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ['Split', 'Images', 'Label files', 'Backgrounds', 'Boxes', 'Issues']
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        layout.addWidget(QLabel('Class distribution (number of valid labeled objects):'))
        self.classes_label = QLabel('Not checked yet')
        self.classes_label.setWordWrap(True)
        layout.addWidget(self.classes_label)
        layout.addWidget(QLabel('Warnings and annotation problems (first 100):'))
        from PySide6.QtWidgets import QTextEdit
        self.issues_text = QTextEdit()
        self.issues_text.setReadOnly(True)
        layout.addWidget(self.issues_text, 1)

    def choose_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose dataset data.yaml', '', 'YAML files (*.yaml *.yml)'
        )
        if path:
            self.path_edit.setText(path)

    def start_audit(self):
        path = Path(self.path_edit.text().strip())
        if not path.is_file():
            QMessageBox.warning(self, 'Dataset not found', 'First select an existing data.yaml file.')
            return
        self.audit = None
        self.save_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.status.setText('Checking images and labels...')
        self.worker = DatasetAuditWorker(str(path.resolve()), self)
        self.worker.completed.connect(self.on_result)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(lambda: self.check_button.setEnabled(True))
        self.worker.start()

    def on_result(self, result):
        self.audit = result
        rows = result['summary']
        self.table.setRowCount(len(rows))
        fields = ('split', 'images', 'label_files', 'background_images', 'boxes', 'issues')
        for r, record in enumerate(rows):
            for c, field in enumerate(fields):
                self.table.setItem(r, c, QTableWidgetItem(str(record[field])))
        self.classes_label.setText(
            ', '.join(f"{result['names'][cls]}: {count}"
                      for cls, count in result['per_class'].items())
        )
        self.issues_text.setPlainText(
            '\n'.join(result['errors']) if result['errors']
            else 'No annotation format issues found in the checked splits.'
        )
        self.status.setText(
            f"Dataset checked: {result['images']} images, {result['boxes']} valid boxes, "
            f"{result['issues']} structural/annotation problems. "
            'This is a structural audit, not an accuracy score. '
            'Inspect samples visually before using the data in a thesis.'
        )
        self.save_button.setEnabled(True)

    def on_error(self, message):
        self.status.setText('Could not check this dataset. See the error message.')
        QMessageBox.critical(self, 'Dataset check failed', message)

    def export_audit(self):
        if not self.audit:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export dataset summary', 'step10_dataset_audit.csv', 'CSV (*.csv)'
        )
        if not filename:
            return
        fields = ['dataset_yaml', 'split', 'image_folder', 'images', 'label_files',
                  'background_images', 'boxes', 'issues']
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for record in self.audit['summary']:
                    writer.writerow({'dataset_yaml': self.audit['yaml'], **record})
            self.status.setText(f'Dataset audit exported to {filename}')
        except OSError as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, 'Dataset check running', 'Finish checking before closing this window.'
            )
            event.ignore()
            return
        super().closeEvent(event)



# ============================================================================
# Step 11: beginner-friendly, background-thread YOLO fine-tuning.
# Training uses ORIGINAL labeled images. We will evaluate enhancement methods
# on a HELD-OUT set in a future experiment.
# ============================================================================

class UnderwaterTrainingWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset_yaml, starting_weights, epochs, batch, image_size,
                 device_choice, parent=None):
        super().__init__(parent)
        self.dataset_yaml = str(dataset_yaml)
        self.starting_weights = starting_weights
        self.epochs = epochs
        self.batch = batch
        self.image_size = image_size
        self.device_choice = device_choice

    def run(self):
        try:
            import yaml
            with open(self.dataset_yaml, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
            if not isinstance(data, dict):
                raise ValueError('The data.yaml file is not a valid dataset configuration.')
            class_names = dataset_class_names(data.get('names'))
            for split in ('train', 'val'):
                folder = locate_dataset_split(self.dataset_yaml, data, split)
                if not folder or not any(
                    f.suffix.lower() in DATASET_IMAGE_SUFFIXES
                    for f in folder.rglob('*') if f.is_file()
                ):
                    raise ValueError(f'The {split} images folder is empty.')
            self.progress.emit(0, self.epochs, 'Loading the pretrained YOLO model...')
            model = YOLO(self.starting_weights)

            def epoch_finished(trainer):
                # This callback runs inside the background training thread.
                # Emit a Qt signal: never directly modify widgets from here.
                number = min(int(trainer.epoch) + 1, self.epochs)
                self.progress.emit(number, self.epochs,
                                   f'Completed epoch {number} of {self.epochs}')

            model.add_callback('on_fit_epoch_end', epoch_finished)
            project = Path(self.dataset_yaml).resolve().parent / 'training_runs'
            options = dict(
                data=self.dataset_yaml,
                epochs=self.epochs,
                imgsz=self.image_size,
                batch=self.batch,
                workers=0,  # Reliable for Windows and frozen applications.
                project=str(project),
                name='aquarium_training',
                exist_ok=False,  # Never overwrite an earlier research run.
                plots=True,
                save=True,
                verbose=True,
            )
            if self.device_choice != 'Automatic':
                options['device'] = {'CPU': 'cpu', 'NVIDIA GPU (CUDA:0)': 0,
                                     'Apple Silicon (MPS)': 'mps'}[self.device_choice]
            self.progress.emit(0, self.epochs,
                               'Training started. See VS Code terminal for detailed logs.')
            model.train(**options)
            trainer = model.trainer
            run_dir = Path(trainer.save_dir).resolve()
            best_path = run_dir / 'weights' / 'best.pt'
            last_path = run_dir / 'weights' / 'last.pt'
            if not best_path.is_file():
                raise RuntimeError('Training ended without best.pt. Check the terminal logs.')
            # Keep a compact experiment record for thesis reproducibility.
            record = {
                'dataset_yaml': self.dataset_yaml,
                'starting_weights': self.starting_weights,
                'epochs_requested': self.epochs,
                'batch_size': self.batch,
                'image_size': self.image_size,
                'device_selection': self.device_choice,
                'class_names': class_names,
                'best_checkpoint': str(best_path),
                'last_checkpoint': str(last_path) if last_path.is_file() else '',
            }
            with (run_dir / 'step11_experiment.json').open('w', encoding='utf-8') as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
            self.completed.emit({
                'best_path': str(best_path),
                'last_path': str(last_path) if last_path.is_file() else '',
                'run_dir': str(run_dir),
                'dataset_yaml': self.dataset_yaml,
                'starting_weights': self.starting_weights,
                'epochs_requested': self.epochs,
                'batch': self.batch,
                'image_size': self.image_size,
                'classes': class_names,
            })
        except Exception as error:
            self.failed.emit(f'{type(error).__name__}: {error}')


class UnderwaterTrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 11 — Train Underwater YOLO')
        self.resize(790, 520)
        self.worker = None
        self.last_result = None
        outer = QVBoxLayout(self)
        intro = QLabel(
            'Teach YOLO your aquarium classes using the labeled TRAIN images. '
            'The VAL images are used during training; keep TEST images for later.'
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        yaml_row = QHBoxLayout()
        self.yaml_field = QLineEdit()
        self.yaml_field.setPlaceholderText('Choose the Aquarium data.yaml file')
        yaml_row.addWidget(self.yaml_field)
        choose = QPushButton('Choose data.yaml')
        choose.clicked.connect(self.choose_yaml)
        yaml_row.addWidget(choose)
        outer.addLayout(yaml_row)

        options = QGridLayout()
        options.addWidget(QLabel('Starting model:'), 0, 0)
        self.model_choice = QComboBox()
        self.model_choice.addItems(['yolo26n.pt'])
        options.addWidget(self.model_choice, 0, 1)
        options.addWidget(QLabel('Training epochs:'), 1, 0)
        self.epochs_box = QSpinBox()
        self.epochs_box.setRange(1, 200)
        self.epochs_box.setValue(10)
        self.epochs_box.setToolTip('One epoch means one pass through all training images.')
        options.addWidget(self.epochs_box, 1, 1)
        options.addWidget(QLabel('Images per batch:'), 2, 0)
        self.batch_box = QSpinBox()
        self.batch_box.setRange(1, 32)
        self.batch_box.setValue(4)
        options.addWidget(self.batch_box, 2, 1)
        options.addWidget(QLabel('Image resolution:'), 3, 0)
        self.size_choice = QComboBox()
        self.size_choice.addItems(['640', '416', '320'])
        options.addWidget(self.size_choice, 3, 1)
        options.addWidget(QLabel('Training hardware:'), 4, 0)
        self.device_choice = QComboBox()
        self.device_choice.addItems(['Automatic', 'CPU', 'NVIDIA GPU (CUDA:0)',
                                     'Apple Silicon (MPS)'])
        options.addWidget(self.device_choice, 4, 1)
        outer.addLayout(options)

        note = QLabel(
            'Begin with 10 epochs to check the workflow. Full scientific runs will '
            'need more carefully chosen settings. Training may be slow without a GPU. '
            'Results will be saved in a new training_runs folder beside data.yaml.'
        )
        note.setWordWrap(True)
        outer.addWidget(note)
        self.start_button = QPushButton('Start Training')
        self.start_button.clicked.connect(self.start_training)
        outer.addWidget(self.start_button)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        outer.addWidget(self.bar)
        self.status = QLabel('Ready. Select your Aquarium data.yaml to begin.')
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.result_label = QLabel('Your trained best.pt file path will appear here.')
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.result_label)
        self.use_evaluation_button = QPushButton('Open Step 9 Evaluation')
        self.use_evaluation_button.setEnabled(False)
        self.use_evaluation_button.clicked.connect(self.open_evaluation)
        outer.addWidget(self.use_evaluation_button)
        outer.addStretch()

    def choose_yaml(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Choose training dataset YAML', '', 'YAML (*.yaml *.yml)')
        if filename:
            self.yaml_field.setText(filename)

    def start_training(self):
        path = self.yaml_field.text().strip().strip('"')
        if not Path(path).is_file():
            QMessageBox.warning(self, 'Missing dataset', 'Select an existing data.yaml first.')
            return
        if self.worker and self.worker.isRunning():
            return
        self.last_result = None
        self.use_evaluation_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.bar.setValue(0)
        self.status.setText('Preparing training...')
        self.result_label.setText('Training has not completed yet.')
        self.worker = UnderwaterTrainingWorker(
            path, self.model_choice.currentText(), self.epochs_box.value(),
            self.batch_box.value(), int(self.size_choice.currentText()),
            self.device_choice.currentText(), self)
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_thread_finished)
        self.worker.start()

    def on_progress(self, epoch, total, message):
        self.bar.setValue(round(epoch / max(total, 1) * 100))
        self.status.setText(message)

    def on_completed(self, result):
        self.last_result = result
        self.bar.setValue(100)
        self.status.setText('Training finished successfully!')
        self.result_label.setText(
            f"Your trained model is here:\n{result['best_path']}\n\n"
            'Copy this path or use it in the Step 9 evaluation window.')
        self.use_evaluation_button.setEnabled(True)

    def on_failed(self, message):
        self.status.setText('Training failed. Your old application is unchanged.')
        self.result_label.setText(message)
        QMessageBox.critical(self, 'Training Error', message)

    def on_thread_finished(self):
        self.start_button.setEnabled(True)

    def open_evaluation(self):
        if not self.last_result:
            return
        dialog = GroundTruthEvaluationDialog(self)
        dialog.yaml_edit.setText(self.last_result['dataset_yaml'])
        dialog.weights_edit.setText(self.last_result['best_path'])
        dialog.exec()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, 'Training still running',
                'Keep this window open until training finishes. '
                'Closing it now could interrupt saving the model.')
            event.ignore()
            return
        super().closeEvent(event)


# ============================================================================
# Step 15 — Controlled training experiments (separate from Step 11 practice).
# Uses TRAIN and VAL only. Never uses TEST for selecting the model.
# Each run has a fresh output folder and a reproducibility manifest.
# ============================================================================

class ResearchTrainingWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset_yaml, epochs, batch, image_size, patience,
                 seed, device_choice, parent=None):
        super().__init__(parent)
        self.dataset_yaml = str(Path(dataset_yaml).resolve())
        self.epochs = epochs
        self.batch = batch
        self.image_size = image_size
        self.patience = patience
        self.seed = seed
        self.device_choice = device_choice

    def run(self):
        try:
            import yaml
            import ultralytics
            import torch

            with open(self.dataset_yaml, encoding='utf-8') as file:
                data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError('The dataset YAML must describe a dataset.')
            names = dataset_class_names(data.get('names'))
            split_counts = {}
            for split in ('train', 'val'):
                folder = locate_dataset_split(self.dataset_yaml, data, split)
                images = [p for p in folder.rglob('*')
                          if p.is_file() and p.suffix.lower() in DATASET_IMAGE_SUFFIXES]
                if not images:
                    raise ValueError(f'{split} folder does not contain images.')
                split_counts[split] = len(images)
            if self.device_choice == 'NVIDIA GPU (CUDA:0)' and not torch.cuda.is_available():
                raise ValueError('CUDA GPU not available. Choose Automatic or CPU.')
            if self.device_choice == 'Apple Silicon (MPS)' and not torch.backends.mps.is_available():
                raise ValueError('Apple MPS not available. Choose Automatic or CPU.')

            starting_model = 'yolo26n.pt'
            model = YOLO(starting_model)
            self.progress.emit(0, self.epochs, 'Preparing a new training experiment...')

            def epoch_finished(trainer):
                epoch = min(int(trainer.epoch) + 1, self.epochs)
                self.progress.emit(epoch, self.epochs,
                                   f'Epoch {epoch}/{self.epochs} completed (early stopping may finish sooner)')

            model.add_callback('on_fit_epoch_end', epoch_finished)
            output_root = Path(self.dataset_yaml).parent / 'training_runs'
            options = {
                'data': self.dataset_yaml,
                'epochs': self.epochs,
                'batch': self.batch,
                'imgsz': self.image_size,
                'patience': self.patience,
                'seed': self.seed,
                'deterministic': True,
                'workers': 0,  # Compatible with Windows Qt application process.
                'project': str(output_root),
                'name': 'step15_research',
                'exist_ok': False,
                'plots': True,
                'save': True,
                'verbose': True,
            }
            if self.device_choice != 'Automatic':
                options['device'] = {'CPU': 'cpu', 'NVIDIA GPU (CUDA:0)': 0,
                                     'Apple Silicon (MPS)': 'mps'}[self.device_choice]
            self.progress.emit(0, self.epochs, 'Training started. Detailed log is in the terminal.')
            model.train(**options)
            trainer = model.trainer
            if trainer is None:
                raise RuntimeError('Training did not produce a trainer instance.')
            run_dir = Path(trainer.save_dir).resolve()
            best = run_dir / 'weights' / 'best.pt'
            if not best.is_file():
                raise RuntimeError('Training stopped without creating best.pt. Check terminal output.')
            actual_epochs = int(trainer.epoch) + 1
            manifest = {
                'step': 15,
                'purpose': 'Validation-based model training, never test-set tuning',
                'dataset_yaml': self.dataset_yaml,
                'dataset_image_counts': split_counts,
                'class_names': names,
                'starting_weights': starting_model,
                'epochs_requested': self.epochs,
                'epochs_completed': actual_epochs,
                'early_stopping_patience': self.patience,
                'batch_size': self.batch,
                'image_size': self.image_size,
                'seed': self.seed,
                'deterministic_requested': True,
                'device_selection': self.device_choice,
                'ultralytics_version': ultralytics.__version__,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'best_checkpoint': str(best),
                'training_results_csv': str(run_dir / 'results.csv'),
                'note': 'GPU algorithms may still be nondeterministic; test split not used here.',
            }
            with (run_dir / 'step15_experiment.json').open('w', encoding='utf-8') as file:
                json.dump(manifest, file, indent=2, ensure_ascii=False)
            self.completed.emit({
                'run_dir': str(run_dir),
                'best_path': str(best),
                'dataset_yaml': self.dataset_yaml,
                'epochs_completed': actual_epochs,
                'epochs_requested': self.epochs,
                'manifest': str(run_dir / 'step15_experiment.json'),
                'results_csv': str(run_dir / 'results.csv'),
            })
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class ResearchTrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 15 — Improve Your Underwater Model')
        self.resize(780, 560)
        self.worker = None
        self.last_result = None
        layout = QVBoxLayout(self)
        explanation = QLabel(
            'This is a careful training experiment, not just a 10-epoch practice run. '
            'We train on TRAIN images, choose the best checkpoint using VAL, '
            'and keep TEST for final unbiased evaluation.'
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        dataset_row = QHBoxLayout()
        self.yaml_field = QLineEdit()
        self.yaml_field.setPlaceholderText('Select the Aquarium data.yaml')
        dataset_row.addWidget(self.yaml_field)
        browse = QPushButton('Choose data.yaml')
        browse.clicked.connect(self.choose_yaml)
        dataset_row.addWidget(browse)
        layout.addLayout(dataset_row)

        form = QGridLayout()
        form.addWidget(QLabel('Starting model:'), 0, 0)
        form.addWidget(QLabel('yolo26n.pt (same baseline for every training run)'), 0, 1)
        form.addWidget(QLabel('Maximum epochs:'), 1, 0)
        self.epochs_box = QSpinBox()
        self.epochs_box.setRange(1, 300)
        self.epochs_box.setValue(50)
        self.epochs_box.setToolTip('An epoch is one pass over the training set. Training may stop early.')
        form.addWidget(self.epochs_box, 1, 1)
        form.addWidget(QLabel('Early-stopping patience:'), 2, 0)
        self.patience_box = QSpinBox()
        self.patience_box.setRange(1, 100)
        self.patience_box.setValue(12)
        self.patience_box.setToolTip('Stop after this many epochs without validation improvement.')
        form.addWidget(self.patience_box, 2, 1)
        form.addWidget(QLabel('Batch size:'), 3, 0)
        self.batch_box = QSpinBox()
        self.batch_box.setRange(1, 32)
        self.batch_box.setValue(4)
        form.addWidget(self.batch_box, 3, 1)
        form.addWidget(QLabel('Image resolution:'), 4, 0)
        self.size_box = QComboBox()
        self.size_box.addItems(['640', '416', '320'])
        form.addWidget(self.size_box, 4, 1)
        form.addWidget(QLabel('Random seed:'), 5, 0)
        self.seed_box = QSpinBox()
        self.seed_box.setRange(0, 999999)
        self.seed_box.setValue(42)
        self.seed_box.setToolTip('Controls many sources of randomness for a more repeatable experiment.')
        form.addWidget(self.seed_box, 5, 1)
        form.addWidget(QLabel('Hardware:'), 6, 0)
        self.device_box = QComboBox()
        self.device_box.addItems(['Automatic', 'CPU', 'NVIDIA GPU (CUDA:0)',
                                  'Apple Silicon (MPS)'])
        form.addWidget(self.device_box, 6, 1)
        layout.addLayout(form)

        warning = QLabel(
            'Suggested first research run: 50 epochs maximum, patience 12, batch 4, '
            'image size 640, seed 42. Training may take a long time on a CPU. '
            'Keep this window open until it finishes. Do not use your TEST images '
            'to select training settings.'
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.start_button = QPushButton('Start Step 15 Training')
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel('Ready. Choose a dataset to begin.')
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.result_label = QLabel('When finished, your best.pt and experiment record will appear here.')
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result_label)
        self.evaluate_button = QPushButton('Evaluate Trained Model (Step 9)')
        self.evaluate_button.setEnabled(False)
        self.evaluate_button.clicked.connect(self.open_evaluation)
        layout.addWidget(self.evaluate_button)
        layout.addStretch()

    def choose_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose your underwater dataset', '', 'Dataset YAML (*.yaml *.yml)')
        if path:
            self.yaml_field.setText(path)

    def start_training(self):
        path = Path(self.yaml_field.text().strip().strip('"'))
        if not path.is_file():
            QMessageBox.warning(self, 'Dataset missing', 'First choose an existing data.yaml file.')
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.last_result = None
        self.start_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText('Checking dataset and loading the YOLO model...')
        self.result_label.setText('Training is in progress. See terminal for detailed logs.')
        self.worker = ResearchTrainingWorker(
            path, self.epochs_box.value(), self.batch_box.value(),
            int(self.size_box.currentText()), self.patience_box.value(),
            self.seed_box.value(), self.device_box.currentText(), self)
        self.worker.progress.connect(self.show_progress)
        self.worker.completed.connect(self.completed_training)
        self.worker.failed.connect(self.failed_training)
        self.worker.finished.connect(self.training_finished)
        self.worker.start()

    def show_progress(self, epoch, total, message):
        self.progress_bar.setValue(round(100 * epoch / max(total, 1)))
        self.status_label.setText(message)

    def completed_training(self, result):
        self.last_result = result
        self.progress_bar.setValue(100)
        self.status_label.setText(
            f"Training completed ({result['epochs_completed']} epochs, "
            f"requested maximum {result['epochs_requested']}).")
        self.result_label.setText(
            f"Your new model:\n{result['best_path']}\n\n"
            f"Experiment settings:\n{result['manifest']}\n\n"
            f"Training history:\n{result['results_csv']}")
        self.evaluate_button.setEnabled(True)

    def failed_training(self, message):
        self.status_label.setText('Training failed. Previous best.pt files remain unchanged.')
        self.result_label.setText(message)
        QMessageBox.critical(self, 'Step 15 Training Error', message)

    def training_finished(self):
        self.start_button.setEnabled(True)

    def open_evaluation(self):
        if not self.last_result:
            return
        dlg = GroundTruthEvaluationDialog(self)
        dlg.yaml_edit.setText(self.last_result['dataset_yaml'])
        dlg.weights_edit.setText(self.last_result['best_path'])
        dlg.exec()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, 'Training active',
                                    'Leave this dialog open while training is running.')
            event.ignore()
            return
        super().closeEvent(event)


# ============================================================================
# Step 13: evaluate the SAME trained model against the SAME labeled test
# images after each image transformation. Only images are modified; ground
# truth labels are copied without changing their box coordinates.
# ============================================================================

class EnhancementAccuracyWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, yaml_file, weights, split, settings, parent=None):
        super().__init__(parent)
        self.yaml_file = str(yaml_file)
        self.weights = str(weights)
        self.split = split
        self.settings = dict(settings)

    def run(self):
        try:
            import yaml
            with open(self.yaml_file, 'r', encoding='utf-8') as file:
                dataset = yaml.safe_load(file)
            if not isinstance(dataset, dict):
                raise ValueError('data.yaml must contain a YAML mapping.')
            class_names = dataset_class_names(dataset.get('names'))
            source_dir = locate_dataset_split(self.yaml_file, dataset, self.split)
            if source_dir is None:
                raise ValueError(f"No '{self.split}' split in the dataset configuration.")
            images = sorted(p for p in source_dir.rglob('*')
                            if p.is_file() and p.suffix.lower() in DATASET_IMAGE_SUFFIXES)
            if not images:
                raise ValueError(f'No images found inside {source_dir}.')
            model = YOLO(self.weights)
            if getattr(model, 'task', None) != 'detect':
                raise ValueError('Please select an object detection checkpoint (.pt).')
            model_raw = model.names
            model_names = ({int(k): str(v) for k, v in model_raw.items()}
                           if isinstance(model_raw, dict)
                           else {i: str(v) for i, v in enumerate(model_raw)})
            if (set(model_names) != set(class_names) or
                    any(model_names[i].strip().casefold() != name.strip().casefold()
                        for i, name in class_names.items())):
                raise ValueError('Model classes do not match dataset classes.\n'
                                 f'Model: {model_names}\nDataset: {class_names}\n'
                                 'Select the best.pt trained on this dataset.')

            # A separate temporary dataset is constructed for each method.
            # Original data and annotations remain untouched.
            results = []
            with tempfile.TemporaryDirectory(prefix='underwater_step13_') as temp_dir:
                root = Path(temp_dir)
                for method_index, method in enumerate(COMPARISON_METHODS, 1):
                    self.progress.emit(
                        f'Method {method_index}/4: {method}. Preparing {len(images)} images...')
                    method_root = root / f'method_{method_index}'
                    destination_images = method_root / 'images' / 'eval'
                    destination_labels = method_root / 'labels' / 'eval'
                    destination_images.mkdir(parents=True, exist_ok=True)
                    destination_labels.mkdir(parents=True, exist_ok=True)
                    for index, image_path in enumerate(images):
                        if self.isInterruptionRequested():
                            raise RuntimeError('Accuracy evaluation interrupted.')
                        rel = image_path.relative_to(source_dir)
                        gt_label = annotation_path_for(image_path, source_dir)
                        if gt_label.is_file():
                            target_label = (destination_labels / rel).with_suffix('.txt')
                            target_label.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(gt_label, target_label)
                        if method == 'Original (No Enhancement)':
                            target_image = destination_images / rel
                            target_image.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(image_path, target_image)
                        else:
                            target_image = (destination_images / rel).with_suffix('.png')
                            target_image.parent.mkdir(parents=True, exist_ok=True)
                            pixels = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                                                  cv2.IMREAD_COLOR)
                            if pixels is None:
                                raise ValueError(f'Cannot decode image: {image_path}')
                            output = apply_enhancement(pixels, method, self.settings[method])
                            success, encoded = cv2.imencode('.png', output)
                            if not success:
                                raise ValueError(f'Cannot encode image: {image_path}')
                            encoded.tofile(str(target_image))
                        if (index + 1) % 100 == 0:
                            self.progress.emit(
                                f'{method}: prepared {index + 1}/{len(images)} images')
                    # Use the same split name in a generated isolated YAML.
                    # Ultralytics expects the image and label trees to match.
                    temporary_yaml = method_root / 'data.yaml'
                    temporary_data = {
                        'path': str(method_root.resolve()),
                        'train': 'images/eval',
                        'val': 'images/eval',
                        'test': 'images/eval',
                        'nc': len(class_names),
                        'names': [class_names[i] for i in range(len(class_names))],
                    }
                    with open(temporary_yaml, 'w', encoding='utf-8') as file:
                        yaml.safe_dump(temporary_data, file, sort_keys=False)
                    self.progress.emit(f'{method}: calculating precision, recall, mAP...')
                    scores = model.val(
                        data=str(temporary_yaml), split='val', imgsz=640,
                        batch=4, conf=0.001, iou=0.7, workers=0,
                        plots=False, verbose=False,
                        project=str(root / 'validation_runs'),
                        name=f'method_{method_index}', exist_ok=True,
                    )
                    box = scores.box
                    result = {
                        'method': method,
                        'precision': float(box.mp),
                        'recall': float(box.mr),
                        'map50': float(box.map50),
                        'map5095': float(box.map),
                        'images': len(images),
                        'parameter': ('none' if method == 'Original (No Enhancement)'
                                      else self.settings[method]),
                        'split': self.split,
                        'weights': str(Path(self.weights).resolve()),
                        'dataset_yaml': str(Path(self.yaml_file).resolve()),
                        'imgsz': 640,
                        'conf': 0.001,
                        'nms_iou': 0.7,
                    }
                    if not all(np.isfinite(result[k])
                               for k in ('precision', 'recall', 'map50', 'map5095')):
                        raise ValueError(f'Invalid validation scores for {method}.')
                    results.append(result)
                    self.progress.emit(f'{method}: mAP@50={result["map50"]:.4f}')
            self.completed.emit(results)
        except Exception as error:
            import traceback
            self.failed.emit(f'{error}\n\n{traceback.format_exc()}')


# ============================================================================
# Step 14: thesis-oriented accuracy report. Results are measured by Step 13;
# this report never invents or recomputes model predictions.
# ============================================================================

ACCURACY_METRICS = (
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('map50', 'mAP@50'),
    ('map5095', 'mAP@50-95'),
)


def validate_accuracy_records(records):
    """Validate one four-method comparison before charting or comparing scores."""
    if len(records) != len(COMPARISON_METHODS):
        raise ValueError('Expected exactly four rows (one per enhancement method).')
    by_method = {}
    for row in records:
        method = str(row.get('method', '')).strip()
        if method not in COMPARISON_METHODS or method in by_method:
            raise ValueError('Results must contain each of the four methods exactly once.')
        clean = dict(row)
        for field, _ in ACCURACY_METRICS:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{method}: invalid {field}.') from error
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{method}: {field} must be between 0 and 1.')
            clean[field] = value
        try:
            count = int(row['images'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{method}: invalid image count.') from error
        if count < 1:
            raise ValueError(f'{method}: image count must be positive.')
        clean['images'] = count
        by_method[method] = clean
    ordered = [by_method[m] for m in COMPARISON_METHODS]
    if len({r['images'] for r in ordered}) != 1:
        raise ValueError('Image counts differ. Only compare runs on identical images.')
    # Empty values are permitted for old CSVs, but if identifying metadata are
    # present, they must not conflict across the four rows.
    for field in ('weights', 'dataset_yaml', 'split', 'imgsz', 'conf', 'nms_iou'):
        values = {str(r.get(field, '')).strip() for r in ordered
                  if str(r.get(field, '')).strip()}
        if len(values) > 1:
            raise ValueError(f'Inconsistent {field} between the four methods.')
    return ordered


class AccuracyMetricChart(QWidget):
    """Bar chart using a fixed 0..1 metric axis; can draw to screen or image."""
    def __init__(self, records, key, label, parent=None):
        super().__init__(parent)
        self.records = records
        self.key = key
        self.label = label
        self.setMinimumSize(730, 244)

    def draw_chart(self, painter, width):
        h = 244
        painter.fillRect(0, 0, width, h, QColor('white'))
        painter.setPen(QColor('#163043'))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(18, 27, self.label + ' (scale: 0 to 1)')
        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        left = 186
        bar_area = max(100, width - left - 94)
        painter.setPen(QPen(QColor('#d9e2e8'), 1))
        for tick in range(5):
            x = left + round(bar_area * tick / 4)
            painter.drawLine(x, 43, x, 221)
            painter.setPen(QColor('#596b78'))
            painter.drawText(x - 12, 239, f'{tick/4:.2f}')
            painter.setPen(QPen(QColor('#d9e2e8'), 1))
        colors = ('#176b87', '#168579', '#c58b2a', '#7860af')
        for i, row in enumerate(self.records):
            y = 52 + i * 43
            painter.setPen(QColor('#163043'))
            method = row['method'].replace(' (No Enhancement)', '')
            painter.drawText(12, y + 17, method)
            value = row[self.key]
            painter.fillRect(left, y, max(1, round(bar_area * value)), 22,
                             QColor(colors[i]))
            painter.drawText(left + round(bar_area * value) + 7, y + 17,
                             f'{value:.4f}')

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.draw_chart(painter, self.width())
        finally:
            painter.end()


class AccuracyReportDialog(QDialog):
    """Step 14: chart actual Step 13 accuracy metrics and baseline differences."""
    def __init__(self, records, source_label='Step 13 results', parent=None):
        super().__init__(parent)
        self.records = validate_accuracy_records(records)
        self.setWindowTitle('Step 14 — Accuracy Charts and Research Summary')
        self.resize(940, 840)
        layout = QVBoxLayout(self)
        title = QLabel(f'Ground-truth accuracy report — {source_label}')
        title.setWordWrap(True)
        layout.addWidget(title)
        intro = QLabel(
            'All numbers come from the labeled-data evaluation, not box counts. '
            'Changes are percentage POINTS relative to the original-image baseline. '
            'A higher score on this dataset is not proof of a general improvement; '
            'we will test repeated runs and other underwater conditions later.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        baseline = self.records[0]
        self.table = QTableWidget(4, 6)
        self.table.setHorizontalHeaderLabels([
            'Method', 'Precision', 'Recall', 'mAP@50', 'mAP@50-95',
            'mAP@50-95 vs original (pp)'])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(180)
        for i, record in enumerate(self.records):
            delta = (record['map5095'] - baseline['map5095']) * 100
            cells = [record['method'],
                     *[f'{record[k]:.4f}' for k, _ in ACCURACY_METRICS],
                     f'{delta:+.2f} pp']
            for j, value in enumerate(cells):
                self.table.setItem(i, j, QTableWidgetItem(value))
        layout.addWidget(self.table)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_host = QWidget()
        graph_layout = QVBoxLayout(scroll_host)
        self.charts = [AccuracyMetricChart(self.records, field, title)
                       for field, title in ACCURACY_METRICS]
        for chart in self.charts:
            graph_layout.addWidget(chart)
        scroll.setWidget(scroll_host)
        layout.addWidget(scroll, 1)
        controls = QHBoxLayout()
        export_png = QPushButton('Save Thesis Charts PNG')
        export_png.clicked.connect(self.save_png)
        controls.addWidget(export_png)
        export_csv = QPushButton('Export Accuracy Summary CSV')
        export_csv.clicked.connect(self.save_summary)
        controls.addWidget(export_csv)
        close = QPushButton('Close')
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        layout.addLayout(controls)

    def save_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Step 14 charts', 'step14_accuracy_charts.png',
            'PNG Images (*.png)')
        if not path:
            return
        try:
            width = 900
            height = len(self.charts) * 244 + 20
            image = QImage(width, height, QImage.Format.Format_RGB32)
            if image.isNull():
                raise MemoryError('Cannot allocate image for charts.')
            image.fill(QColor('white'))
            painter = QPainter(image)
            if not painter.isActive():
                raise RuntimeError('Cannot start chart drawing.')
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                for i, chart in enumerate(self.charts):
                    painter.save()
                    try:
                        painter.translate(0, 10 + i * 244)
                        chart.draw_chart(painter, width)
                    finally:
                        painter.restore()
            finally:
                # Qt must finish painting before QImage is saved or destroyed.
                painter.end()
            if not image.save(path, 'PNG'):
                raise OSError('Qt could not save this PNG.')
            QMessageBox.information(self, 'Saved', 'Accuracy charts saved successfully.')
        except Exception as error:
            QMessageBox.critical(self, 'Chart export failed', str(error))

    def save_summary(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Step 14 summary', 'step14_accuracy_summary.csv',
            'CSV Files (*.csv)')
        if not path:
            return
        baseline = self.records[0]
        try:
            fields = ['method', 'images'] + [key for key, _ in ACCURACY_METRICS]
            fields += ['precision_delta_pp', 'recall_delta_pp',
                       'map50_delta_pp', 'map5095_delta_pp']
            with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for record in self.records:
                    row = {field: record[field] for field in fields[:6]}
                    for metric, _ in ACCURACY_METRICS:
                        row[metric + '_delta_pp'] = round(
                            100 * (record[metric] - baseline[metric]), 3)
                    writer.writerow(row)
            QMessageBox.information(self, 'Saved', 'Summary CSV saved successfully.')
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, 'Summary export failed', str(error))


class EnhancementAccuracyDialog(QDialog):
    def __init__(self, settings, weights, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 13 — Ground-Truth Enhancement Accuracy')
        self.resize(900, 600)
        self.settings = dict(settings)
        self.worker = None
        self.results = []
        layout = QVBoxLayout(self)
        notice = QLabel(
            'This checks the SAME trained detector on ORIGINAL, CLAHE, WHITE BALANCE, '
            'and GAMMA versions of the SAME labeled evaluation images. '
            'Use the Aquarium test split if it exists. Your original files are not modified.'
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

        yaml_row = QHBoxLayout()
        self.yaml_path = QLineEdit()
        self.yaml_path.setPlaceholderText('Aquarium dataset data.yaml')
        yaml_row.addWidget(self.yaml_path, 1)
        choose_yaml = QPushButton('Choose data.yaml')
        choose_yaml.clicked.connect(self.select_yaml)
        yaml_row.addWidget(choose_yaml)
        layout.addLayout(yaml_row)

        model_row = QHBoxLayout()
        self.model_path = QLineEdit(weights)
        model_row.addWidget(QLabel('Trained model:'))
        model_row.addWidget(self.model_path, 1)
        choose_model = QPushButton('Choose best.pt')
        choose_model.clicked.connect(self.select_weights)
        model_row.addWidget(choose_model)
        layout.addLayout(model_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel('Evaluation split:'))
        self.split_selector = QComboBox()
        self.split_selector.addItems(['test', 'val'])
        split_row.addWidget(self.split_selector)
        split_row.addStretch()
        layout.addLayout(split_row)
        settings_text = ', '.join(
            f'{method}: {self.settings[method]}'
            for method in ('CLAHE', 'White Balance', 'Gamma Correction'))
        settings_label = QLabel('Current enhancement slider settings: ' + settings_text)
        settings_label.setWordWrap(True)
        layout.addWidget(settings_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton('Run 4-Method Accuracy Comparison')
        self.start_button.clicked.connect(self.start)
        buttons.addWidget(self.start_button)
        self.export_button = QPushButton('Export Accuracy CSV')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        buttons.addWidget(self.export_button)
        self.view_charts_button = QPushButton('Step 14: View Accuracy Charts')
        self.view_charts_button.setEnabled(False)
        self.view_charts_button.clicked.connect(self.open_charts)
        buttons.addWidget(self.view_charts_button)
        layout.addLayout(buttons)
        self.status = QLabel('Ready. Select your labeled dataset and trained best.pt.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.table = QTableWidget(4, 6)
        self.table.setHorizontalHeaderLabels([
            'Enhancement', 'Images', 'Precision', 'Recall', 'mAP@50', 'mAP@50–95'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        for i, method in enumerate(COMPARISON_METHODS):
            self.table.setItem(i, 0, QTableWidgetItem(method))
            for j in range(1, 6):
                self.table.setItem(i, j, QTableWidgetItem('—'))
        layout.addWidget(self.table)
        note = QLabel(
            'Reminder: a higher mAP means better measured detection performance '
            'on this labeled split. The four runs use identical model weights and '
            'validation settings. Save the exported CSV for your thesis.'
        )
        note.setWordWrap(True)
        layout.addWidget(note)

    def select_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose dataset YAML', '', 'YAML (*.yaml *.yml)')
        if path:
            self.yaml_path.setText(path)

    def select_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose trained checkpoint', '', 'PyTorch weights (*.pt)')
        if path:
            self.model_path.setText(path)

    def start(self):
        if self.worker is not None and self.worker.isRunning():
            return
        dataset_yaml = self.yaml_path.text().strip()
        weights = self.model_path.text().strip()
        if not Path(dataset_yaml).is_file():
            QMessageBox.warning(self, 'Missing dataset', 'Choose your Aquarium data.yaml.')
            return
        if not weights or (weights != 'yolo26n.pt' and not Path(weights).is_file()):
            QMessageBox.warning(self, 'Missing weights', 'Choose your trained best.pt file.')
            return
        self.results = []
        self.export_button.setEnabled(False)
        self.view_charts_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.status.setText('Preparing the first evaluation...')
        self.worker = EnhancementAccuracyWorker(
            dataset_yaml, weights, self.split_selector.currentText(),
            self.settings, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.finished_comparison)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def finished_comparison(self, results):
        self.results = results
        for i, record in enumerate(results):
            for j, value in enumerate([
                record['method'], str(record['images']),
                f"{record['precision']:.4f}", f"{record['recall']:.4f}",
                f"{record['map50']:.4f}", f"{record['map5095']:.4f}",
            ]):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.status.setText('Done! All four methods were evaluated on labeled images.')
        self.export_button.setEnabled(True)

    def show_error(self, message):
        self.status.setText('Evaluation failed — see the error details.')
        QMessageBox.critical(self, 'Accuracy comparison error', message)

    def worker_finished(self):
        self.start_button.setEnabled(True)

    def export(self):
        if not self.results:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Step 13 results', 'step13_accuracy_comparison.csv',
            'CSV files (*.csv)')
        if not filename:
            return
        try:
            columns = list(self.results[0].keys())
            with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.results)
            self.status.setText(f'CSV saved: {filename}')
        except OSError as error:
            QMessageBox.critical(self, 'Could not save CSV', str(error))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, 'Still evaluating',
                'Please finish the comparison before closing this window.')
            event.ignore()
            return
        super().closeEvent(event)



# ============================================================================
# Step 16: Final controlled 2-model x 4-enhancement evaluation.
# Both checkpoints are tested against the SAME labeled images, with identical
# preprocessing and validation settings. No test data is used for training.
# ============================================================================

class FinalExperimentWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, data_yaml, old_weights, new_weights, split, settings, parent=None):
        super().__init__(parent)
        self.data_yaml = str(data_yaml)
        self.checkpoints = [('Step 11 (10-epoch)', str(old_weights)),
                            ('Step 15 (improved)', str(new_weights))]
        self.split = str(split)
        self.settings = dict(settings)

    def run(self):
        try:
            import yaml
            with open(self.data_yaml, 'r', encoding='utf-8') as stream:
                data = yaml.safe_load(stream)
            if not isinstance(data, dict):
                raise ValueError('data.yaml must be a YAML dictionary.')
            names = dataset_class_names(data.get('names'))
            image_dir = locate_dataset_split(self.data_yaml, data, self.split)
            if image_dir is None:
                raise ValueError(f'The dataset has no {self.split!r} split.')
            images = sorted(p for p in image_dir.rglob('*')
                            if p.is_file() and p.suffix.lower() in DATASET_IMAGE_SUFFIXES)
            if not images:
                raise ValueError('The selected split contains no images.')
            if Path(self.checkpoints[0][1]).resolve() == Path(self.checkpoints[1][1]).resolve():
                raise ValueError('Choose two DIFFERENT checkpoints (old and improved).')

            # Build just four temporary image splits and reuse for BOTH models.
            # This prevents differences from image conversion or file selection.
            with tempfile.TemporaryDirectory(prefix='underwater_step16_') as workspace:
                root = Path(workspace)
                yaml_by_method = {}
                for mi, method in enumerate(COMPARISON_METHODS, 1):
                    if self.isInterruptionRequested():
                        raise RuntimeError('Experiment was interrupted.')
                    self.progress.emit(f'Preparing {method}: {mi}/4 ...')
                    method_root = root / f'method_{mi}'
                    output_images = method_root / 'images' / 'eval'
                    output_labels = method_root / 'labels' / 'eval'
                    output_images.mkdir(parents=True)
                    output_labels.mkdir(parents=True)
                    for number, img in enumerate(images, 1):
                        if self.isInterruptionRequested():
                            raise RuntimeError('Experiment was interrupted.')
                        rel = img.relative_to(image_dir)
                        label = annotation_path_for(img, image_dir)
                        if label.exists():
                            dst_label = (output_labels / rel).with_suffix('.txt')
                            dst_label.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(label, dst_label)
                        # Image filenames and relative paths are preserved on original.
                        dst_image = (output_images / rel) if mi == 1 else (output_images / rel).with_suffix('.png')
                        dst_image.parent.mkdir(parents=True, exist_ok=True)
                        if mi == 1:
                            shutil.copy2(img, dst_image)
                        else:
                            pixels = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)
                            if pixels is None:
                                raise ValueError(f'Cannot read {img}')
                            changed = apply_enhancement(pixels, method, self.settings[method])
                            ok, encoded = cv2.imencode('.png', changed)
                            if not ok:
                                raise ValueError(f'Cannot encode enhanced image {img}')
                            encoded.tofile(str(dst_image))
                        if number % 100 == 0:
                            self.progress.emit(f'{method}: {number}/{len(images)} prepared')
                    config = {'path': str(method_root.resolve()),
                              'train': 'images/eval', 'val': 'images/eval', 'test': 'images/eval',
                              'names': [names[i] for i in range(len(names))], 'nc': len(names)}
                    config_path = method_root / 'data.yaml'
                    with open(config_path, 'w', encoding='utf-8') as stream:
                        yaml.safe_dump(config, stream, sort_keys=False)
                    yaml_by_method[method] = str(config_path)

                records = []
                for model_index, (model_label, weights) in enumerate(self.checkpoints, 1):
                    model = YOLO(weights)
                    if getattr(model, 'task', None) != 'detect':
                        raise ValueError(f'{model_label} is not an object detector.')
                    model_raw = model.names
                    actual_names = ({int(k): str(v) for k, v in model_raw.items()}
                                    if isinstance(model_raw, dict)
                                    else {i: str(v) for i, v in enumerate(model_raw)})
                    if set(actual_names) != set(names) or any(
                        actual_names[i].strip().casefold() != names[i].strip().casefold()
                        for i in names
                    ):
                        raise ValueError(
                            f'{model_label} classes do not match the labeled dataset.\n'
                            f'Model: {actual_names}\nDataset: {names}\n'
                            'Use two best.pt files trained on the SAME Aquarium classes.')
                    for mi, method in enumerate(COMPARISON_METHODS, 1):
                        if self.isInterruptionRequested():
                            raise RuntimeError('Experiment was interrupted.')
                        index = (model_index - 1) * 4 + mi
                        self.progress.emit(f'Evaluation {index}/8: {model_label} / {method}')
                        metrics = model.val(
                            data=yaml_by_method[method], split='val', imgsz=640,
                            batch=4, conf=0.001, iou=0.7, workers=0,
                            plots=False, verbose=False,
                            project=str(root / 'validation_runs'),
                            name=f'model_{model_index}_method_{mi}', exist_ok=True,
                        )
                        row = {
                            'model': model_label, 'method': method,
                            'precision': float(metrics.box.mp),
                            'recall': float(metrics.box.mr),
                            'map50': float(metrics.box.map50),
                            'map5095': float(metrics.box.map),
                            'images': len(images), 'split': self.split,
                            'weights': str(Path(weights).resolve()),
                            'dataset_yaml': str(Path(self.data_yaml).resolve()),
                            'parameter': ('none' if mi == 1 else self.settings[method]),
                            'imgsz': 640, 'batch': 4, 'conf': 0.001,
                            'nms_iou': 0.7,
                        }
                        if any(not np.isfinite(row[key]) for key in
                               ('precision', 'recall', 'map50', 'map5095')):
                            raise ValueError('Non-finite metric returned by YOLO validation.')
                        records.append(row)
                        self.progress.emit(
                            f'{index}/8 complete: {model_label}, {method}: '
                            f'mAP@50={row["map50"]:.4f}')
                self.completed.emit(records)
        except Exception as error:
            import traceback
            self.failed.emit(f'{error}\n\n{traceback.format_exc()}')


class FinalExperimentDialog(QDialog):
    """Export full model-by-method scores and a compact paired baseline summary."""
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 16 — Final Controlled Research Experiment')
        self.resize(1120, 700)
        self.settings = dict(settings)
        self.worker = None
        self.records = []
        layout = QVBoxLayout(self)
        intro = QLabel(
            'Compare TWO trained Aquarium models on the SAME labeled test images. '
            'Each model is evaluated on Original, CLAHE, White Balance and Gamma. '
            'This runs EIGHT evaluations and does not change your dataset.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        dataset_row = QHBoxLayout()
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText('Choose Aquarium data.yaml')
        dataset_row.addWidget(QLabel('Dataset:'))
        dataset_row.addWidget(self.dataset_path, 1)
        dataset_choose = QPushButton('Browse...')
        dataset_choose.clicked.connect(self.choose_dataset)
        dataset_row.addWidget(dataset_choose)
        layout.addLayout(dataset_row)

        self.old_path = QLineEdit()
        self.new_path = QLineEdit()
        self.old_path.setPlaceholderText('Step 11 best.pt (10-epoch model)')
        self.new_path.setPlaceholderText('Step 15 best.pt (improved model)')
        for title, field in [('Old model:', self.old_path), ('New model:', self.new_path)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(title))
            row.addWidget(field, 1)
            browse = QPushButton('Browse...')
            browse.clicked.connect(lambda checked=False, target=field: self.choose_model(target))
            row.addWidget(browse)
            layout.addLayout(row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel('Split:'))
        self.split = QComboBox()
        self.split.addItems(['test', 'val'])
        split_row.addWidget(self.split)
        split_row.addStretch()
        layout.addLayout(split_row)
        method_info = QLabel('Enhancement settings: ' + ', '.join(
            f'{method}={self.settings[method]}'
            for method in ('CLAHE', 'White Balance', 'Gamma Correction')))
        method_info.setWordWrap(True)
        layout.addWidget(method_info)
        row = QHBoxLayout()
        self.run_button = QPushButton('Run Final Experiment (8 Evaluations)')
        self.run_button.clicked.connect(self.start)
        row.addWidget(self.run_button)
        self.export_button = QPushButton('Export Final Results CSV')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        row.addWidget(self.export_button)
        self.summary_button = QPushButton('Export Comparison Summary CSV')
        self.summary_button.setEnabled(False)
        self.summary_button.clicked.connect(self.export_summary)
        row.addWidget(self.summary_button)
        layout.addLayout(row)
        self.status = QLabel('Ready. Select the dataset and both trained best.pt files.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(8, 6)
        self.table.setHorizontalHeaderLabels(
            ['Model', 'Enhancement', 'Precision', 'Recall', 'mAP@50', 'mAP@50–95'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        disclaimer = QLabel(
            'Scientific note: a test split that you already used for model decisions is no '
            'longer an untouched final holdout. Keep tuning on validation data and, where '
            'possible, reserve a separate unseen test set for final claims. '
            'mAP measures accuracy; detection counts and confidence do not.')
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)

    def choose_dataset(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Choose data.yaml', '', 'YAML (*.yaml *.yml)')
        if filename:
            self.dataset_path.setText(filename)

    def choose_model(self, target):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Choose best.pt', '', 'YOLO weights (*.pt)')
        if filename:
            target.setText(filename)

    def start(self):
        if self.worker is not None and self.worker.isRunning():
            return
        dataset = self.dataset_path.text().strip()
        old = self.old_path.text().strip()
        new = self.new_path.text().strip()
        if not Path(dataset).is_file() or not Path(old).is_file() or not Path(new).is_file():
            QMessageBox.warning(self, 'Missing files',
                                'Choose your real data.yaml and BOTH best.pt files.')
            return
        if Path(old).resolve() == Path(new).resolve():
            QMessageBox.warning(self, 'Same model',
                                'Old and improved models must be different files.')
            return
        self.records = []
        self.table.clearContents()
        self.export_button.setEnabled(False)
        self.summary_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status.setText('Preparing experiment data...')
        self.worker = FinalExperimentWorker(dataset, old, new,
                                            self.split.currentText(), self.settings, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.completed.connect(self.finished_experiment)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(lambda: self.run_button.setEnabled(True))
        self.worker.start()

    def finished_experiment(self, rows):
        self.records = rows
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row['model'], row['method'],
                      f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                      f"{row['map50']:.4f}", f"{row['map5095']:.4f}"]
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(value))
        self.status.setText('All 8 evaluations completed. Export both CSV files.')
        self.export_button.setEnabled(True)
        self.summary_button.setEnabled(True)

    def show_error(self, detail):
        self.status.setText('Experiment failed. See error details.')
        QMessageBox.critical(self, 'Step 16 experiment error', detail)

    def export_csv(self):
        if not self.records:
            return
        name, _ = QFileDialog.getSaveFileName(
            self, 'Save full research results', 'step16_final_results.csv',
            'CSV (*.csv)')
        if not name:
            return
        try:
            with open(name, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
            self.status.setText(f'Full results saved: {name}')
        except OSError as error:
            QMessageBox.critical(self, 'CSV export error', str(error))

    def export_summary(self):
        if not self.records:
            return
        name, _ = QFileDialog.getSaveFileName(
            self, 'Save comparison summary', 'step16_comparison_summary.csv',
            'CSV (*.csv)')
        if not name:
            return
        try:
            by_model = {}
            for row in self.records:
                by_model.setdefault(row['model'], {})[row['method']] = row
            summary = []
            for model, methods in by_model.items():
                baseline = methods['Original (No Enhancement)']
                for method, row in methods.items():
                    summary.append({
                        'model': model, 'method': method,
                        'map50': row['map50'], 'map5095': row['map5095'],
                        'map50_delta_from_original': row['map50'] - baseline['map50'],
                        'map5095_delta_from_original': row['map5095'] - baseline['map5095'],
                    })
            with open(name, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            self.status.setText(f'Summary saved: {name}')
        except Exception as error:
            QMessageBox.critical(self, 'Summary export error', str(error))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, 'Experiment running',
                'Please finish the experiment before closing this window.')
            event.ignore()
            return
        super().closeEvent(event)



# ============================================================================
# Step 17: Research analysis of aggregate Step 16 results.
# Uses observed CSV values only; no significance tests from eight aggregate rows.
# ============================================================================

STEP17_METRICS = (
    ('precision', 'Precision'), ('recall', 'Recall'),
    ('map50', 'mAP@50'), ('map5095', 'mAP@50-95'),
)


def read_step16_results(path):
    """Validate the complete 2x4 experiment before generating any conclusions."""
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    required = {'model', 'method', 'precision', 'recall', 'map50', 'map5095',
                'images', 'split', 'dataset_yaml', 'weights', 'parameter',
                'imgsz', 'batch', 'conf', 'nms_iou'}
    if len(rows) != 8:
        raise ValueError(f'Expected exactly 8 experiment rows, found {len(rows)}. '
                         'Export the FULL Step 16 results CSV.')
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Wrong CSV format. Use Step 16: Export Final Results CSV, '
                         'not Export Comparison Summary CSV.')
    models = list(dict.fromkeys(r['model'].strip() for r in rows))
    if len(models) != 2 or any(not m for m in models):
        raise ValueError('The CSV must contain exactly two named models.')
    expected_methods = set(COMPARISON_METHODS)
    records = {}
    for row in rows:
        model, method = row['model'].strip(), row['method'].strip()
        if method not in expected_methods:
            raise ValueError(f'Unknown enhancement method: {method}')
        key = model, method
        if key in records:
            raise ValueError(f'Duplicate evaluation: {model}, {method}')
        parsed = dict(row)
        for metric, label in STEP17_METRICS:
            try:
                x = float(row[metric])
            except (TypeError, ValueError) as err:
                raise ValueError(f'Invalid {label} for {model} / {method}') from err
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f'{label} outside 0–1 range: {x}')
            parsed[metric] = x
        try:
            count = int(row['images'])
        except (TypeError, ValueError) as err:
            raise ValueError('Invalid image count.') from err
        if count < 1:
            raise ValueError('Image count must be positive.')
        parsed['images'] = count
        records[key] = parsed
    if set(records) != {(m, method) for m in models for method in COMPARISON_METHODS}:
        raise ValueError('Missing model/method combination in the Step 16 CSV.')
    # These properties need to be identical to support a controlled comparison.
    for field in ('images', 'split', 'dataset_yaml', 'imgsz', 'batch', 'conf', 'nms_iou'):
        values = {str(r[field]).strip() for r in records.values()}
        if len(values) != 1:
            raise ValueError(f'Inconsistent {field} across experiment rows: {values}')
    for method in COMPARISON_METHODS:
        parameter_values = {str(records[(m, method)]['parameter']).strip() for m in models}
        if len(parameter_values) != 1:
            raise ValueError(f'Enhancement parameter differs between models: {method}')
    weights_by_model = {}
    for model in models:
        weights = {records[(model, method)]['weights'].strip() for method in COMPARISON_METHODS}
        if len(weights) != 1:
            raise ValueError(f'Inconsistent model checkpoints for {model}')
        weights_by_model[model] = next(iter(weights))
    if weights_by_model[models[0]] == weights_by_model[models[1]]:
        raise ValueError('Both models refer to the same checkpoint.')
    return models, records


def build_step17_analysis(models, records):
    """Descriptive, paired-by-method comparisons; no unsupported inference."""
    old, new = models
    baseline = 'Original (No Enhancement)'
    data = []
    for method in COMPARISON_METHODS:
        a, b = records[(old, method)], records[(new, method)]
        item = {'method': method}
        for metric, _ in STEP17_METRICS:
            item[f'old_{metric}'] = a[metric]
            item[f'new_{metric}'] = b[metric]
            item[f'improvement_{metric}_pp'] = (b[metric] - a[metric]) * 100
            for label, row, model in [('old', a, old), ('new', b, new)]:
                item[f'{label}_{metric}_vs_original_pp'] = (
                    row[metric] - records[(model, baseline)][metric]) * 100
        data.append(item)
    return data


def step17_markdown_report(source, models, records, differences):
    old, new = models
    common = records[(old, COMPARISON_METHODS[0])]
    header = [
        '# Step 17: Descriptive Research Analysis', '',
        f'Source CSV: `{source}`', '',
        f'Comparison: **{old}** versus **{new}**', '',
        f"Dataset: `{common['dataset_yaml']}`  ",
        f"Dataset split: `{common['split']}`  ",
        f"Images per condition: {common['images']}  ",
        f"Image size: {common['imgsz']}; batch: {common['batch']}; "
        f"confidence: {common['conf']}; NMS IoU: {common['nms_iou']}", '',
        '## Model comparison (percentage-point differences)', '',
        '| Enhancement | Old mAP@50-95 | New mAP@50-95 | Difference (pp) |',
        '|---|---:|---:|---:|',
    ]
    for row in differences:
        header.append(f"| {row['method']} | {row['old_map5095']:.4f} | "
                      f"{row['new_map5095']:.4f} | {row['improvement_map5095_pp']:+.2f} |")
    header += ['', '## Enhancement compared with original images', '',
               '| Model | Enhancement | mAP@50-95 change (pp) |',
               '|---|---|---:|']
    for label, model in [('old', old), ('new', new)]:
        for row in differences[1:]:
            header.append(f"| {model} | {row['method']} | "
                          f"{row[f'{label}_map5095_vs_original_pp']:+.2f} |")
    baseline = differences[0]
    delta = baseline['improvement_map5095_pp']
    if delta > 0:
        interpretation = 'higher'
    elif delta < 0:
        interpretation = 'lower'
    else:
        interpretation = 'equal'
    header += ['', '## Interpretation', '',
               f'On the original images, the newer model had {interpretation} '
               f'mAP@50-95 by {abs(delta):.2f} percentage points compared with '
               'the earlier model. This is an observed difference on this split, '
               'not proof of improvement on future data.', '',
               '## Limitations and responsible reporting', '',
               '- These eight rows contain *aggregate* metrics only. They cannot '
               'support p-values, confidence intervals, or claims of statistical significance.',
               '- The training runs may differ in more than epoch count, so this '
               'comparison does not isolate the causal effect of training duration.',
               '- If the test images informed earlier decisions, this is not a '
               'fully untouched final test set. Use a truly unseen holdout for final claims.',
               '- Aquarium imagery is not representative of all open-water environments.',
               '- Report failures, class-wise performance and qualitative examples '
               'alongside these aggregate metrics.',
               '- For inferential statistics, retain image-level paired predictions '
               'or evaluation data and use a suitable image-level resampling design.',
               '', '## Reproducibility', '',
               f'- Earlier checkpoint: `{records[(old, COMPARISON_METHODS[0])]["weights"]}`',
               f'- Improved checkpoint: `{records[(new, COMPARISON_METHODS[0])]["weights"]}`',
               '- Enhance the same images with the same parameters for each model.',
               '- Preserve original results CSV, dataset version and software versions.',
               '']
    return '\n'.join(header)


class Step17MetricChart(QWidget):
    """Two-model chart; painter is always closed to avoid the Qt Step 8 bug."""
    def __init__(self, rows, key='map5095', parent=None):
        super().__init__(parent)
        self.rows = rows
        self.key = key
        self.setMinimumHeight(300)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        try:
            self.draw_chart(painter, self.width(), self.height())
        finally:
            painter.end()

    def draw_chart(self, painter, width, height):
        painter.fillRect(0, 0, width, height, QColor('#ffffff'))
        painter.setPen(QColor('#173247'))
        painter.setFont(QFont('Arial', 10))
        painter.drawText(16, 25, 'mAP@50–95  |  Old model vs improved model  |  0–1 scale')
        left = 152
        available = max(90, width - left - 75)
        for i in range(5):
            x = left + round(available * i / 4)
            painter.setPen(QPen(QColor('#e3e9ed'), 1))
            painter.drawLine(x, 39, x, height - 32)
            painter.setPen(QColor('#445968'))
            painter.drawText(x - 10, height - 10, f'{i/4:.2f}')
        for i, row in enumerate(self.rows):
            y = 55 + i * 57
            painter.setPen(QColor('#173247'))
            painter.drawText(10, y + 15, row['method'].replace(' (No Enhancement)', ''))
            for j, (field, color) in enumerate((('old_map5095', '#176b87'),
                                                ('new_map5095', '#168579'))):
                value = row[field]
                painter.fillRect(left, y + j * 20, max(1, round(available * value)), 16,
                                 QColor(color))
                painter.setPen(QColor('#173247'))
                painter.drawText(min(width - 57, left + round(available * value) + 5),
                                 y + j * 20 + 13, f'{value:.3f}')
        painter.fillRect(17, height - 29, 12, 10, QColor('#176b87'))
        painter.fillRect(145, height - 29, 12, 10, QColor('#168579'))
        painter.drawText(34, height - 19, 'Old model')
        painter.drawText(162, height - 19, 'Improved model')


class Step17AnalysisDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 17 — Scientific Results Analysis')
        self.resize(1080, 790)
        self.models = None
        self.records = None
        self.differences = None
        self.source = None
        layout = QVBoxLayout(self)
        introductory = QLabel('Open the FULL Step 16 results CSV (8 rows). '
                             'Compare accuracy differences and export a thesis draft report. '
                             'This is descriptive analysis, NOT a statistical significance test.')
        introductory.setWordWrap(True)
        layout.addWidget(introductory)
        choose = QPushButton('Open Step 16 Final Results CSV')
        choose.clicked.connect(self.load_csv)
        layout.addWidget(choose)
        self.source_label = QLabel('No results loaded yet.')
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            'Method', 'Old mAP@50', 'New mAP@50', 'Change (pp)',
            'Old mAP@50-95', 'New mAP@50-95', 'Change (pp)',
            'New enhancement vs original (pp)'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        self.chart = Step17MetricChart([])
        layout.addWidget(self.chart)
        note = QLabel('CAUTION: Eight aggregate results are not enough to claim '
                      'statistical significance. This window only reports observed '
                      'percentage-point differences. No p-values are calculated.')
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QHBoxLayout()
        self.export_csv = QPushButton('Export Analysis CSV')
        self.export_csv.setEnabled(False)
        self.export_csv.clicked.connect(self.save_csv)
        controls.addWidget(self.export_csv)
        self.export_report = QPushButton('Export Thesis Report (.md)')
        self.export_report.setEnabled(False)
        self.export_report.clicked.connect(self.save_report)
        controls.addWidget(self.export_report)
        self.export_png = QPushButton('Save Comparison Graph PNG')
        self.export_png.setEnabled(False)
        self.export_png.clicked.connect(self.save_png)
        controls.addWidget(self.export_png)
        layout.addLayout(controls)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Step 16 full results CSV', '', 'CSV Files (*.csv)')
        if not path:
            return
        try:
            models, records = read_step16_results(path)
            differences = build_step17_analysis(models, records)
        except Exception as error:
            QMessageBox.critical(self, 'Cannot analyze CSV', str(error))
            return
        self.source = path
        self.models = models
        self.records = records
        self.differences = differences
        self.source_label.setText(f'Loaded: {Path(path).name}  |  '
                                  f'{models[0]} vs {models[1]}')
        self.table.setRowCount(len(differences))
        for i, row in enumerate(differences):
            fields = [row['method'], f"{row['old_map50']:.4f}",
                      f"{row['new_map50']:.4f}",
                      f"{row['improvement_map50_pp']:+.2f}",
                      f"{row['old_map5095']:.4f}",
                      f"{row['new_map5095']:.4f}",
                      f"{row['improvement_map5095_pp']:+.2f}",
                      f"{row['new_map5095_vs_original_pp']:+.2f}"]
            for j, text in enumerate(fields):
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.chart.rows = differences
        self.chart.update()
        for button in (self.export_csv, self.export_report, self.export_png):
            button.setEnabled(True)

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Step 17 analysis',
                                              'step17_analysis.csv', 'CSV Files (*.csv)')
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.differences[0]))
                writer.writeheader()
                writer.writerows(self.differences)
            QMessageBox.information(self, 'Saved', 'Analysis CSV exported.')
        except Exception as error:
            QMessageBox.critical(self, 'Export failed', str(error))

    def save_report(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save descriptive thesis draft',
                                              'step17_research_analysis.md',
                                              'Markdown (*.md)')
        if not path:
            return
        try:
            report = step17_markdown_report(self.source, self.models,
                                            self.records, self.differences)
            Path(path).write_text(report, encoding='utf-8')
            QMessageBox.information(self, 'Saved', 'Markdown report exported.')
        except Exception as error:
            QMessageBox.critical(self, 'Export failed', str(error))

    def save_png(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save analysis chart',
                                              'step17_model_comparison.png',
                                              'PNG Image (*.png)')
        if not path:
            return
        try:
            image = QImage(1050, 320, QImage.Format.Format_RGB32)
            if image.isNull():
                raise MemoryError('Could not allocate chart image.')
            image.fill(QColor('white'))
            painter = QPainter(image)
            if not painter.isActive():
                raise RuntimeError('Could not create image painter.')
            try:
                self.chart.draw_chart(painter, image.width(), image.height())
            finally:
                painter.end()
            if not image.save(path, 'PNG'):
                raise OSError('Qt failed to save the PNG file.')
            QMessageBox.information(self, 'Saved', 'Graph PNG exported.')
        except Exception as error:
            QMessageBox.critical(self, 'Chart export failed', str(error))


class UnderwaterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Underwater Vision AI")
        self.setWindowIcon(QApplication.instance().windowIcon())
        self.resize(1230, 820)

        self.original = None
        self.processed = None
        self.annotated = None
        self.model = None
        self.model_weights = "yolo26n.pt"
        self.method_values = {
            method: settings[2] for method, settings in METHOD_SETTINGS.items()
        }
        self.setup_ui()
        self.configure_shortcuts()

    def setup_ui(self):
        """Create the Ocean Studio workstation without changing research logic."""
        self.setWindowTitle("Underwater Vision AI  |  Ocean Studio")
        self.resize(1480, 920)
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(OCEAN_STYLESHEET)

        container = QWidget()
        self.setCentralWidget(container)
        host_layout = QVBoxLayout(container)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        self.background_label = QLabel(container)
        self.background_label.setScaledContents(False)
        self.background_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.background_label.lower()

        root = QWidget(container)
        root.setObjectName("root")
        self.content_root = root
        host_layout.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(23, 18, 23, 17)
        layout.setSpacing(16)

        # Brand header — balanced, spacious and deliberately low visual noise.
        heading = QHBoxLayout()
        heading.setSpacing(13)
        # No image in the upper-left corner. Keep the Ocean Studio title only.
        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        brand = QLabel("OCEAN STUDIO")
        brand.setObjectName("brandTitle")
        subtitle = QLabel("UNDERWATER VISION AI  /  COMPUTER VISION RESEARCH WORKSPACE")
        subtitle.setObjectName("subTitle")
        brand_text.addWidget(brand)
        brand_text.addWidget(subtitle)
        heading.addLayout(brand_text)
        heading.addStretch(1)
        self.logs_button = QPushButton("Open terminal / logs")
        self.logs_button.setToolTip("Open a live terminal showing Ocean Studio's application log")
        self.logs_button.clicked.connect(self.open_application_logs)
        heading.addWidget(self.logs_button)
        header_tag = QLabel("●  RESEARCH EDITION")
        header_tag.setObjectName("headerTag")
        heading.addWidget(header_tag)
        layout.addLayout(heading)

        body = QHBoxLayout()
        body.setSpacing(16)
        layout.addLayout(body, 1)

        # Left: controls live in an independent scroll area, preventing clipping
        # on smaller laptop screens.
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setObjectName("sidebarScroll")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFixedWidth(326)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 7, 0)
        side.setSpacing(15)
        sidebar_scroll.setWidget(sidebar)
        body.addWidget(sidebar_scroll)

        model_card, model_layout = self._make_card("01", "DETECTION MODEL", "Choose the brain behind the boxes")
        self.model_path_label = QLabel("yolo26n.pt  •  pretrained")
        self.model_path_label.setObjectName("modelName")
        self.model_path_label.setToolTip("Currently selected YOLO weights")
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.model_path_label)
        self.choose_model_button = QPushButton("Load my best.pt")
        self.choose_model_button.setObjectName("accentButton")
        self.choose_model_button.clicked.connect(self.choose_detection_model)
        model_layout.addWidget(self.choose_model_button)
        self.reset_model_button = QPushButton("Use pretrained YOLO")
        self.reset_model_button.clicked.connect(self.reset_detection_model)
        model_layout.addWidget(self.reset_model_button)
        self.model_classes_label = QLabel("Classes appear here when a model is loaded")
        self.model_classes_label.setObjectName("hintText")
        self.model_classes_label.setWordWrap(True)
        self.model_classes_label.setMaximumHeight(67)
        model_layout.addWidget(self.model_classes_label)
        self.test_detection_button = QPushButton("Test detection  ·  low confidence")
        self.test_detection_button.setToolTip(
            "Diagnostic: run the selected model on the original image with confidence 0.05."
        )
        self.test_detection_button.clicked.connect(self.test_detection)
        model_layout.addWidget(self.test_detection_button)
        side.addWidget(model_card)

        enhance_card, enhance_layout = self._make_card("02", "IMAGE ENHANCEMENT", "Tune underwater colors and contrast")
        method_title = QLabel("ENHANCEMENT METHOD")
        method_title.setObjectName("fieldLabel")
        enhance_layout.addWidget(method_title)
        self.enhancement_choice = QComboBox()
        self.enhancement_choice.addItems(list(METHOD_SETTINGS.keys()))
        self.enhancement_choice.currentTextChanged.connect(self.on_method_changed)
        enhance_layout.addWidget(self.enhancement_choice)
        self.setting_label = QLabel("CLAHE clip limit: 2.0")
        self.setting_label.setObjectName("valueText")
        enhance_layout.addWidget(self.setting_label)
        self.enhancement_slider = QSlider(Qt.Orientation.Horizontal)
        self.enhancement_slider.valueChanged.connect(self.on_enhancement_value_changed)
        enhance_layout.addWidget(self.enhancement_slider)
        self.reset_button = QPushButton("Reset enhancement setting")
        self.reset_button.clicked.connect(self.reset_method_setting)
        enhance_layout.addWidget(self.reset_button)
        self.enhance_button = QPushButton("Apply enhancement")
        self.enhance_button.setObjectName("accentButton")
        self.enhance_button.clicked.connect(self.enhance)
        enhance_layout.addWidget(self.enhance_button)
        side.addWidget(enhance_card)

        confidence_card, confidence_layout = self._make_card("03", "DETECTION SETTINGS", "Adjust sensitivity before detecting")
        self.confidence_label = QLabel("YOLO confidence: 0.25")
        self.confidence_label.setObjectName("valueText")
        confidence_layout.addWidget(self.confidence_label)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(5, 95)
        self.confidence_slider.setValue(25)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        confidence_layout.addWidget(self.confidence_slider)
        confidence_hint = QLabel("Lower values show more possible detections, including false positives.")
        confidence_hint.setObjectName("hintText")
        confidence_hint.setWordWrap(True)
        confidence_layout.addWidget(confidence_hint)
        side.addWidget(confidence_card)
        side.addStretch(1)

        # Right: large image-first canvas with a restrained action bar.
        workspace = QVBoxLayout()
        workspace.setSpacing(13)
        body.addLayout(workspace, 1)
        workspace_header = QHBoxLayout()
        workspace_heading = QVBoxLayout()
        workspace_heading.setSpacing(2)
        ws_title = QLabel("Image workspace")
        ws_title.setObjectName("sectionTitle")
        ws_desc = QLabel("Explore  /  Enhance  /  Detect  /  Compare")
        ws_desc.setObjectName("subTitle")
        workspace_heading.addWidget(ws_title)
        workspace_heading.addWidget(ws_desc)
        workspace_header.addLayout(workspace_heading)
        workspace_header.addStretch()
        # Independent workspace logo: does NOT reuse the personal logo.
        # To change this logo, replace assets/workspace_logo.png.
        self.workspace_logo = QLabel()
        self.workspace_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.workspace_logo.setFixedSize(200, 86)
        main_logo = logo_pixmap("workspace_logo.png", 194, 80)
        if main_logo is not None:
            self.workspace_logo.setPixmap(main_logo)
            self.workspace_logo.setToolTip("Workspace logo: assets/workspace_logo.png")
        else:
            self.workspace_logo.setText("ADD WORKSPACE LOGO")
            self.workspace_logo.setObjectName("workspaceLogoText")
            self.workspace_logo.setToolTip("Place your different logo at assets/workspace_logo.png")
        workspace_header.addWidget(self.workspace_logo)
        workspace.addLayout(workspace_header)

        actions = QFrame()
        actions.setObjectName("actionBar")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(8)
        self.open_button = QPushButton("＋  Open image")
        self.open_button.setObjectName("accentButton")
        self.open_button.clicked.connect(self.open_image)
        action_layout.addWidget(self.open_button)
        self.detect_button = QPushButton("◎  Detect objects")
        self.detect_button.setObjectName("detectButton")
        self.detect_button.clicked.connect(self.detect)
        action_layout.addWidget(self.detect_button)
        action_layout.addStretch(1)
        self.reset_image_button = QPushButton("↺  Reset")
        self.reset_image_button.clicked.connect(self.reset_image)
        self.reset_image_button.setToolTip("Restore the original image; keep selected model")
        action_layout.addWidget(self.reset_image_button)
        self.save_button = QPushButton("↓  Save result")
        self.save_button.clicked.connect(self.save_image)
        action_layout.addWidget(self.save_button)
        workspace.addWidget(actions)

        self.original_label = QLabel("OPEN AN IMAGE TO BEGIN")
        self.result_label = QLabel("YOUR RESULT WILL APPEAR HERE")
        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_splitter.setObjectName("imageSplitter")
        for label, title, caption in (
            (self.original_label, "SOURCE IMAGE", "Unmodified reference"),
            (self.result_label, "PROCESSED / AI RESULT", "Enhancement and predictions"),
        ):
            container = QFrame()
            container.setObjectName("imageCard")
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(13, 13, 13, 13)
            container_layout.setSpacing(9)
            title_row = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setObjectName("imageCardTitle")
            hint_label = QLabel(caption)
            hint_label.setObjectName("imageCardHint")
            title_row.addWidget(title_label)
            title_row.addStretch()
            title_row.addWidget(hint_label)
            container_layout.addLayout(title_row)
            label.setObjectName("imageCanvas")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(140, 210)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            label.setWordWrap(True)
            label.installEventFilter(self)
            container_layout.addWidget(label, 1)
            self.image_splitter.addWidget(container)
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.setStretchFactor(0, 1)
        self.image_splitter.setStretchFactor(1, 1)
        workspace.addWidget(self.image_splitter, 1)

        # All twelve existing research actions stay available. The tabs expose
        # them by task rather than by the step number where they were introduced.
        self.compare_button = QPushButton("Compare enhancements")
        self.compare_button.clicked.connect(self.compare_all)
        self.compare_yolo_button = QPushButton("Compare YOLO detections")
        self.compare_yolo_button.clicked.connect(self.compare_detections)
        self.batch_button = QPushButton("Batch YOLO analysis")
        self.batch_button.clicked.connect(self.compare_batch)
        self.report_button = QPushButton("Open batch report")
        self.report_button.clicked.connect(self.open_saved_batch_report)
        self.dataset_check_button = QPushButton("Check underwater dataset")
        self.dataset_check_button.clicked.connect(self.open_dataset_audit)
        self.evaluate_button = QPushButton("Evaluate labeled dataset")
        self.evaluate_button.clicked.connect(self.open_ground_truth_evaluation)
        self.train_button = QPushButton("Train underwater YOLO")
        self.train_button.clicked.connect(self.open_underwater_training)
        self.research_train_button = QPushButton("Improve model training")
        self.research_train_button.clicked.connect(self.open_research_training)
        self.accuracy_compare_button = QPushButton("Compare enhancement accuracy")
        self.accuracy_compare_button.clicked.connect(self.open_enhancement_accuracy)
        self.accuracy_report_button = QPushButton("Open accuracy graphs / CSV")
        self.accuracy_report_button.clicked.connect(self.open_accuracy_report)
        self.final_experiment_button = QPushButton("Final controlled experiment")
        self.final_experiment_button.clicked.connect(self.open_final_experiment)
        self.step17_analysis_button = QPushButton("Analyze final results")
        self.step17_analysis_button.clicked.connect(self.open_step17_analysis)

        research_heading = QHBoxLayout()
        research_label = QLabel("RESEARCH LAB")
        research_label.setObjectName("fieldLabel")
        research_heading.addWidget(research_label)
        research_heading.addStretch(1)
        research_subtitle = QLabel("12 connected tools  ·  All prior steps preserved")
        research_subtitle.setObjectName("hintText")
        research_heading.addWidget(research_subtitle)
        workspace.addLayout(research_heading)
        self.tool_tabs = QTabWidget()
        self.tool_tabs.setObjectName("researchTabs")
        groups = [
            ("Explore & batch", [self.compare_button, self.compare_yolo_button,
                                 self.batch_button, self.report_button]),
            ("Dataset & training", [self.dataset_check_button, self.evaluate_button,
                                    self.train_button, self.research_train_button]),
            ("Research & reports", [self.accuracy_compare_button, self.accuracy_report_button,
                                    self.final_experiment_button, self.step17_analysis_button]),
        ]
        for tab_name, buttons in groups:
            page = QWidget()
            page.setObjectName("researchPage")
            grid = QGridLayout(page)
            grid.setContentsMargins(14, 13, 14, 13)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(9)
            for index, button in enumerate(buttons):
                button.setMinimumHeight(39)
                grid.addWidget(button, index // 2, index % 2)
            self.tool_tabs.addTab(page, tab_name)
        # A discreet fourth tab for thesis author and university branding.
        self.tool_tabs.addTab(self.create_about_tab(), "About  ·  i")
        self.tool_tabs.setMinimumHeight(151)
        self.tool_tabs.setMaximumHeight(190)
        self.tool_tabs.currentChanged.connect(self.update_research_tab_height)
        workspace.addWidget(self.tool_tabs)

        footer = QFrame()
        footer.setObjectName("footer")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(12, 9, 12, 9)
        self.status_indicator = QLabel("●")
        self.status_indicator.setObjectName("statusLight")
        footer_row.addWidget(self.status_indicator)
        self.status = QLabel("Ready. Open an underwater image to begin.")
        self.status.setObjectName("statusText")
        self.status.setWordWrap(True)
        footer_row.addWidget(self.status, 1)
        help_button = QPushButton("?  Quick start")
        help_button.clicked.connect(self.show_quick_start)
        footer_row.addWidget(help_button)
        workspace.addWidget(footer)

        self._image_pixmaps = {}
        self.on_method_changed(self.enhancement_choice.currentText())

    def update_research_tab_height(self, index):
        """Reserve enough room for About without crowding normal research tabs."""
        if self.tool_tabs.tabText(index).startswith("About"):
            self.tool_tabs.setMinimumHeight(270)
            self.tool_tabs.setMaximumHeight(300)
        else:
            self.tool_tabs.setMinimumHeight(151)
            self.tool_tabs.setMaximumHeight(190)

    def create_about_tab(self):
        """Compact, legible author/contact details and two dedicated logo cards."""
        profile = load_profile()
        page = QWidget()
        page.setObjectName("researchPage")
        outer = QHBoxLayout(page)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(20)

        info_card = QFrame()
        info_card.setObjectName("aboutInfoCard")
        info_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 10, 16, 10)
        info_layout.setSpacing(5)
        heading = QLabel("ABOUT THE DEVELOPER")
        heading.setObjectName("aboutSectionLabel")
        info_layout.addWidget(heading)

        fields = (
            ("FIRST NAME", profile["first_name"]),
            ("SURNAME", profile["last_name"]),
            ("EMAIL", profile["email"] or "Not provided yet"),
            ("FACULTY NUMBER", profile["fnum"] or "Not provided yet"),
        )
        field_grid = QGridLayout()
        field_grid.setContentsMargins(0, 2, 0, 0)
        field_grid.setHorizontalSpacing(16)
        field_grid.setVerticalSpacing(8)
        for i, (title, value) in enumerate(fields):
            group = QWidget()
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(2)
            caption = QLabel(title)
            caption.setObjectName("aboutSectionLabel")
            answer = QLabel(value)
            answer.setObjectName("aboutValue")
            answer.setWordWrap(True)
            answer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            group_layout.addWidget(caption)
            group_layout.addWidget(answer)
            field_grid.addWidget(group, i // 2, i % 2)
        info_layout.addLayout(field_grid)
        footer = QLabel(f"{profile['program']}  ·  {profile['university']}")
        footer.setObjectName("aboutFooter")
        footer.setWordWrap(True)
        info_layout.addWidget(footer)
        info_layout.addStretch(1)
        outer.addWidget(info_card, 3)

        logo_wrap = QWidget()
        logo_layout = QHBoxLayout(logo_wrap)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(12)
        for filename, title in (("logo.png", ""),
                                ("university_logo.png", "")):
            card = QFrame()
            card.setObjectName("aboutLogoCard")
            card.setMinimumWidth(144)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(8)
            picture = QLabel()
            picture.setObjectName("aboutLogo")
            picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
            picture.setMinimumSize(116, 120)
            picture.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            pixmap = logo_pixmap(filename, 132, 115)
            if pixmap is not None:
                picture.setPixmap(pixmap)
            else:
                picture.setText("Add logo image\nto assets folder")
                picture.setToolTip(f"Place your image at assets/{filename}")
            label = QLabel(title)
            label.setObjectName("aboutSectionLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(picture, 1)
            card_layout.addWidget(label)
            logo_layout.addWidget(card)
        outer.addWidget(logo_wrap, 2)
        return page

    def _make_card(self, number, heading, hint):
        """Reusable, lightweight side card (no new package dependencies)."""
        card = QFrame()
        card.setObjectName("controlCard")
        column = QVBoxLayout(card)
        column.setContentsMargins(16, 15, 16, 15)
        column.setSpacing(10)
        row = QHBoxLayout()
        index = QLabel(number)
        index.setObjectName("cardIndex")
        row.addWidget(index)
        label = QLabel(heading)
        label.setObjectName("cardHeading")
        row.addWidget(label)
        row.addStretch()
        column.addLayout(row)
        subtitle = QLabel(hint)
        subtitle.setObjectName("hintText")
        subtitle.setWordWrap(True)
        column.addWidget(subtitle)
        return card, column

    def eventFilter(self, obj, event):
        """Re-scale the stored full-resolution preview when panels are resized."""
        if event.type() == QEvent.Type.Resize and obj in (self.original_label, self.result_label):
            self._update_preview(obj)
        return super().eventFilter(obj, event)

    def _update_preview(self, label):
        cache = getattr(self, "_image_pixmaps", None)
        if not cache:
            return
        pixmap = cache.get(label)
        if pixmap is None or pixmap.isNull():
            return
        available = label.contentsRect().size()
        if available.width() < 2 or available.height() < 2:
            return
        label.setPixmap(pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def refresh_background(self):
        """Redraw the optional blurred background to match the current window size."""
        if not hasattr(self, "background_label") or self.background_label is None:
            return
        central = self.centralWidget()
        if central is None:
            return
        self.background_label.setGeometry(0, 0, central.width(), central.height())
        pixmap = build_blurred_background_pixmap(central.width(), central.height())
        if pixmap is None:
            self.background_label.clear()
            self.background_label.hide()
        else:
            self.background_label.setPixmap(pixmap)
            self.background_label.show()
            self.background_label.lower()
        if hasattr(self, "content_root"):
            self.content_root.raise_()

    def open_application_logs(self):
        """Open the live log in a Windows terminal (or an in-app fallback)."""
        if sys.platform == "win32":
            try:
                # PowerShell follows new lines in the log while the app runs.
                # Use argument-list invocation, not shell=True, so spaces in
                # project paths (and user-selected folder names) are safe.
                escaped_path = str(LOG_FILE).replace("'", "''")
                command = (
                    "$Host.UI.RawUI.WindowTitle = 'Ocean Studio - Live Logs'; "
                    "Write-Host 'OCEAN STUDIO / LIVE LOGS' -ForegroundColor Cyan; "
                    f"Write-Host 'File: {escaped_path}' -ForegroundColor DarkGray; "
                    f"Get-Content -LiteralPath '{escaped_path}' -Tail 100 -Wait"
                )
                subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-NoLogo", "-Command", command],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    cwd=str(PROJECT_DIR),
                )
                self.status.setText("Live application logs opened in a separate terminal window.")
                return
            except (OSError, ValueError) as error:
                print(f"Could not open PowerShell terminal: {error}", file=sys.stderr)

        # Cross-platform fallback when PowerShell is unavailable.
        dialog = QDialog(self)
        dialog.setWindowTitle("Ocean Studio — Application Logs")
        dialog.resize(850, 520)
        column = QVBoxLayout(dialog)
        viewer = QPlainTextEdit(dialog)
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        column.addWidget(viewer)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        column.addWidget(button_box)

        def reload_log():
            try:
                content = LOG_FILE.read_text(encoding="utf-8", errors="replace")
                viewer.setPlainText(content[-100000:])
                scrollbar = viewer.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            except OSError as error:
                viewer.setPlainText(f"Unable to read log file: {error}")

        timer = QTimer(dialog)
        timer.timeout.connect(reload_log)
        timer.start(1000)
        reload_log()
        dialog.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_background()

    def configure_shortcuts(self):
        """Small, memorable keyboard shortcuts for a live demonstration."""
        for sequence, callback in (
            ("Ctrl+O", self.open_image),
            ("Ctrl+S", self.save_image),
            ("Ctrl+R", self.reset_image),
            ("Ctrl+D", self.detect),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)

    def reset_image(self):
        """Remove enhancement and bounding boxes without clearing the source."""
        if self.original is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        self.processed = self.original.copy()
        self.annotated = None
        self.show_image(self.original, self.original_label)
        self.show_image(self.processed, self.result_label)
        self.status.setText("Original image restored. Your selected YOLO model is unchanged.")

    def show_quick_start(self):
        QMessageBox.information(
            self, "Underwater Vision AI — Quick Start",
            "1. Open Image → choose a photo.\n"
            "2. Choose an enhancement and click Enhance.\n"
            "3. Choose your best.pt weights (optional).\n"
            "4. Click Detect Objects or Compare YOLO.\n"
            "5. Use Open terminal / logs to watch live application output.\n"
            "6. Use the tabs for dataset, training, experiments and reports.\n\n"
            "Shortcuts: Ctrl+O = Open | Ctrl+D = Detect | "
            "Ctrl+S = Save | Ctrl+R = Reset Image.\n\n"
            "For thesis results, use labeled test data and report mAP, "
            "precision and recall—not just the number of detected boxes."
        )

    def set_detection_model(self, weights):
        """Load and validate model before switching; retain old one on failure."""
        weights = str(weights)
        if weights != "yolo26n.pt" and not Path(weights).is_file():
            QMessageBox.warning(self, "Model not found", "Select a valid .pt model file.")
            return
        try:
            self.status.setText(f"Loading model: {Path(weights).name}...")
            QApplication.processEvents()
            candidate = YOLO(weights)
            # Load succeeds if this is an Ultralytics-compatible checkpoint.
            if getattr(candidate, "task", None) != "detect":
                raise ValueError("This model is not an object-detection model.")
            self.model_weights = weights
            self.model = candidate
            self.annotated = None
            if self.processed is not None:
                self.show_image(self.processed, self.result_label)
            title = Path(weights).name
            self.model_path_label.setText(title + ("  •  pretrained" if weights == "yolo26n.pt" else "  •  custom"))
            self.model_path_label.setToolTip(weights)
            names = getattr(candidate, "names", {})
            if isinstance(names, dict):
                class_labels = ", ".join(str(value) for value in names.values())
            else:
                class_labels = ", ".join(map(str, names))
            self.model_classes_label.setText("Classes: " + (class_labels or "unknown"))
            self.model_classes_label.setToolTip(class_labels or "unknown")
            if weights != "yolo26n.pt":
                self.confidence_slider.setValue(10)
            self.status.setText(f"Ready with {title}. Click Detect Objects or Compare YOLO.")
        except Exception as error:
            QMessageBox.critical(self, "Model loading error", str(error))
            self.status.setText("Model not changed; try another .pt file.")

    def choose_detection_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose your Step 11 best.pt model", "", "YOLO Weights (*.pt)"
        )
        if path:
            self.set_detection_model(path)

    def reset_detection_model(self):
        self.set_detection_model("yolo26n.pt")

    def open_enhancement_accuracy(self):
        dialog = EnhancementAccuracyDialog(
            self.method_values, self.model_weights, self)
        dialog.exec()

    def open_accuracy_report(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Step 13 accuracy CSV', '', 'CSV Files (*.csv)')
        if not filename:
            return
        try:
            with open(filename, 'r', encoding='utf-8-sig', newline='') as stream:
                records = list(csv.DictReader(stream))
            report = AccuracyReportDialog(records, Path(filename).name, self)
            report.exec()
        except Exception as error:
            QMessageBox.critical(self, 'Invalid accuracy CSV', str(error))

    def open_research_training(self):
        dialog = ResearchTrainingDialog(self)
        dialog.exec()

    def open_final_experiment(self):
        dialog = FinalExperimentDialog(self.method_values, self)
        dialog.exec()

    def open_underwater_training(self):
        dialog = UnderwaterTrainingDialog(self)
        dialog.exec()

    def open_dataset_audit(self):
        dialog = DatasetAuditDialog(self)
        dialog.exec()

    def open_ground_truth_evaluation(self):
        dialog = GroundTruthEvaluationDialog(self)
        dialog.exec()

    def setting_text(self, method, value):
        if method == "CLAHE":
            return f"CLAHE clip limit: {value / 10.0:.1f}"
        if method == "White Balance":
            return f"White balance strength: {value}%"
        if method == "Gamma Correction":
            return f"Gamma: {value / 100.0:.2f}"
        return "Original image (no setting)"

    def on_method_changed(self, method):
        minimum, maximum, _ = METHOD_SETTINGS[method]
        self.enhancement_slider.blockSignals(True)
        self.enhancement_slider.setRange(minimum, maximum)
        value = self.method_values[method]
        self.enhancement_slider.setValue(value)
        self.enhancement_slider.blockSignals(False)
        enabled = method != "Original (No Enhancement)"
        self.enhancement_slider.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.setting_label.setText(self.setting_text(method, value))
        self.status.setText("Choose a setting, then click Enhance to apply it.")

    def on_enhancement_value_changed(self, value):
        method = self.enhancement_choice.currentText()
        self.method_values[method] = value
        self.setting_label.setText(self.setting_text(method, value))
        self.status.setText("Setting changed. Click Enhance to update the image.")

    def reset_method_setting(self):
        method = self.enhancement_choice.currentText()
        default = METHOD_SETTINGS[method][2]
        self.method_values[method] = default
        self.enhancement_slider.setValue(default)
        self.setting_label.setText(self.setting_text(method, default))
        self.status.setText("Default restored. Click Enhance to apply it.")

    def on_confidence_changed(self, value):
        self.confidence_label.setText(f"YOLO confidence: {value / 100.0:.2f}")
        self.status.setText("Confidence changed. Click Detect Objects to apply it.")

    def show_image(self, image, label):
        """Preserve source-resolution pixmaps instead of repeatedly scaling them."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data, width, height, width * channels,
            QImage.Format.Format_RGB888,
        ).copy()
        if not hasattr(self, "_image_pixmaps"):
            self._image_pixmaps = {}
        self._image_pixmaps[label] = QPixmap.fromImage(qimage)
        self._update_preview(label)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Underwater Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Error", "Could not open image.")
            return
        self.original = image
        self.processed = image.copy()
        self.annotated = None
        self.show_image(self.original, self.original_label)
        self.show_image(self.processed, self.result_label)
        self.status.setText("Image loaded. Choose a method and click Enhance.")

    def enhance(self):
        if self.original is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        method = self.enhancement_choice.currentText()
        value = self.method_values[method]
        self.processed = apply_enhancement(self.original, method, value)
        self.annotated = None
        self.show_image(self.processed, self.result_label)
        self.status.setText(f"Applied: {self.setting_text(method, value)}.")

    def compare_all(self):
        """Show the original and all three enhancements from identical input."""
        if self.original is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        try:
            dialog = ComparisonDialog(self.original, self.method_values, self)
            dialog.exec()
            self.status.setText("Comparison completed. All methods used the same original image.")
        except Exception as error:
            QMessageBox.critical(self, "Comparison Error", str(error))

    def compare_detections(self):
        """Run YOLO on all four enhancement methods with identical settings."""
        if self.original is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        try:
            if self.model is None:
                self.status.setText("Loading YOLO model...")
                QApplication.processEvents()
                self.model = YOLO(self.model_weights)
            confidence = self.confidence_slider.value() / 100.0
            dialog = DetectionComparisonDialog(
                self.original, self.method_values, self.model, confidence, self
            )
            dialog.exec()
            self.status.setText("YOLO comparison window closed.")
        except Exception as error:
            QMessageBox.critical(self, "Comparison Error", str(error))

    def compare_batch(self):
        """Evaluate a folder with four methods; inference runs in a worker thread."""
        try:
            confidence = self.confidence_slider.value() / 100.0
            dialog = BatchComparisonDialog(self.method_values, confidence, self.model_weights, self)
            dialog.exec()
            self.status.setText("Batch comparison window closed.")
        except Exception as error:
            QMessageBox.critical(self, "Batch Comparison Error", str(error))

    def open_saved_batch_report(self):
        """Open a CSV exported by Step 7/8 without re-running YOLO."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Exported Batch CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                required = set(BATCH_FIELDS)
                if not required.issubset(set(reader.fieldnames or [])):
                    raise ValueError(
                        "This is not a batch-results CSV. Choose the file exported "
                        "using Export Batch CSV, not the summary CSV."
                    )
                records = list(reader)
            if not records:
                raise ValueError("The CSV contains no results.")
            for row in records:
                if row["method"] not in COMPARISON_METHODS:
                    raise ValueError("Unknown method found in the CSV.")
            dialog = BatchReportDialog(records, Path(path).name, self)
            dialog.exec()
        except Exception as error:
            QMessageBox.critical(self, "Open CSV Error", str(error))

    def test_detection(self):
        """Diagnostic on the untouched original image, at low confidence."""
        if self.original is None:
            QMessageBox.information(self, "First open an image", "Click Open Image first.")
            return
        self._run_detection(self.original, 0.05, diagnostic=True)

    def detect(self):
        if self.processed is None:
            QMessageBox.warning(self, "Warning", "Open an image first.")
            return
        confidence = self.confidence_slider.value() / 100.0
        self._run_detection(self.processed, confidence, diagnostic=False)

    def _run_detection(self, image, confidence, diagnostic=False):
        """Common inference path with useful zero-detection and error messages."""
        try:
            if self.model is None:
                self.status.setText("Loading YOLO model...")
                QApplication.processEvents()
                self.model = YOLO(self.model_weights)
                names = getattr(self.model, "names", {})
                class_labels = ", ".join(str(v) for v in names.values()) if isinstance(names, dict) else ", ".join(map(str, names))
                self.model_classes_label.setText("Classes: " + (class_labels or "unknown"))
                self.model_classes_label.setToolTip(class_labels or "unknown")
            self.status.setText("Running detection...")
            QApplication.processEvents()
            results = self.model.predict(source=image, conf=confidence, verbose=False)
            if not results:
                raise RuntimeError("YOLO returned no prediction result.")
            result = results[0]
            if result.boxes is None:
                raise RuntimeError("The loaded model did not return object detection boxes.")
            self.annotated = result.plot()
            self.show_image(self.annotated, self.result_label)
            count = len(result.boxes)
            class_counts = {}
            for cls_id in result.boxes.cls.tolist():
                label = result.names.get(int(cls_id), str(int(cls_id)))
                class_counts[label] = class_counts.get(label, 0) + 1
            detail = ", ".join(f"{name}: {n}" for name, n in sorted(class_counts.items()))
            model_name = Path(self.model_weights).name
            if count == 0:
                message = (
                    f"{model_name}: 0 detections at confidence {confidence:.2f}. "
                    "The model ran, but found no boxes above that threshold. "
                    "Try 'Test Detection (low confidence)' on an image containing a "
                    "class your model was trained to recognize."
                )
                self.status.setText(message)
                QMessageBox.information(self, "No objects found", message)
            else:
                mode = " [diagnostic / original image]" if diagnostic else ""
                self.status.setText(
                    f"{model_name}: {count} detections at confidence {confidence:.2f}"
                    + mode + (f" | {detail}" if detail else "")
                )
        except Exception as error:
            import traceback
            details = traceback.format_exc()
            self.status.setText("Detection failed. See error dialog.")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Detection Error")
            box.setText(f"Unable to run {Path(self.model_weights).name}: {error}")
            box.setInformativeText(
                "Click 'Show Details' to see the full error. "
                "You can also click 'Use Pretrained YOLO' to test the old model."
            )
            box.setDetailedText(details)
            box.exec()

    def open_step17_analysis(self):
        dialog = Step17AnalysisDialog(self)
        dialog.exec()

    def save_image(self):
        if self.processed is None:
            QMessageBox.warning(self, "Warning", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "underwater_result.png", "PNG Image (*.png)"
        )
        if not path:
            return
        image_to_save = (
            self.annotated if self.annotated is not None else self.processed
        )
        success, encoded = cv2.imencode(".png", image_to_save)
        if success:
            encoded.tofile(path)
            self.status.setText("Image saved.")
        else:
            QMessageBox.warning(self, "Error", "Saving failed.")


if __name__ == "__main__":
    # Give the app its own Windows taskbar identity instead of Python's default.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "OceanStudio.UnderwaterVisionAI.Research"
            )
        except Exception:
            pass
    initialize_logging()
    app = QApplication(sys.argv)
    set_ocean_app_icon(app)
    window = UnderwaterApp()
    window.show()
    sys.exit(app.exec())