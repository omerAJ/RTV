import argparse
import os
import sys

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
_PYQT_PLUGIN_ROOT = None
_PYQT_PLATFORM_ROOT = None
try:
    import PyQt5

    _PYQT_PLUGIN_ROOT = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    _PYQT_PLATFORM_ROOT = os.path.join(_PYQT_PLUGIN_ROOT, "platforms")
    if os.path.isdir(_PYQT_PLUGIN_ROOT):
        os.environ["QT_PLUGIN_PATH"] = _PYQT_PLUGIN_ROOT
    if os.path.isdir(_PYQT_PLATFORM_ROOT):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _PYQT_PLATFORM_ROOT
except Exception:
    pass

import cv2

if _PYQT_PLUGIN_ROOT and os.path.isdir(_PYQT_PLUGIN_ROOT):
    os.environ["QT_PLUGIN_PATH"] = _PYQT_PLUGIN_ROOT
if _PYQT_PLATFORM_ROOT and os.path.isdir(_PYQT_PLATFORM_ROOT):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _PYQT_PLATFORM_ROOT

from PyQt5.QtCore import QEvent, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from tie_tryon import ManualAdjustment, TieTryOnProcessor, load_tie_catalog
from util.image_warp import crop2_169, resize_img


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-first necktie try-on demo")
    parser.add_argument("--camera-id", type=int, default=None, help="Camera index to open")
    parser.add_argument(
        "--catalog",
        default="assets/ties/catalog.json",
        help="Path to the tie catalog JSON manifest",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Launch the window in fullscreen mode")
    parser.add_argument("--debug", action="store_true", help="Draw tracking diagnostics on top of the video")
    return parser


class TieTryOnThread(QThread):
    frame_captured = pyqtSignal(QImage)

    def __init__(self, camera_id, catalog_path, debug):
        super().__init__()
        self.running = True
        self.debug = debug
        self.catalog = load_tie_catalog(catalog_path)
        self.selected_tie_id = None
        self.manual_adjustments = {item.id: ManualAdjustment() for item in self.catalog.items}
        self.processor = TieTryOnProcessor(self.catalog)
        self.processor_closed = False
        self.cap = self._open_camera(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open a webcam. Pass --camera-id to choose a device.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def set_selected_tie(self, tie_id):
        self.selected_tie_id = tie_id

    def nudge(self, dx=0.0, dy=0.0, scale_delta=0.0, rotation_deg=0.0):
        if self.selected_tie_id is None:
            return
        adjustment = self.manual_adjustments[self.selected_tie_id]
        adjustment.offset_x += dx
        adjustment.offset_y += dy
        adjustment.scale_delta = max(-0.45, min(0.8, adjustment.scale_delta + scale_delta))
        adjustment.rotation_deg = max(-35.0, min(35.0, adjustment.rotation_deg + rotation_deg))

    def reset_adjustment(self):
        if self.selected_tie_id is None:
            return
        self.manual_adjustments[self.selected_tie_id] = ManualAdjustment()

    def get_adjustment(self, tie_id):
        if tie_id is None:
            return ManualAdjustment()
        return self.manual_adjustments[tie_id]

    def _open_camera(self, preferred_id):
        if preferred_id is not None:
            cap = cv2.VideoCapture(preferred_id, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
            cap = cv2.VideoCapture(preferred_id)
            if cap.isOpened():
                return cap
            cap.release()

        for source in ((2, cv2.CAP_V4L2), ("/dev/video2", cv2.CAP_V4L2), (1, cv2.CAP_V4L2), (0, cv2.CAP_V4L2)):
            cap = cv2.VideoCapture(*source)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(0)

    def run(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame = resize_img(frame, max_height=1024)
                frame = crop2_169(frame)
                rendered, _state = self.processor.process(
                    frame,
                    self.selected_tie_id,
                    self.get_adjustment(self.selected_tie_id),
                    debug=self.debug,
                )
                rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb.shape
                step = channel * width
                q_img = QImage(rgb.data, width, height, step, QImage.Format_RGB888)
                self.frame_captured.emit(q_img.copy())
        finally:
            self._close_processor()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def _close_processor(self):
        if not self.processor_closed:
            self.processor.close()
            self.processor_closed = True


class CameraApp(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Tie Try-On")
        self.setGeometry(100, 100, 960, 640)
        self.setMinimumSize(300, 200)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        container_list = QWidget()
        layout_list = QVBoxLayout(container_list)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setIconSize(QSize(150, 150))

        none_item = QListWidgetItem()
        none_item.setIcon(QIcon("./assets/none.png"))
        none_item.setText("No Tie")
        none_item.setData(Qt.UserRole, None)
        self.list_widget.addItem(none_item)

        self.tryon_thread = TieTryOnThread(
            camera_id=args.camera_id,
            catalog_path=args.catalog,
            debug=args.debug,
        )

        for item in self.tryon_thread.catalog.items:
            list_item = QListWidgetItem()
            list_item.setIcon(QIcon(str(item.thumbnail_path)))
            list_item.setText(item.name)
            list_item.setData(Qt.UserRole, item.id)
            self.list_widget.addItem(list_item)

        layout_list.addWidget(self.list_widget)
        scroll_area.setWidget(container_list)
        layout.addWidget(scroll_area)
        scroll_area.setFixedWidth(260)

        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.installEventFilter(self)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.tryon_thread.frame_captured.connect(self.update_image)
        self.tryon_thread.start()
        self.list_widget.setCurrentRow(0)

        if args.fullscreen:
            self.setWindowFlag(Qt.FramelessWindowHint)
            self.showFullScreen()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and obj == self.list_widget:
            row = self._row_for_key(event.key())
            if row is not None:
                self.list_widget.setCurrentRow(row)
                return True
        return super().eventFilter(obj, event)

    def _row_for_key(self, key):
        if Qt.Key_0 <= key <= Qt.Key_9:
            row = key - Qt.Key_0
            if row < self.list_widget.count():
                return row
        return None

    def keyPressEvent(self, event):
        row = self._row_for_key(event.key())
        if row is not None:
            self.list_widget.setCurrentRow(row)
            return
        if event.key() == Qt.Key_Left:
            self.tryon_thread.nudge(dx=-6.0)
        elif event.key() == Qt.Key_Right:
            self.tryon_thread.nudge(dx=6.0)
        elif event.key() == Qt.Key_Up:
            self.tryon_thread.nudge(dy=-6.0)
        elif event.key() == Qt.Key_Down:
            self.tryon_thread.nudge(dy=6.0)
        elif event.key() == Qt.Key_BracketLeft:
            self.tryon_thread.nudge(scale_delta=-0.03)
        elif event.key() == Qt.Key_BracketRight:
            self.tryon_thread.nudge(scale_delta=0.03)
        elif event.key() == Qt.Key_Comma:
            self.tryon_thread.nudge(rotation_deg=-1.5)
        elif event.key() == Qt.Key_Period:
            self.tryon_thread.nudge(rotation_deg=1.5)
        elif event.key() == Qt.Key_R:
            self.tryon_thread.reset_adjustment()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()

    def on_selection_changed(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        selected_item = selected_items[0]
        tie_id = selected_item.data(Qt.UserRole)
        self.tryon_thread.set_selected_tie(tie_id)

    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event):
        self.tryon_thread.stop()
        self.tryon_thread.wait()
        event.accept()


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv if argv is None else ["tie_demo.py", *argv])
    window = CameraApp(args)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
