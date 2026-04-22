import sys
import os

#sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
try:
    import PyQt5
    pyqt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    if os.path.isdir(pyqt_plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugins
except Exception:
    pass

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QUrl, QEvent, QFileInfo
# from PyQt5 import QtMultimedia
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from util.image_warp import crop2_169, resize_img

from VITON.viton_upperbody import FrameProcessor
from util.camera_util import list_available_cameras


BUILTIN_GARMENTS = [
    ("lab_03", "lab_03_vmsdp2ta"),
    ("lab_04", "lab_04_vmsdp2ta"),
    ("lab_07", "lab_07_vmsdp2ta"),
    ("jin_17", "jin_17_vmsdp2ta"),
    ("jin_18", "jin_18_vmsdp2ta"),
    ("jin_22", "jin_22_vmsdp2ta"),
]


def build_demo_garments(checkpoints_dir="./rtv_ckpts", garment_images_dir="./assets/garment_images"):
    builtins = []
    builtin_checkpoint_names = set()

    for thumbnail_name, checkpoint_name in BUILTIN_GARMENTS:
        builtin_checkpoint_names.add(checkpoint_name)
        builtins.append(
            {
                "display_name": thumbnail_name,
                "checkpoint_name": checkpoint_name,
                "thumbnail_path": os.path.join(garment_images_dir, f"{thumbnail_name}_white_bg.jpg"),
            }
        )

    custom_garments = []
    if os.path.isdir(checkpoints_dir):
        for checkpoint_name in sorted(os.listdir(checkpoints_dir)):
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
            if not os.path.isdir(checkpoint_path):
                continue
            if checkpoint_name in builtin_checkpoint_names:
                continue
            if not os.path.isfile(os.path.join(checkpoint_path, "latest_net_G.pth")):
                continue

            thumbnail_path = os.path.join(garment_images_dir, f"{checkpoint_name}_white_bg.jpg")
            custom_garments.append(
                {
                    "display_name": checkpoint_name,
                    "checkpoint_name": checkpoint_name,
                    "thumbnail_path": thumbnail_path if os.path.isfile(thumbnail_path) else None,
                }
            )

    return builtins + custom_garments


class VitonThread(QThread):
    frameCaptured = pyqtSignal(QImage)

    def __init__(self,garment_id_list):
        super().__init__()
        self.cap = self.get_camera()
        if not self.cap.isOpened():
            print("Failed to open the selected camera.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.frame_processor = FrameProcessor(garment_id_list)

    def set_taregt_id(self, id):
        print(id)
        self.frame_processor.set_target_garment(id)

    def get_camera(self):
        # Prefer the DroidCam virtual webcam when it is present.
        for source in ((2, cv2.CAP_V4L2), ("/dev/video2", cv2.CAP_V4L2), (1, cv2.CAP_V4L2), (0, cv2.CAP_V4L2)):
            cap = cv2.VideoCapture(*source)
            if cap.isOpened():
                print(f"Using camera source: {source[0]}")
                return cap
            cap.release()
        print("Falling back to default camera source 0")
        return cv2.VideoCapture(0, cv2.CAP_V4L2)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                # Keep the webcam's native orientation. The forced 90 degree
                # rotation made standard upright webcams appear sideways and
                # reduced usable framing before the 16:9 crop.
                frame=cv2.flip(frame, 1)
                frame=resize_img(frame,max_height=1024)
                frame=crop2_169(frame)
                frame = self.frame_processor(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.frameCaptured.emit(q_img)
            else:
                print("Camera opened but no frame was read from the selected source.")


    def stop(self):
        self.running = False
        self.cap.release()

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Try-On")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(200, 150)
        self.start_fullscreen = os.environ.get("RTV_FULLSCREEN", "").lower() in {"1", "true", "yes"}
        if self.start_fullscreen:
            self.setWindowFlag(Qt.FramelessWindowHint)
            self.showFullScreen()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)

        # Create a scroll area for the horizontal layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create a QWidget to hold the list
        container_list = QWidget()
        layout_list = QVBoxLayout(container_list)

        # Create a QListWidget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)  # Ensure only one item can be selected at a time
        self.list_widget.setIconSize(QSize(150,150))

        # Add none item
        item = QListWidgetItem()
        item.setIcon(QIcon('./assets/none.png'))
        item.setData(Qt.UserRole, -1)
        self.list_widget.addItem(item)

        demo_garments = build_demo_garments()
        garment_name_list = [garment["checkpoint_name"] for garment in demo_garments]

        for i, garment in enumerate(demo_garments):
            item = QListWidgetItem()
            if garment["thumbnail_path"] is not None:
                item.setIcon(QIcon(garment["thumbnail_path"]))
            else:
                item.setIcon(QIcon("./assets/none.png"))
                item.setText(garment["display_name"])
            item.setData(Qt.UserRole, i)
            self.list_widget.addItem(item)
        #for i in range(9):
        #    item = QListWidgetItem()
        #    item.setIcon(QIcon('./figures/female_garments/lab_'+str(i).zfill(2)+'_white_bg.jpg'))
        #    item.setData(Qt.UserRole, 16+i)
        #    self.list_widget.addItem(item)
        layout_list.addWidget(self.list_widget)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.installEventFilter(self)


        
        scroll_area.setWidget(container_list)

        layout.addWidget(scroll_area)
        scroll_area.setFixedWidth(230)

        #layout.setStretch(0, 9)
        #layout.setStretch(1, 2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.viton_thread = VitonThread(garment_name_list)
        self.viton_thread.frameCaptured.connect(self.update_image)
        self.viton_thread.start()

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress):
            # print(f'keypress obj: {obj}')
            if obj == self.list_widget:
                self.list_widget.setCurrentRow(event.key()-48)
                return True
        
        return super(CameraApp, self).eventFilter(obj, event)
            
        
                
    def keyPressEvent(self, e):
        print(f'key: {e.key()}')
        if e.key() == Qt.Key_0:
            self.list_widget.setCurrentRow(0)
        elif e.key() == Qt.Key_1:
            self.list_widget.setCurrentRow(1)
        elif e.key() == Qt.Key_2:
            self.list_widget.setCurrentRow(2)
        elif e.key() == Qt.Key_3:
             self.list_widget.setCurrentRow(3)
        elif e.key() == Qt.Key_4:
             self.list_widget.setCurrentRow(4)
        elif e.key() == Qt.Key_5:
             self.list_widget.setCurrentRow(5)
        elif e.key() == Qt.Key_Q or e.key() == Qt.Key_Escape:
            print("Exiting application...")
            self.close()  # Triggers closeEvent

    def on_selection_changed(self):
        print('selection changed')
        selected_items = self.list_widget.selectedItems() 
        ## TODO: test if it can keep playing ...
        if selected_items:
            selected_item = selected_items[0]
            item_id = selected_item.data(Qt.UserRole)  # Retrieve the ID from the item

            ## change garment
            self.viton_thread.set_taregt_id(item_id)
            
            #self.trigger_function(item_id)  # Pass the ID to the trigger function



    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.viton_thread.stop()
        self.viton_thread.wait()
        print("Viton thread stopped")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
