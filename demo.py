import sys
import os

#sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QUrl, QEvent, QFileInfo
# from PyQt5 import QtMultimedia
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from util.image_warp import crop2_169, resize_img

from VITON.viton_fullbody_seq import FullBodySeqFrameProcessor

class VitonThread(QThread):
    frameCaptured = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.frame_processor = FullBodySeqFrameProcessor('coat_seq_vmssdp2ta_576')
        #self.frame_processor = FullBodyFrameProcessor('han_baseline_vmsdp2ta_576')
        self.use_vmssdp = False



    def run(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                ## ichao: remove flip (Nov 13, 2024)
                frame=cv2.flip(frame, 1)
                frame=resize_img(frame,max_height=1024)
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame=crop2_169(frame)

                frame = self.frame_processor.forward(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.frameCaptured.emit(q_img)


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

        # Create a QWidget to hold the list

        #layout.setStretch(0, 9)
        #layout.setStretch(1, 2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.viton_thread = VitonThread()
        self.viton_thread.frameCaptured.connect(self.update_image)
        self.viton_thread.start()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(event)


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
