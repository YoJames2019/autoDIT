import sys
import subprocess
import pkg_resources

required_packages = {'numpy', 'opencv-python', 'ultralytics', 'torch', 'pyqt6'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required_packages - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

from datetime import datetime
import cv2
import os
from PyQt6.QtWidgets import (
  QApplication, 
  QWidget, 
  QLabel, 
  QPushButton,
  QFileDialog,
  QGridLayout,
  QLineEdit,
  QTextEdit,
  QProgressBar,
  QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, QTimer, QThread, pyqtSignal
from pathlib import Path
from functools import partial
from autoDIT import autoDITWorker

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        self.setWindowTitle('AutoDIT')
        
        appLayout = QGridLayout()
        self.setGeometry(100, 100, 1100, 80)


        setupLayout = QGridLayout()

        self.video_dir_view = QLineEdit()
        self.video_dir_view.setReadOnly(True)
        self.video_dir_view.textChanged.connect(self.check_line_edits)

        self.audio_dir_view = QLineEdit()
        self.audio_dir_view.setReadOnly(True)
        self.audio_dir_view.textChanged.connect(self.check_line_edits)

        self.output_dir_view = QLineEdit()
        self.output_dir_view.setReadOnly(True)
        self.output_dir_view.textChanged.connect(self.check_line_edits)


        setupLayout.addWidget(QLabel("Film Video Folder:"), 0, 0)
        setupLayout.addWidget(self.video_dir_view, 0, 1)
        self.video_browse_button = QPushButton('Browse', clicked=partial(self.open_file_dialog, self.video_dir_view))
        setupLayout.addWidget(self.video_browse_button, 0, 2)

        setupLayout.addWidget(QLabel("Film Audio Folder:"), 1, 0)
        setupLayout.addWidget(self.audio_dir_view, 1, 1)
        self.audio_browse_button = QPushButton('Browse', clicked=partial(self.open_file_dialog, self.audio_dir_view))
        setupLayout.addWidget(self.audio_browse_button, 1, 2)

        setupLayout.addWidget(QLabel("AutoDIT Output Folder:"), 2, 0)
        setupLayout.addWidget(self.output_dir_view, 2, 1)
        self.output_browse_button = QPushButton('Browse', clicked=partial(self.open_file_dialog, self.output_dir_view))
        setupLayout.addWidget(self.output_browse_button, 2, 2)


        self.run_button = QPushButton('Run AutoDIT', clicked=self.run_autoDIT)
        self.run_button.setEnabled(False)

        setupLayout.addWidget(self.run_button, 3, 0, 1, 3)

        previewLayout = QGridLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        previewLayout.addWidget(self.progress_bar, 0, 0, 1, 2)

        self.image_preview = QLabel(self)
        self.image_preview.setFixedSize(800, 800)
        self.image_preview.setScaledContents(True)
        self.set_image_preview(os.path.join(self.src_dir, "assets/AutoDIT BG.png"))

        previewLayout.addWidget(self.image_preview, 1, 0)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        previewLayout.addWidget(self.log_box, 1, 1)


        appLayout.addLayout(setupLayout, 0, 0)
        appLayout.addLayout(previewLayout, 1, 0)
        self.setLayout(appLayout)

        self.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)



    @pyqtSlot()
    def check_line_edits(self):
        if all([self.video_dir_view.text(), self.audio_dir_view.text(), self.output_dir_view.text()]):
            self.run_button.setEnabled(True)
        else:
            self.run_button.setEnabled(False)


    def open_file_dialog(self, line_edit):
        filename = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            options=QFileDialog.Option.ShowDirsOnly
        )

        if filename:
            path = Path(filename)
            line_edit.setText(str(path))


    def run_autoDIT(self):
        self.add_log("Running AutoDIT")
        self.add_log(f"Video Folder: {self.video_dir_view.text()}")
        self.add_log(f"Audio Folder: {self.audio_dir_view.text()}")
        self.add_log(f"Output Folder: {self.output_dir_view.text()}\n")


        self.run_button.setEnabled(False)
        self.video_browse_button.setEnabled(False)
        self.audio_browse_button.setEnabled(False)
        self.output_browse_button.setEnabled(False)
        
        self.DITThread = autoDITWorker(self.video_dir_view.text(), self.audio_dir_view.text(), self.output_dir_view.text())
        self.DITThread.progress.connect(self.update_ui)
        self.DITThread.image_preview_signal.connect(self.set_image_preview)
        self.DITThread.log_signal.connect(self.add_log)
        self.DITThread.progress_bar_signal.connect(self.update_progress_bar)
        self.DITThread.progress_bar_range_signal.connect(self.set_progress_bar_range)
        self.DITThread.finished.connect(self.autoDIT_finished)
        self.DITThread.start()
        

    def add_log(self, text, indent=0, new_line=False):
        new_line = "\n" if new_line else ""
        self.log_box.append(f"{new_line}{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} - {'       ' * indent}{text}")
        self.update_ui()


    def set_image_preview(self, image):

        if type(image) is str:
            pixmap = QPixmap(image)
            self.image_preview.setPixmap(pixmap)
            return

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        bytes_per_line = 3 * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_preview.setPixmap(pixmap)
        self.update_ui()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(self.progress_bar.value() + value)

    def set_progress_bar_range(self, min, max):
        self.progress_bar.setRange(min, max)
        self.progress_bar.setValue(min)

    def update_ui(self):
        QApplication.processEvents()

    def autoDIT_finished(self):
        self.run_button.setEnabled(True)
        self.video_browse_button.setEnabled(True)
        self.audio_browse_button.setEnabled(True)
        self.output_browse_button.setEnabled(True)
        self.set_image_preview(os.path.join(self.src_dir, "assets/AutoDIT BG.png"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
