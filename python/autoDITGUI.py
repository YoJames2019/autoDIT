import sys
import subprocess
import pkg_resources

required_packages = {'numpy', 'opencv-python', 'ultralytics', 'torch', 'pyqt6'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required_packages - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import cv2
from PyQt6.QtWidgets import (
  QApplication, 
  QWidget, 
  QLabel, 
  QPushButton,
  QFileDialog,
  QGridLayout,
  QListWidget,
  QLineEdit,
  QTextEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt, QPoint
from pathlib import Path
from functools import partial
from autoDIT import autoDIT


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        setupLayout.addWidget(QPushButton('Browse', clicked=partial(self.open_file_dialog, self.video_dir_view)), 0, 2)

        setupLayout.addWidget(QLabel("Film Audio Folder:"), 1, 0)
        setupLayout.addWidget(self.audio_dir_view, 1, 1)
        setupLayout.addWidget(QPushButton('Browse', clicked=partial(self.open_file_dialog, self.audio_dir_view)), 1, 2)

        setupLayout.addWidget(QLabel("AutoDIT Output Folder:"), 2, 0)
        setupLayout.addWidget(self.output_dir_view, 2, 1)
        setupLayout.addWidget(QPushButton('Browse', clicked=partial(self.open_file_dialog, self.output_dir_view)), 2, 2)


        self.run_button = QPushButton('Run AutoDIT', clicked=self.run_autoDIT)
        self.run_button.setEnabled(False)

        setupLayout.addWidget(self.run_button, 3, 0, 1, 3)


        previewLayout = QGridLayout()

        self.image_preview = QLabel(self)
        self.image_preview.setFixedSize(800, 800)
        self.image_preview.setScaledContents(True)
        pixmap = QPixmap("./assets/AutoDIT BG.png")
        self.image_preview.setPixmap(pixmap)

        previewLayout.addWidget(self.image_preview, 0, 0)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        previewLayout.addWidget(self.log_box, 0, 1)


        appLayout.addLayout(setupLayout, 0, 0)
        appLayout.addLayout(previewLayout, 1, 0)
        self.setLayout(appLayout)


        self.show()


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
        self.add_log(f"Audio Folder: {self.output_dir_view.text()}")
        self.add_log(f"Output Folder: {self.output_dir_view.text()}\n")

        self.run_button.setEnabled(False)
        DIT = autoDIT(self.set_image_preview, self.add_log, self.video_dir_view.text(), self.audio_dir_view.text(), self.output_dir_view.text())
        DIT.run() #some kind of arg here for the image preview and progress bar
        self.run_button.setEnabled(True)
        pixmap = QPixmap("./assets/AutoDIT BG.png")
        self.image_preview.setPixmap(pixmap)
        

    def add_log(self, text, indent=0):
        self.log_box.append(f"{'       ' * indent}{text}")
        QApplication.processEvents()


    def set_image_preview(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        bytes_per_line = 3 * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_preview.setPixmap(pixmap)
        QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
