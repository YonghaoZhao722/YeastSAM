import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame, QGridLayout
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YeastSAM Tools Launcher")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        self.resize(250, 250)

        def create_section(title, buttons):
            frame = QFrame()
            frame.setObjectName("sectionFrame")
            frame.setFrameStyle(QFrame.Box | QFrame.Raised)
            frame.setLineWidth(1)
            frame.setStyleSheet("""
                QFrame#sectionFrame {
                    border: 1px solid lightgray;
                    padding: 8px;
                }
            """)
            outer_layout = QVBoxLayout()

            title_layout = QHBoxLayout()
            label = QLabel(title)
            title_layout.addWidget(label, alignment=Qt.AlignLeft | Qt.AlignTop)
            title_layout.addStretch()
            outer_layout.addLayout(title_layout)

            button_layout = QVBoxLayout()
            button_layout.addStretch()
            for text, callback in buttons:
                btn = QPushButton(text)
                btn.setFixedWidth(200)  # Set consistent width for all buttons
                btn.clicked.connect(callback)
                button_layout.addWidget(btn, alignment=Qt.AlignHCenter)
            button_layout.addStretch()

            outer_layout.addLayout(button_layout)

            frame.setLayout(outer_layout)
            return frame

        # Section 1
        frame1 = create_section("1. Generate Masks", [
            ("napari", self.launch_napari)
        ])
        # Section 2
        frame2 = create_section("2. Optional Tools", [
            ("Shift Analyzer", self.launch_shift_analyzer),
            ("Apply Registration", self.launch_apply_registration)
        ])
        # Section 3
        frame3 = create_section("3. Convert to Outline File", [
            ("Mask2Outline", self.launch_mask2outline)
        ])
        # Section 4
        frame4 = create_section("4. Separation Module", [
            ("Mask Editor", self.launch_mask_editor)
        ])

        grid_layout.addWidget(frame1, 0, 0)
        grid_layout.addWidget(frame2, 0, 1)
        grid_layout.addWidget(frame3, 1, 0)
        grid_layout.addWidget(frame4, 1, 1)

        container = QWidget()
        container.setLayout(grid_layout)
        self.setCentralWidget(container)

    def launch_napari(self):
        import napari
        viewer = napari.Viewer() 


    def launch_shift_analyzer(self):
        subprocess.Popen([sys.executable, "tools/shift.py"])

    def launch_apply_registration(self):
        subprocess.Popen([sys.executable, "tools/registration.py"])

    def launch_mask2outline(self):
        subprocess.Popen([sys.executable, "tools/Mask2Outline.py"])

    def launch_mask_editor(self):
        subprocess.Popen([sys.executable, "tools/mask_editor.py"])

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
