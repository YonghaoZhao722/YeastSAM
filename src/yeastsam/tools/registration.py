import sys
import os
import numpy as np

# Version identifier for update system
VERSION = "1.0.0"

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QPushButton, 
                           QFileDialog, QSlider, QGroupBox, QGridLayout, QDialog, QLineEdit, QMessageBox, QRadioButton, QProgressDialog)
from PyQt5.QtCore import Qt
from skimage import io
import cv2

class RegistrationTool(QDialog):
    def __init__(self, parent=None, offset_x=0, offset_y=0):
        super().__init__(parent)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.mask_path = None
        self.output_path = None
        self.process_mode = "file"  # "file" or "folder"
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # Set initially on top
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Mask Registration Tool")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout()
        
        # Processing mode selection
        mode_group = QHBoxLayout()
        self.file_mode = QRadioButton("Single File")
        self.folder_mode = QRadioButton("Folder")
        self.file_mode.setChecked(True)
        self.file_mode.toggled.connect(self.update_mode)
        mode_group.addWidget(QLabel("Processing Mode:"))
        mode_group.addWidget(self.file_mode)
        mode_group.addWidget(self.folder_mode)
        layout.addLayout(mode_group)
        
        # File selection
        file_group = QHBoxLayout()
        self.mask_entry = QLineEdit()
        self.mask_entry.setPlaceholderText("Mask file path")
        mask_btn = QPushButton("Browse Mask")
        mask_btn.clicked.connect(self.browse_mask)
        file_group.addWidget(QLabel("Mask:"))
        file_group.addWidget(self.mask_entry)
        file_group.addWidget(mask_btn)
        layout.addLayout(file_group)
        
        # Output path selection
        output_group = QHBoxLayout()
        self.output_entry = QLineEdit()
        self.output_entry.setPlaceholderText("Output path")
        output_btn = QPushButton("Browse Output")
        output_btn.clicked.connect(self.browse_output)
        output_group.addWidget(QLabel("Output:"))
        output_group.addWidget(self.output_entry)
        output_group.addWidget(output_btn)
        layout.addLayout(output_group)
        
        # Offset controls
        offset_group = QHBoxLayout()
        self.x_offset = QLineEdit(str(self.offset_x))
        self.y_offset = QLineEdit(str(self.offset_y))
        offset_group.addWidget(QLabel("X Offset:"))
        offset_group.addWidget(self.x_offset)
        offset_group.addWidget(QLabel("Y Offset:"))
        offset_group.addWidget(self.y_offset)
        layout.addLayout(offset_group)
        
        # Apply button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply)
        layout.addWidget(apply_btn)
        
        self.setLayout(layout)
    
    def update_mode(self):
        if self.file_mode.isChecked():
            self.process_mode = "file"
            self.mask_entry.setPlaceholderText("Mask file path")
            self.output_entry.setPlaceholderText("Output file path")
        else:
            self.process_mode = "folder"
            self.mask_entry.setPlaceholderText("Mask folder path")
            self.output_entry.setPlaceholderText("Output folder path")
    
    def browse_mask(self):
        if self.process_mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Select Mask Image File", "", 
                                                "TIFF files (*.tif);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Mask Folder")
        
        if path:
            self.mask_path = path
            self.mask_entry.setText(path)
            
            # Check if it's a single file and verify it's a single layer
            if self.process_mode == "file" and os.path.isfile(path):
                try:
                    img = io.imread(path)
                    # Check if the image is a multi-layer stack
                    if len(img.shape) == 3 and img.shape[2] > 4:
                        QMessageBox.warning(self, "Warning", 
                                           f"The selected file appears to be a multi-layer stack with {img.shape[0]} layers.\n"
                                           "Please select a single-layer mask image for proper registration.")
                except Exception as e:
                    print(f"Error checking image: {str(e)}")
    
    def browse_output(self):
        if self.process_mode == "file":
            path, _ = QFileDialog.getSaveFileName(self, "Save Aligned Mask As", "", 
                                                "TIFF files (*.tif);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        
        if path:
            self.output_path = path
            self.output_entry.setText(path)
    
    def get_offset_values(self):
        try:
            x = float(self.x_offset.text())
            y = float(self.y_offset.text())
            return x, y
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid offset values. Please enter numbers.")
            return None, None
            
    def shift_without_warp(self, image, x_shift, y_shift):
        """
        Shift an image without using warpAffine, preserving exact pixel values
        """
        # Get dimensions
        if image.ndim == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            
        # Create output image of same shape and data type
        if channels == 1 and image.ndim == 2:
            result = np.zeros((height, width), dtype=image.dtype)
        else:
            result = np.zeros((height, width, channels), dtype=image.dtype)
            
        # Calculate source and destination regions
        x_shift = int(round(x_shift))
        y_shift = int(round(y_shift))
        
        # Source region in the original image
        src_x_start = max(0, -x_shift)
        src_y_start = max(0, -y_shift)
        src_x_end = min(width, width - x_shift)
        src_y_end = min(height, height - y_shift)
        
        # Destination region in the result image
        dst_x_start = max(0, x_shift)
        dst_y_start = max(0, y_shift)
        dst_x_end = min(width, width + x_shift)
        dst_y_end = min(height, height + y_shift)
        
        # Make sure the regions have the same size
        width_to_copy = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        height_to_copy = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        
        if width_to_copy <= 0 or height_to_copy <= 0:
            return result  # Nothing to copy
            
        src_x_end = src_x_start + width_to_copy
        src_y_end = src_y_start + height_to_copy
        dst_x_end = dst_x_start + width_to_copy
        dst_y_end = dst_y_start + height_to_copy
        
        # Copy the relevant region
        if channels == 1 and image.ndim == 2:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end]
        else:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end, :]
                
        return result
    
    def apply(self):
        if not self.mask_path:
            QMessageBox.critical(self, "Error", "Please select a mask file/folder.")
            return
        
        if not self.output_path:
            QMessageBox.critical(self, "Error", "Please select an output path.")
            return
        
        x, y = self.get_offset_values()
        if x is None or y is None:
            return
        
        # Additional check for multi-layer images in single file mode
        if self.process_mode == "file" and os.path.isfile(self.mask_path):
            try:
                mask_image = io.imread(self.mask_path)
                if len(mask_image.shape) == 3 and mask_image.shape[2] > 4:
                    QMessageBox.critical(self, "Error", 
                                       f"The selected file is a multi-layer stack with {mask_image.shape[0]} layers.\n"
                                       "Registration requires a single-layer mask image.\n"
                                       "Please use a different file or extract a single layer first.")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error validating image: {str(e)}")
                return
        
        # Round shifts to integers to prevent interpolation
        x_shift = int(round(x))
        y_shift = int(round(y))
        
        if self.process_mode == "file":
            # Process single file
            try:
                # Read image with scikit-image to preserve original format
                mask_image = io.imread(self.mask_path)
                
                # Shift the image using our custom function
                aligned_mask = self.shift_without_warp(mask_image, x_shift, y_shift)
                
                # Save using scikit-image to preserve format
                io.imsave(self.output_path, aligned_mask, check_contrast=False)
                
                QMessageBox.information(self, "Success", f"Aligned mask saved to: {self.output_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error processing file: {str(e)}")
                return
        else:
            # Process folder
            mask_files = [f for f in os.listdir(self.mask_path) 
                         if f.lower().endswith(('.tif', '.tiff', '.png', '.bmp'))]
            if not mask_files:
                QMessageBox.critical(self, "Error", "No image files found in the selected folder.")
                return
            
            # Create progress dialog
            progress = QProgressDialog("Processing files...", "Cancel", 0, len(mask_files), self)
            progress.setWindowModality(Qt.WindowModal)
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_path, exist_ok=True)
            
            processed_count = 0
            
            for i, mask_file in enumerate(mask_files):
                if progress.wasCanceled():
                    break
                
                progress.setLabelText(f"Processing {mask_file}...")
                progress.setValue(i)
                
                try:
                    # Process each file
                    mask_path = os.path.join(self.mask_path, mask_file)
                    output_path = os.path.join(self.output_path, f"{mask_file}")
                    
                    # Read image with scikit-image
                    mask_image = io.imread(mask_path)
                    
                    # Shift the image using our custom function
                    aligned_mask = self.shift_without_warp(mask_image, x_shift, y_shift)
                    
                    # Save using scikit-image to preserve format
                    io.imsave(output_path, aligned_mask, check_contrast=False)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {mask_file}: {str(e)}")
            
            progress.setValue(len(mask_files))
            QMessageBox.information(self, "Success", f"Processed {processed_count} of {len(mask_files)} images. All aligned masks saved to: {self.output_path}")
        
        # self.accept()

    def showEvent(self, event):
        super().showEvent(event)
        # After showing, remove the always-on-top flag
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

def main(offset_x=0, offset_y=0, parent=None):
    app = QApplication(sys.argv)
    window = RegistrationTool(parent, offset_x, offset_y)
    window.show()
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main()) 