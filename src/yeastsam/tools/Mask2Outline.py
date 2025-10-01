import sys
import os
import numpy as np

# Version identifier for update system
VERSION = "1.1.0"

from skimage import measure
import tifffile as tiff
import json
from pathlib import Path
import threading
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QPushButton, 
                           QFileDialog, QSlider, QGroupBox, QGridLayout, QDialog, 
                           QLineEdit, QMessageBox, QRadioButton, QProgressDialog, QProgressBar,
                           QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# This function can be used by PyInstaller when bundling the application
# Use by adding this to the .spec file:
# from Mask2Outline import get_tcltk_files
# a.datas += get_tcltk_files()
def get_tcltk_files():
    """
    Returns a list of tuples for PyInstaller to include Tcl/Tk library files.
    Usage in .spec file:
    from Mask2Outline import get_tcltk_files
    a.datas += get_tcltk_files()
    """
    import os
    import tkinter
    import tcl
    import tkinter.filedialog
    
    tcl_files = []
    
    # Get Tcl/Tk library paths
    tcl_dir = os.path.dirname(tcl.__file__)
    tk_dir = os.path.join(os.path.dirname(tkinter.__file__), "tk")
    
    # Add Tcl files
    for root, dirs, files in os.walk(tcl_dir):
        for file in files:
            if file.endswith('.tcl') or file.endswith('.txt') or file.endswith('.msg'):
                full_file = os.path.join(root, file)
                rel_file = os.path.relpath(full_file, os.path.dirname(tcl_dir))
                tcl_files.append((os.path.join('tcl', rel_file), full_file, 'DATA'))
    
    # Add Tk files
    for root, dirs, files in os.walk(tk_dir):
        for file in files:
            if file.endswith('.tcl') or file.endswith('.txt') or file.endswith('.msg'):
                full_file = os.path.join(root, file)
                rel_file = os.path.relpath(full_file, os.path.dirname(tk_dir))
                tcl_files.append((os.path.join('tk', rel_file), full_file, 'DATA'))
    
    return tcl_files

# Fix Tcl/Tk initialization issues
def setup_tcl_tk_env():
    if getattr(sys, 'frozen', False):
        # First attempt - locate the bundled tcl/tk files
        bundle_dir = os.path.dirname(sys.executable)
        if sys.platform == 'darwin':  # macOS
            # For macOS app bundles, the structure is different
            # Check inside app bundle resources
            resources_dir = os.path.join(os.path.dirname(os.path.dirname(bundle_dir)), 'Resources')
            
            # Try to set environment variables for bundled tcl/tk
            if os.path.exists(os.path.join(resources_dir, 'tcl8.6')):
                os.environ['TCL_LIBRARY'] = os.path.join(resources_dir, 'tcl8.6')
                print(f"Set TCL_LIBRARY to: {os.environ['TCL_LIBRARY']}")
            
            if os.path.exists(os.path.join(resources_dir, 'tk8.6')):
                os.environ['TK_LIBRARY'] = os.path.join(resources_dir, 'tk8.6')
                print(f"Set TK_LIBRARY to: {os.environ['TK_LIBRARY']}")

# Try to set up Tcl/Tk environment
try:
    setup_tcl_tk_env()
except Exception as e:
    print(f"Error during Tcl/Tk setup: {e}")

# Import tkinter after setting environment variables
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"Error importing tkinter: {e}")
    # Show error message using system dialogs if tkinter fails
    error_msg = "Cannot initialize the application interface. Tcl/Tk libraries missing."
    try:
        if sys.platform == 'darwin':
            cmd = ["osascript", "-e", f'display dialog "{error_msg}" buttons {{"OK"}} default button "OK" with icon stop']
            subprocess.run(cmd)
        elif sys.platform == 'win32':
            cmd = ["powershell", "-Command", f"[System.Windows.Forms.MessageBox]::Show('{error_msg}', 'Application Error', 'OK', 'Error')"]
            subprocess.run(cmd)
    except:
        pass
    sys.exit(1)

class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_path, fish_image_path, output_path, process_mode, metadata=None):
        super().__init__()
        self.input_path = input_path
        self.fish_image_path = fish_image_path
        self.output_path = output_path
        self.process_mode = process_mode
        self.metadata = metadata or {
            'pix_xy': 160,
            'pix_z': 300,
            'ri': 1.518,
            'ex': 583,
            'em': 547,
            'na': 1.4,
            'type': 'widefield'
        }

    def run(self):
        try:
            if self.process_mode == "file":
                self.progress_updated.emit(10, "Processing file...")
                
                if not self.output_path:
                    self.output_path = os.path.splitext(self.input_path)[0] + '.txt'
                
                cy3_filename = os.path.basename(self.fish_image_path)
                self.convert_tiff_to_fishquant(self.input_path, cy3_filename, self.output_path, self.metadata)
                
                self.progress_updated.emit(100, f"Completed! Output saved to: {self.output_path}")
                self.processing_finished.emit(f"Completed! Output saved to: {self.output_path}")
            
            else:
                self.progress_updated.emit(0, "Processing folder...")
                input_dir = self.input_path
                cy3_dir = self.fish_image_path
                output_dir = self.output_path
                
                tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff')) and '_dic' in f.lower()]
                if not tiff_files:
                    self.error_occurred.emit("No TIFF mask files found in the selected folder.")
                    return
                
                processed_files = 0
                for i, filename in enumerate(tiff_files):
                    input_file = os.path.join(input_dir, filename)
                    output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
                    
                    cy3_filename = self.find_matching_cy3(filename, cy3_dir)
                    if not cy3_filename:
                        print(f"Warning: No matching CY3 file found for {filename} in {cy3_dir}")
                        continue
                    
                    progress = int((i / len(tiff_files)) * 100)
                    self.progress_updated.emit(progress, f"Processing {i+1}/{len(tiff_files)}: {filename}")
                    
                    try:
                        self.convert_tiff_to_fishquant(input_file, cy3_filename, output_file, self.metadata)
                        processed_files += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                
                self.progress_updated.emit(100, f"Completed! Processed {processed_files}/{len(tiff_files)} files.")
                self.processing_finished.emit(f"Completed! Processed {processed_files}/{len(tiff_files)} files.")
        
        except Exception as e:
            self.error_occurred.emit(f"An error occurred during processing:\n{str(e)}")

    def find_matching_cy3(self, mask_filename, cy3_folder):
        """Find matching CY3 file based on mask filename in the CY3 folder"""
        mask_filename_lower = mask_filename.lower()
        
        # Extract numeric ID (e.g., "1" from "1_DIC_s1")
        numeric_id_match = re.search(r'_(\d+)_dic_', mask_filename_lower)
        if not numeric_id_match:
            numeric_id_match = re.search(r'(\d+)_dic_', mask_filename_lower)
            
        # Extract sequence ID (e.g., "s1" from "1_DIC_s1")
        sequence_id_match = re.search(r'_s(\d+)', mask_filename_lower)
        
        if numeric_id_match and sequence_id_match:
            numeric_id = numeric_id_match.group(1)
            sequence_id = sequence_id_match.group(1)
            
            print(f"Looking for match with numeric ID: {numeric_id}, sequence ID: s{sequence_id}")
            
            # Look for CY3 files with both the numeric ID and sequence ID
            for file in os.listdir(cy3_folder):
                file_lower = file.lower()
                if file_lower.endswith(('.tif', '.tiff')):
                    # Check if file contains w[number][identifier] pattern (e.g., w2CY3, w1CY5, w3DAPI)
                    w_pattern = re.search(r'w(\d+)([a-zA-Z]+\d*)', file_lower)
                    has_marker = w_pattern is not None
                    
                    # Extract indicator information if pattern is found
                    if has_marker:
                        indicator_index = w_pattern.group(1)  # The integer part
                        indicator_view = w_pattern.group(2)   # The identifier part (e.g., cy3, cy5, dapi)
                        print(f"Found marker: w{indicator_index}{indicator_view} in {file}")
                    
                    # Check if file has the same numeric ID and sequence ID
                    has_numeric_id = f"_{numeric_id}_" in file_lower or file_lower.startswith(f"{numeric_id}_")
                    # Use regex for exact sequence ID matching to avoid s1 matching s10, s11, etc.
                    sequence_pattern = rf"_s{sequence_id}(?:[._]|$)"
                    has_sequence_id = re.search(sequence_pattern, file_lower) is not None
                    
                    print(f"  Testing file: {file}")
                    print(f"    has_marker: {has_marker}, has_numeric_id: {has_numeric_id}, has_sequence_id: {has_sequence_id}")
                    
                    if has_marker and has_numeric_id and has_sequence_id:
                        print(f"Match found: {file}")
                        return file
            
            print(f"No CY3 match found for {mask_filename}")
            return None
        else:
            print(f"Could not extract numeric ID or sequence ID from {mask_filename}")
            
            # Fallback method - extract base name and try to match
            mask_base = os.path.splitext(mask_filename)[0].replace('_DIC', '', 1).lower()
            print(f"Using fallback method with mask base: {mask_base}")
            
            for file in os.listdir(cy3_folder):
                if file.lower().endswith(('.tif', '.tiff')):
                    cy3_base = os.path.splitext(file)[0].lower()
                    cy3_parts = cy3_base.split('_')
                    mask_parts = mask_base.split('_')
                    
                    # Check if major parts of the filenames match
                    if len(mask_parts) > 2 and len(cy3_parts) > 2:
                        # Compare the main parts of the filename
                        common_prefix = '_'.join(mask_parts[:2])
                        # Check for w[number][identifier] pattern in cy3 filename
                        w_pattern = re.search(r'w(\d+)([a-zA-Z]+\d*)', cy3_base)
                        if common_prefix in cy3_base and w_pattern:
                            print(f"Fallback match found: {file}")
                            return file
            
            return None

    def convert_tiff_to_fishquant(self, input_file, cy3_filename, output_file, metadata):
        mask = tiff.imread(input_file)
        
        cell_labels = np.unique(mask)
        cell_labels = cell_labels[cell_labels != 0]
        
        cell_contours = []
        for label in cell_labels:
            binary_mask = (mask == label).astype(np.uint8)
            contours = measure.find_contours(binary_mask, level=0.5)
            if contours:
                contour = max(contours, key=len)
                min_y = np.min(contour[:, 0])
                cell_contours.append((label, contour, min_y))
        
        cell_contours = sorted(cell_contours, key=lambda x: x[2])
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w') as fid:
            fid.write('FISH-QUANT\tv2a\n')
            from datetime import datetime
            current_date = datetime.now().strftime("%d-%b-%Y")
            fid.write(f'RESULTS OF SPOT DETECTION PERFORMED ON {current_date}\n')
            fid.write('COMMENT\t\n')
            fid.write(f'IMG_Raw\t{cy3_filename}\n')
            fid.write('IMG_Filtered\t\n')
            fid.write('IMG_DAPI\t\n')
            fid.write('IMG_TS_label\t\n')
            fid.write('FILE_settings\t\n')
            fid.write('PARAMETERS\n')
            fid.write('Pix-XY\tPix-Z\tRI\tEx\tEm\tNA\tType\n')
            fid.write(f'{metadata["pix_xy"]}\t{metadata["pix_z"]}\t{metadata["ri"]}\t{metadata["ex"]}\t{metadata["em"]}\t{metadata["na"]}\t{metadata["type"]}\n')
            
            for i, (label, contour, min_y) in enumerate(cell_contours, 1):
                y_coords = contour[:, 0]
                x_coords = contour[:, 1]
                
                fid.write(f'CELL\tCell_{i}\n')
                fid.write('X_POS')
                for x in x_coords:
                    fid.write(f'\t{int(x)}')
                fid.write('\tEND\n')
                fid.write('Y_POS')
                for y in y_coords:
                    fid.write(f'\t{int(y)}')
                fid.write('\tEND\n')
                fid.write('Z_POS\t\n')

class Mask2OutlineTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF to FISH-QUANT Converter")
        self.setGeometry(100, 100, 600, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # Set initially on top
        
        # Variables
        self.input_path = ""
        self.fish_image_path = ""
        self.output_path = ""
        self.process_mode = "file"
        self.last_input_dir = os.path.expanduser("~")
        self.last_output_dir = os.path.expanduser("~")
        self.last_fish_dir = os.path.expanduser("~")
        
        # Default metadata values
        self.metadata = {
            'pix_xy': 160,
            'pix_z': 300,
            'ri': 1.518,
            'ex': 583,
            'em': 547,
            'na': 1.4,
            'type': 'widefield'
        }
        
        self.load_settings()
        self.initUI()
    
    def showEvent(self, event):
        super().showEvent(event)
        # After showing, remove the always-on-top flag
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()
    
    def load_settings(self):
        self.settings_file = os.path.join(os.path.expanduser("~"), ".tiff_converter_settings.json")
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.last_input_dir = settings.get('last_input_dir', os.path.expanduser("~"))
                    self.last_output_dir = settings.get('last_output_dir', os.path.expanduser("~"))
                    self.last_fish_dir = settings.get('last_fish_dir', os.path.expanduser("~"))
                    self.metadata.update(settings.get('metadata', {}))
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_settings(self):
        try:
            settings = {
                'last_input_dir': self.last_input_dir,
                'last_output_dir': self.last_output_dir,
                'last_fish_dir': self.last_fish_dir,
                'metadata': self.metadata
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Processing mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout()
        self.file_mode = QRadioButton("Process Single File")
        self.folder_mode = QRadioButton("Process Folder")
        self.file_mode.setChecked(True)
        self.file_mode.toggled.connect(self.update_mode)
        mode_layout.addWidget(self.file_mode)
        mode_layout.addWidget(self.folder_mode)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QGridLayout()
        
        # TIFF Mask input
        input_layout.addWidget(QLabel("TIFF Mask:"), 0, 0)
        self.input_entry = QLineEdit()
        input_layout.addWidget(self.input_entry, 0, 1)
        input_btn = QPushButton("Browse...")
        input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(input_btn, 0, 2, 1, 2)
        
        # Image input
        input_layout.addWidget(QLabel("FISH Image/Folder:"), 1, 0)
        self.fish_entry = QLineEdit()
        input_layout.addWidget(self.fish_entry, 1, 1)
        fish_btn = QPushButton("Browse...")
        fish_btn.clicked.connect(self.browse_fish_image)
        input_layout.addWidget(fish_btn, 1, 2, 1, 2)
        
        # Add separator line
        input_layout.addWidget(QLabel(""), 2, 0, 1, 4)
        
        # Pixel size XY
        input_layout.addWidget(QLabel("Pixel Size XY (nm):"), 3, 0)
        self.pix_xy_spinbox = QDoubleSpinBox()
        self.pix_xy_spinbox.setRange(0.1, 10000.0)
        self.pix_xy_spinbox.setDecimals(2)
        self.pix_xy_spinbox.setSingleStep(0.1)
        self.pix_xy_spinbox.setValue(self.metadata['pix_xy'])
        self.pix_xy_spinbox.valueChanged.connect(lambda v: self.metadata.update({'pix_xy': v}))
        input_layout.addWidget(self.pix_xy_spinbox, 3, 1)
        
        # Pixel size Z
        input_layout.addWidget(QLabel("Pixel Size Z (nm):"), 3, 2)
        self.pix_z_spinbox = QDoubleSpinBox()
        self.pix_z_spinbox.setRange(0.1, 10000.0)
        self.pix_z_spinbox.setDecimals(2)
        self.pix_z_spinbox.setSingleStep(0.1)
        self.pix_z_spinbox.setValue(self.metadata['pix_z'])
        self.pix_z_spinbox.valueChanged.connect(lambda v: self.metadata.update({'pix_z': v}))
        input_layout.addWidget(self.pix_z_spinbox, 3, 3)
        
        # Refractive Index
        input_layout.addWidget(QLabel("Refractive Index:"), 4, 0)
        self.ri_spinbox = QDoubleSpinBox()
        self.ri_spinbox.setRange(1.0, 2.0)
        self.ri_spinbox.setDecimals(3)
        self.ri_spinbox.setSingleStep(0.001)
        self.ri_spinbox.setValue(self.metadata['ri'])
        self.ri_spinbox.valueChanged.connect(lambda v: self.metadata.update({'ri': v}))
        input_layout.addWidget(self.ri_spinbox, 4, 1)
        
        # Numerical Aperture
        input_layout.addWidget(QLabel("Numerical Aperture:"), 4, 2)
        self.na_spinbox = QDoubleSpinBox()
        self.na_spinbox.setRange(0.1, 2.0)
        self.na_spinbox.setDecimals(2)
        self.na_spinbox.setSingleStep(0.01)
        self.na_spinbox.setValue(self.metadata['na'])
        self.na_spinbox.valueChanged.connect(lambda v: self.metadata.update({'na': v}))
        input_layout.addWidget(self.na_spinbox, 4, 3)
        
        # Excitation wavelength
        input_layout.addWidget(QLabel("Excitation (nm):"), 5, 0)
        self.ex_spinbox = QSpinBox()
        self.ex_spinbox.setRange(300, 800)
        self.ex_spinbox.setValue(self.metadata['ex'])
        self.ex_spinbox.valueChanged.connect(lambda v: self.metadata.update({'ex': v}))
        input_layout.addWidget(self.ex_spinbox, 5, 1)
        
        # Emission wavelength
        input_layout.addWidget(QLabel("Emission (nm):"), 5, 2)
        self.em_spinbox = QSpinBox()
        self.em_spinbox.setRange(300, 800)
        self.em_spinbox.setValue(self.metadata['em'])
        self.em_spinbox.valueChanged.connect(lambda v: self.metadata.update({'em': v}))
        input_layout.addWidget(self.em_spinbox, 5, 3)
        
        # Microscope - changed to text input
        input_layout.addWidget(QLabel("Microscope:"), 6, 0)
        self.type_entry = QLineEdit()
        self.type_entry.setText(self.metadata.get('type', 'widefield'))
        self.type_entry.textChanged.connect(lambda v: self.metadata.update({'type': v}))
        input_layout.addWidget(self.type_entry, 6, 1, 1, 2)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Output section
        output_group = QGroupBox("Output")
        output_layout = QGridLayout()
        
        output_layout.addWidget(QLabel("Output Location:"), 0, 0)
        self.output_entry = QLineEdit()
        output_layout.addWidget(self.output_entry, 0, 1)
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_btn, 0, 2)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        

        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)
        
        # Convert button
        convert_btn = QPushButton("Convert")
        convert_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(convert_btn)
        
        # Instructions
        instructions = QLabel("Instructions: Select a TIFF mask and FISH image (file mode) or folder (folder mode).\n"
                            "In folder mode, FISH files will be matched from the FISH folder.")
        instructions.setStyleSheet("color: gray;")
        main_layout.addWidget(instructions)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def update_mode(self):
        self.process_mode = "file" if self.file_mode.isChecked() else "folder"
    
    def browse_input(self):
        if self.process_mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Select TIFF Mask File", 
                                                self.last_input_dir,
                                                "TIFF files (*.tif *.tiff);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder Containing TIFF Masks",
                                                  self.last_input_dir)
        
        if path:
            self.input_path = path
            self.input_entry.setText(path)
            self.last_input_dir = os.path.dirname(path) if os.path.isfile(path) else path
            self.save_settings()
    
    def browse_fish_image(self):
        if self.process_mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Select CY3 Image File", 
                                                self.last_fish_dir,
                                                "TIFF files (*.tif *.tiff);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder Containing CY3 Images",
                                                  self.last_fish_dir)
        
        if path:
            self.fish_image_path = path
            self.fish_entry.setText(path)
            self.last_fish_dir = os.path.dirname(path) if os.path.isfile(path) else path
            self.save_settings()
    
    def browse_output(self):
        if self.process_mode == "file":
            default_filename = ""
            if self.input_path:
                default_filename = os.path.splitext(os.path.basename(self.input_path))[0] + ".txt"
            
            path, _ = QFileDialog.getSaveFileName(self, "Save FISH-QUANT Outline File",
                                                os.path.join(self.last_output_dir, default_filename),
                                                "Text files (*.txt);;All files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Output Folder for FISH-QUANT Files",
                                                  self.last_output_dir)
        
        if path:
            self.output_path = path
            self.output_entry.setText(path)
            self.last_output_dir = os.path.dirname(path) if os.path.isfile(path) else path
            self.save_settings()
    
    def start_processing(self):
        if not self.input_path:
            QMessageBox.critical(self, "Error", "Please select an input TIFF mask file or folder.")
            return
        if not self.fish_image_path:
            QMessageBox.critical(self, "Error", "Please select a CY3 image file or folder.")
            return
        if not self.output_path and self.process_mode == "folder":
            QMessageBox.critical(self, "Error", "Please select an output folder.")
            return
        
        # Save metadata settings before processing
        self.save_settings()
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(
            self.input_path,
            self.fish_image_path,
            self.output_path,
            self.process_mode,
            self.metadata
        )
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.processing_complete)
        self.processing_thread.error_occurred.connect(self.show_error)
        
        # Start processing
        self.processing_thread.start()
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def processing_complete(self, message):
        self.status_label.setText(message)
        QMessageBox.information(self, "Success", message)
    
    def show_error(self, message):
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", message)

def main():
    app = QApplication(sys.argv)
    window = Mask2OutlineTool()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
