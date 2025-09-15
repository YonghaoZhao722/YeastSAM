import sys
import os
import numpy as np

# Version identifier for update system
VERSION = "1.1.2"

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QFileDialog, QSlider, QGroupBox, QGridLayout, QDialog, QLineEdit, QMessageBox, QRadioButton, QProgressDialog, QCheckBox)
from PyQt5.QtCore import Qt, QPoint, QTimer
import registration
import cv2
from skimage import exposure
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
import io
import re

class MaskVisualizationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mask_img = None
        self.mask_file_path = None
        self.bg_img = None
        self.colored_mask = None
        self.processed_bg = None
        self.offset_x = 0
        self.offset_y = 0
        self.last_pos = None
        self.drag_enabled = False
        # Removed colormap variables since we now use fixed 'viridis' colormap
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # Set initially on top
        
        # Batch processing variables
        self.mask_folder = ""
        self.fish_folder = ""
        self.output_folder = ""
        self.mask_files = []
        self.current_mask_index = 0
        self.is_batch_mode = False
        self.current_fish_file = ""
        
        # Use a more explicit approach to manage the plots
        self.bg_plot = None
        self.mask_plot = None
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Shift Analyzer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout (horizontal orientation)
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left sidebar for controls
        left_sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(5)  # Reduced spacing for tighter layout
        
        # File input controls group (unified for both modes)
        input_group = QGroupBox("File Input")
        input_layout = QVBoxLayout()
        
        # Processing mode selection
        mode_layout = QHBoxLayout()
        self.single_mode = QRadioButton("Single File")
        self.batch_mode = QRadioButton("Batch (Folders)")
        self.single_mode.setChecked(True)
        self.single_mode.toggled.connect(self.toggle_processing_mode)
        mode_layout.addWidget(self.single_mode)
        mode_layout.addWidget(self.batch_mode)
        input_layout.addLayout(mode_layout)
        
        # Single file mode controls
        self.single_controls = QWidget()
        single_layout = QVBoxLayout()
        single_layout.setContentsMargins(0, 0, 0, 0)
        
        # Mask file selection
        mask_file_layout = QHBoxLayout()
        mask_file_layout.addWidget(QLabel("Mask:"))
        self.mask_file_btn = QPushButton("Select")
        self.mask_file_btn.clicked.connect(self.select_mask_file)
        mask_file_layout.addWidget(self.mask_file_btn)
        single_layout.addLayout(mask_file_layout)
        
        self.mask_file_label = QLabel("No file selected")
        self.mask_file_label.setWordWrap(True)
        single_layout.addWidget(self.mask_file_label)
        
        self.single_controls.setLayout(single_layout)
        input_layout.addWidget(self.single_controls)
        
        # Batch mode controls
        self.batch_controls = QWidget()
        batch_layout = QVBoxLayout()
        batch_layout.setContentsMargins(0, 0, 0, 0)
        
        # Mask folder selection
        mask_folder_layout = QHBoxLayout()
        mask_folder_layout.addWidget(QLabel("Mask Folder:"))
        self.mask_folder_btn = QPushButton("Select")
        self.mask_folder_btn.clicked.connect(self.select_mask_folder)
        mask_folder_layout.addWidget(self.mask_folder_btn)
        batch_layout.addLayout(mask_folder_layout)
        
        self.mask_folder_label = QLabel("No mask folder selected")
        self.mask_folder_label.setWordWrap(True)
        batch_layout.addWidget(self.mask_folder_label)
        
        # FISH folder selection
        fish_folder_layout = QHBoxLayout()
        fish_folder_layout.addWidget(QLabel("FISH Folder:"))
        self.fish_folder_btn = QPushButton("Select")
        self.fish_folder_btn.clicked.connect(self.select_fish_folder)
        fish_folder_layout.addWidget(self.fish_folder_btn)
        batch_layout.addLayout(fish_folder_layout)
        
        self.fish_folder_label = QLabel("No FISH folder selected")
        self.fish_folder_label.setWordWrap(True)
        batch_layout.addWidget(self.fish_folder_label)
        
        # Output folder selection for auto-save
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_btn = QPushButton("Select")
        self.output_folder_btn.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(self.output_folder_btn)
        batch_layout.addLayout(output_folder_layout)
        
        self.output_folder_label = QLabel("No output folder selected")
        self.output_folder_label.setWordWrap(True)
        batch_layout.addWidget(self.output_folder_label)

        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.prev_mask)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_mask)
        nav_layout.addWidget(self.next_btn)
        batch_layout.addLayout(nav_layout)

        # Auto-save option
        self.auto_save_checkbox = QCheckBox("Auto-save when navigating")
        self.auto_save_checkbox.setChecked(True)  # Default enabled
        batch_layout.addWidget(self.auto_save_checkbox)
        
        self.batch_controls.setLayout(batch_layout)
        self.batch_controls.setVisible(False)  # Hidden initially
        input_layout.addWidget(self.batch_controls)
        
        input_group.setLayout(input_layout)
        sidebar_layout.addWidget(input_group)
        
        # Mask display controls group
        mask_display_group = QGroupBox("Display Controls")
        controls_layout = QVBoxLayout()
        
        # Mask opacity
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_display)
        opacity_layout.addWidget(self.opacity_slider)
        controls_layout.addLayout(opacity_layout)
        
        # Background file selection (single mode only)
        self.bg_file_controls = QWidget()
        bg_file_layout = QVBoxLayout()
        bg_file_layout.setContentsMargins(0, 0, 0, 0)
        
        bg_select_layout = QHBoxLayout()
        bg_select_layout.addWidget(QLabel("Background:"))
        self.bg_file_btn = QPushButton("Select")
        self.bg_file_btn.clicked.connect(self.select_bg_file)
        bg_select_layout.addWidget(self.bg_file_btn)
        bg_file_layout.addLayout(bg_select_layout)
        
        self.bg_file_label = QLabel("No file selected")
        self.bg_file_label.setWordWrap(True)
        bg_file_layout.addWidget(self.bg_file_label)
        
        self.bg_file_controls.setLayout(bg_file_layout)
        controls_layout.addWidget(self.bg_file_controls)
        
        # Brightness and contrast controls
        bg_controls_layout = QHBoxLayout()
        bg_select_layout.addWidget(QLabel("Brightness and Contrast:"))
        
        self.auto_adjust_btn = QPushButton("Auto Adjust")
        self.auto_adjust_btn.clicked.connect(self.auto_adjust_bg)
        bg_controls_layout.addWidget(self.auto_adjust_btn)
        
        self.reset_bg_btn = QPushButton("Reset")
        self.reset_bg_btn.clicked.connect(self.reset_bg_adjustment)
        bg_controls_layout.addWidget(self.reset_bg_btn)
        
        controls_layout.addLayout(bg_controls_layout)
        
        mask_display_group.setLayout(controls_layout)
        sidebar_layout.addWidget(mask_display_group)
        
        # Positioning Controls - directly after Display Controls
        offset_group = QGroupBox("Positioning Controls")
        offset_layout = QVBoxLayout()
        
        self.offset_label = QLabel("Offset: X=0.0, Y=0.0")
        offset_layout.addWidget(self.offset_label)
        
        # Cell selection and deletion
        self.selected_cell_id = 0
        self.selected_cell_label = QLabel("Selected: None")
        offset_layout.addWidget(self.selected_cell_label)
        
        self.delete_cell_btn = QPushButton("Delete Selected Cell")
        self.delete_cell_btn.clicked.connect(self.delete_selected_cell)
        self.delete_cell_btn.setEnabled(False)
        offset_layout.addWidget(self.delete_cell_btn)
        
        # Add step size adjustment
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.1", "0.5", "1", "5", "10", "20"])
        self.step_combo.setCurrentIndex(1)  # Default to 1 pixel step
        step_layout.addWidget(self.step_combo)
        offset_layout.addLayout(step_layout)
        
        # Arrow controls
        arrow_grid = QGridLayout()
        
        # Up button
        self.up_btn = QPushButton("↑")
        self.up_btn.clicked.connect(self.move_up)
        arrow_grid.addWidget(self.up_btn, 0, 1)
        
        # Left button
        self.left_btn = QPushButton("←")
        self.left_btn.clicked.connect(self.move_left)
        arrow_grid.addWidget(self.left_btn, 1, 0)
        
        # Right button
        self.right_btn = QPushButton("→")
        self.right_btn.clicked.connect(self.move_right)
        arrow_grid.addWidget(self.right_btn, 1, 2)
        
        # Down button
        self.down_btn = QPushButton("↓")
        self.down_btn.clicked.connect(self.move_down)
        arrow_grid.addWidget(self.down_btn, 2, 1)
        
        offset_layout.addLayout(arrow_grid)
        
        # Reset button
        self.reset_offset_btn = QPushButton("Reset Position")
        self.reset_offset_btn.clicked.connect(self.reset_offset)
        offset_layout.addWidget(self.reset_offset_btn)
        
        offset_group.setLayout(offset_layout)
        sidebar_layout.addWidget(offset_group)
        
        # Apply & Registration buttons
        apply_layout = QHBoxLayout()
        
        # Save Mask button
        self.save_btn = QPushButton("Save Mask")
        self.save_btn.clicked.connect(self.save_shifted_mask)
        apply_layout.addWidget(self.save_btn)
        
        # Apply Registration button (open registration dialog)
        self.apply_btn = QPushButton("Apply Registration")
        self.apply_btn.clicked.connect(self.apply_registration)
        apply_layout.addWidget(self.apply_btn)
        
        sidebar_layout.addLayout(apply_layout)
        
        # Stretch at the bottom to push controls up
        sidebar_layout.addStretch()
        
        # Set fixed width for sidebar
        left_sidebar.setLayout(sidebar_layout)
        left_sidebar.setFixedWidth(300)
        main_layout.addWidget(left_sidebar)
        
        # Right area for visualization (expanded)
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        
        # Visualization area
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        viz_layout.addWidget(self.canvas)
        
        viz_widget.setLayout(viz_layout)
        main_layout.addWidget(viz_widget, stretch=1)  # Make it expandable
        
        # Mouse events for dragging the mask
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Keyboard events for arrow key navigation
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize empty display
        self.bg_plot = None
        self.mask_plot = None
        self.update_display()
        
        # Set initial window title
        self.update_window_title()
    
    def update_window_title(self):
        """Update window title with current mask filename"""
        if self.mask_file_path:
            mask_filename = os.path.basename(self.mask_file_path)
            if self.is_batch_mode and hasattr(self, 'mask_files') and self.mask_files:
                current_num = self.current_mask_index + 1
                total_num = len(self.mask_files)
                self.setWindowTitle(f"Shift Analyzer - [{current_num}/{total_num}] {mask_filename}")
            else:
                self.setWindowTitle(f"Shift Analyzer - {mask_filename}")
        else:
            if self.is_batch_mode:
                self.setWindowTitle('Shift Analyzer - Batch Mode')
            else:
                self.setWindowTitle('Shift Analyzer - Single File Mode')
    
    def toggle_processing_mode(self):
        """Toggle between single file and batch processing modes"""
        # Auto-save current mask before switching modes (if auto-save is enabled)
        if hasattr(self, 'mask_files') and self.mask_files:
            print(f"\n--- Switching processing mode ---")
            self.auto_save_current_mask()
        
        self.is_batch_mode = self.batch_mode.isChecked()
        
        # Show/hide appropriate controls
        self.single_controls.setVisible(not self.is_batch_mode)
        self.batch_controls.setVisible(self.is_batch_mode)
        self.bg_file_controls.setVisible(not self.is_batch_mode)
        
        # Update window title based on current state
        self.update_window_title()
    
    def select_mask_folder(self):
        """Select folder containing mask files"""
        folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder", "")
        if folder:
            self.mask_folder = folder
            self.mask_folder_label.setText(os.path.basename(folder))
            
            # Find all mask files in the folder, excluding system files
            self.mask_files = []
            for file in os.listdir(folder):
                # Skip macOS system files that start with "._"
                if file.startswith('._'):
                    continue
                if file.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.bmp')) and '_dic' in file.lower():
                    self.mask_files.append(file)
            
            self.mask_files.sort()  # Sort alphabetically
            self.current_mask_index = 0
            
            if self.mask_files:
                self.update_current_file_info()
                if self.fish_folder:  # If fish folder is also selected
                    self.load_current_mask_and_fish()
            else:
                QMessageBox.warning(self, "Warning", "No mask files found in the selected folder.")
    
    def select_fish_folder(self):
        """Select folder containing FISH image files"""
        folder = QFileDialog.getExistingDirectory(self, "Select FISH Image Folder", "")
        if folder:
            self.fish_folder = folder
            self.fish_folder_label.setText(os.path.basename(folder))
            
            if self.mask_files:  # If mask files are already loaded
                self.load_current_mask_and_fish()
    
    def select_output_folder(self):
        """Select folder for saving shifted masks"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            self.output_folder = folder
            self.output_folder_label.setText(os.path.basename(folder))
    
    def find_matching_fish(self, mask_filename):
        """Find matching FISH file based on mask filename, using logic from Mask2Outline.py"""
        if not self.fish_folder:
            return None
            
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
            
            # Look for FISH files with both the numeric ID and sequence ID
            for file in os.listdir(self.fish_folder):
                # Skip macOS system files that start with "._"
                if file.startswith('._'):
                    continue
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
            
            print(f"No FISH match found for {mask_filename}")
            return None
        else:
            print(f"Could not extract numeric ID or sequence ID from {mask_filename}")
            
            # Fallback method - extract base name and try to match
            mask_base = os.path.splitext(mask_filename)[0].replace('_DIC', '', 1).lower()
            print(f"Using fallback method with mask base: {mask_base}")
            
            for file in os.listdir(self.fish_folder):
                # Skip macOS system files that start with "._"
                if file.startswith('._'):
                    continue
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
    
    def load_current_mask_and_fish(self):
        """Load current mask and its matching FISH image"""
        if not self.mask_files or self.current_mask_index >= len(self.mask_files):
            return
            
        current_mask = self.mask_files[self.current_mask_index]
        mask_path = os.path.join(self.mask_folder, current_mask)
        
        # Clear previous mask and background data
        if self.mask_plot is not None:
            try:
                self.mask_plot.remove()
            except:
                pass
            self.mask_plot = None
        
        if self.bg_plot is not None:
            try:
                self.bg_plot.remove()
            except:
                pass
            self.bg_plot = None
        
        self.colored_mask = None
        self.bg_img = None
        self.processed_bg = None
        
        # Load mask
        try:
            self.mask_img = imread(mask_path, as_gray=True)
            self.mask_file_path = mask_path
            self.mask_file_label.setText(current_mask)
            
            # Reset offset when loading a new mask
            self.offset_x = 0
            self.offset_y = 0
            self.offset_label.setText("Offset: X=0, Y=0")
            
            # Clear cell selection
            self.selected_cell_id = 0
            self.update_selected_cell_display()
            
            # Update window title with current mask
            self.update_window_title()
            
            print(f"Loaded mask: {current_mask}")
            print(f"Shape: {self.mask_img.shape}")
            print(f"Data type: {self.mask_img.dtype}")
            print(f"Min value: {np.min(self.mask_img)}, Max value: {np.max(self.mask_img)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mask {current_mask}: {str(e)}")
            return
        
        # Find and load matching FISH image
        fish_file = self.find_matching_fish(current_mask)
        print(f"Attempting to match mask '{current_mask}' with FISH image...")
        if fish_file:
            print(f"Found matching FISH file: {fish_file}")
            fish_path = os.path.join(self.fish_folder, fish_file)
            try:
                fish_img = imread(fish_path)
                
                # Handle different image types
                if len(fish_img.shape) == 3 and fish_img.shape[0] > 4:
                    # Z-stack with multiple slices (first dimension is the stack)
                    max_projection = np.max(fish_img, axis=0)
                    self.bg_file_label.setText(f"FISH: {fish_file} (Z-stack with {fish_img.shape[0]} slices)")
                else:
                    # Single 2D image or RGB image
                    max_projection = fish_img
                    self.bg_file_label.setText(f"FISH: {fish_file}")
                
                # Convert 16-bit to 8-bit if needed with proper scaling
                if max_projection.dtype == np.uint16:
                    max_val = np.max(max_projection)
                    if max_val > 0:
                        scaled = (max_projection.astype(np.float32) * 255 / max_val).astype(np.uint8)
                    else:
                        scaled = max_projection.astype(np.uint8)
                    max_projection = scaled
                
                self.bg_img = max_projection
                self.current_fish_file = fish_file
                
                # Reset background plot to force redraw
                self.bg_plot = None
                
                # Auto adjust brightness/contrast by default
                self.auto_adjust_bg()
                
                print(f"Loaded FISH image: {fish_file}")
                print(f"Shape: {self.bg_img.shape}")
                print(f"Data type: {self.bg_img.dtype}")
                
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to load FISH image {fish_file}: {str(e)}")
                self.bg_img = None
                self.bg_file_label.setText("No matching FISH image found")
        else:
            self.bg_img = None
            self.bg_file_label.setText("No matching FISH image found")
        
        # Update display
        self.update_colormap()
    
    def update_current_file_info(self):
        """Update the current file information and window title"""
        if self.mask_files:
            total_files = len(self.mask_files)
            # Enable/disable navigation buttons
            self.prev_btn.setEnabled(self.current_mask_index > 0)
            self.next_btn.setEnabled(self.current_mask_index < total_files - 1)
            
            # Update window title if mask is loaded
            if self.mask_file_path:
                self.update_window_title()
        else:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.update_window_title()
    
    def auto_save_current_mask(self):
        """Auto-save current mask (whether shifted or not) if output folder is set and auto-save is enabled"""
        # Check if auto-save is enabled
        if hasattr(self, 'auto_save_checkbox') and not self.auto_save_checkbox.isChecked():
            print("⏭️ Auto-save skipped: Auto-save is disabled by user")
            return
            
        if not self.output_folder:
            print("📁 Auto-save skipped: No output folder selected")
            return
            
        if not (self.mask_img is not None and self.mask_file_path is not None):
            print("🚫 Auto-save skipped: No mask loaded")
            return
            
        # Always save, even if no offset is applied
        if self.offset_x == 0 and self.offset_y == 0:
            print(f"💾 Auto-saving mask (no offset): {os.path.basename(self.mask_file_path)}")
        else:
            print(f"💾 Auto-saving mask (offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}): {os.path.basename(self.mask_file_path)}")
            
        try:
            # Generate output filename - use original filename
            output_filename = os.path.basename(self.mask_file_path)
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Use current mask (which may have deletions) and apply any offset
            current_mask = self.mask_img.copy()
            x_shift = int(round(self.offset_x))
            y_shift = int(round(self.offset_y))
            processed_mask = self.shift_without_warp(current_mask, x_shift, y_shift)
            
            # Save mask with original filename
            imsave(output_path, processed_mask, check_contrast=False)
            
            if self.offset_x == 0 and self.offset_y == 0:
                print(f"✅ Auto-saved: {output_filename} (no offset)")
            else:
                print(f"✅ Auto-saved: {output_filename} (offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f})")
            
            # Update window title to show save status temporarily
            original_title = self.windowTitle()
            self.setWindowTitle(f"{original_title} - SAVED ✅")
            # Reset title after 2 seconds
            QTimer.singleShot(2000, lambda: self.update_window_title())
                
        except Exception as e:
            print(f"❌ Error auto-saving mask: {str(e)}")
            # Update window title to show error status temporarily
            original_title = self.windowTitle()
            self.setWindowTitle(f"{original_title} - SAVE ERROR ❌")
            # Reset title after 2 seconds
            QTimer.singleShot(2000, lambda: self.update_window_title())
    
    def prev_mask(self):
        """Go to previous mask in the batch"""
        if self.current_mask_index > 0:
            print(f"\n--- Navigating to previous mask ---")
            # Auto-save current mask before switching
            self.auto_save_current_mask()
            
            self.current_mask_index -= 1
            self.update_current_file_info()
            self.load_current_mask_and_fish()
    
    def next_mask(self):
        """Go to next mask in the batch"""
        if self.current_mask_index < len(self.mask_files) - 1:
            print(f"\n--- Navigating to next mask ---")
            # Auto-save current mask before switching
            self.auto_save_current_mask()
            
            self.current_mask_index += 1
            self.update_current_file_info()
            self.load_current_mask_and_fish()
    
    def get_step_size(self):
        """Get the current step size from the combo box"""
        return float(self.step_combo.currentText())
        
    def move_up(self):
        """Move mask up by the step size"""
        step = self.get_step_size()
        self.offset_y -= step  # Subtract because Y is inverted in image coordinates
        self.update_offset_display()
        self.update_display()
    
    def move_down(self):
        """Move mask down by the step size"""
        step = self.get_step_size()
        self.offset_y += step  # Add because Y is inverted in image coordinates
        self.update_offset_display()
        self.update_display()
    
    def move_left(self):
        """Move mask left by the step size"""
        step = self.get_step_size()
        self.offset_x -= step
        self.update_offset_display()
        self.update_display()
    
    def move_right(self):
        """Move mask right by the step size"""
        step = self.get_step_size()
        self.offset_x += step
        self.update_offset_display()
        self.update_display()
    
    def update_offset_display(self):
        """Update the offset display label"""
        self.offset_label.setText(f"Offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
    
    def on_key_press(self, event):
        """Handle key press events for arrow keys and navigation"""
        if event.key == 'up':
            self.move_up()
        elif event.key == 'down':
            self.move_down()
        elif event.key == 'left':
            self.move_left()
        elif event.key == 'right':
            self.move_right()
        elif event.key == 'pageup' and self.is_batch_mode:
            self.prev_mask()
        elif event.key == 'pagedown' and self.is_batch_mode:
            self.next_mask()
        
    def reset_offset(self):
        self.offset_x = 0
        self.offset_y = 0
        self.update_offset_display()
        self.update_display()
    
    def delete_selected_cell(self):
        """Delete the currently selected cell"""
        if self.selected_cell_id > 0 and self.mask_img is not None:
            deleted_id = self.selected_cell_id
            
            # Set selected cell pixels to 0 (background)
            self.mask_img[self.mask_img == self.selected_cell_id] = 0
            
            # Ensure mask data type is preserved (important for saving)
            if self.mask_img.dtype != np.uint16:
                self.mask_img = self.mask_img.astype(np.uint16)
            
            # Clear selection
            self.selected_cell_id = 0
            self.update_selected_cell_display()
            
            # Force complete figure cleanup and regeneration
            self.figure.clear()  # Clear entire figure
            self.colored_mask = None  # Clear cached colored mask
            self.mask_plot = None    # Clear plot cache
            self.bg_plot = None      # Clear background plot cache
            
            # Recreate the axes and clear any residual content
            self.ax = self.figure.add_subplot(111)
            self.ax.clear()  # Clear axes content
            self.ax.set_axis_off()
            
            # Force immediate canvas draw to clear old content
            self.canvas.draw()
            
            # Regenerate everything from scratch
            self.update_colormap()   # This will regenerate colored mask and call update_display
            
            # Force canvas redraw to ensure no lingering graphics
            self.canvas.draw_idle()
            
            print(f"Deleted cell {deleted_id}")
    
    def update_selected_cell_display(self):
        """Update the selected cell display label and button state"""
        if self.selected_cell_id > 0:
            self.selected_cell_label.setText(f"Selected: Cell {self.selected_cell_id}")
            self.delete_cell_btn.setEnabled(True)
        else:
            self.selected_cell_label.setText("Selected: None")
            self.delete_cell_btn.setEnabled(False)
        
    def select_mask_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Mask File", "", 
                                                 "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")
        if file_path:
            try:
                # Clear previous mask data completely
                if self.mask_plot is not None:
                    try:
                        self.mask_plot.remove()
                    except:
                        pass
                    self.mask_plot = None
                
                # Clear the colored mask to ensure no residual data
                self.colored_mask = None
                
                # Load the new mask
                self.mask_img = imread(file_path, as_gray=True)
                self.mask_file_path = file_path
                self.mask_file_label.setText(os.path.basename(file_path))
                
                # Update window title with current mask filename
                self.update_window_title()
                
                # Reset offset when loading a new mask
                self.offset_x = 0
                self.offset_y = 0
                self.offset_label.setText("Offset: X=0, Y=0")
                
                # Clear cell selection
                self.selected_cell_id = 0
                self.update_selected_cell_display()
                
                # Force a complete redraw
                self.update_colormap()
                
                # Print mask info for debugging
                print(f"Loaded mask: {os.path.basename(file_path)}")
                print(f"Shape: {self.mask_img.shape}")
                print(f"Data type: {self.mask_img.dtype}")
                print(f"Min value: {np.min(self.mask_img)}, Max value: {np.max(self.mask_img)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.mask_file_label.setText(f"Error: {str(e)}")
                self.mask_file_path = None
                self.update_window_title()
    
    def select_bg_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Background Image(s)", "", 
                                              "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")
        if files:
            try:
                # Check if a single file is selected
                if len(files) == 1:
                    # Try to load as multi-page TIFF (Z-stack)
                    img = imread(files[0])
                    
                    # If the image is a 3D stack
                    if len(img.shape) == 3 and img.shape[0] > 4:
                        # It's a Z-stack with multiple slices (first dimension is the stack)
                        max_projection = np.max(img, axis=0)
                        self.bg_file_label.setText(f"Z-stack with {img.shape[0]} slices")
                    else:
                        # Single 2D image or RGB image
                        max_projection = img
                        self.bg_file_label.setText(os.path.basename(files[0]))
                else:
                    # Multiple files selected
                    images = []
                    for f in files:
                        img = imread(f)
                        # Handle case where image itself is a stack
                        if len(img.shape) == 2:  # 2D grayscale image
                            images.append(img)
                        elif len(img.shape) == 3 and img.shape[2] <= 4:  # RGB/RGBA image
                            images.append(img)
                        elif len(img.shape) == 3:  # Multi-page Z-stack
                            # For each slice in the stack
                            for i in range(img.shape[0]):
                                images.append(img[i])
                        else:  # Regular 2D image
                            images.append(img)
                    
                    # Ensure all images have the same dimensions
                    first_shape = images[0].shape
                    images = [img for img in images if img.shape == first_shape]
                    
                    # Perform max projection
                    if images:
                        max_projection = np.max(np.array(images), axis=0)
                        self.bg_file_label.setText(f"{len(images)} slices merged")
                    else:
                        raise ValueError("No compatible images found for merging")
                
                # Convert 16-bit to 8-bit if needed with proper scaling
                if max_projection.dtype == np.uint16:
                    # Scale properly from uint16 to uint8
                    max_val = np.max(max_projection)
                    if max_val > 0:  # Avoid division by zero
                        scaled = (max_projection.astype(np.float32) * 255 / max_val).astype(np.uint8)
                    else:
                        scaled = max_projection.astype(np.uint8)
                    max_projection = scaled
                
                self.bg_img = max_projection
                
                # Reset background plot to force redraw
                self.bg_plot = None
                
                # Print background image info for debugging
                print(f"Loaded background image")
                print(f"Shape: {self.bg_img.shape}")
                print(f"Data type: {self.bg_img.dtype}")
                print(f"Min value: {np.min(self.bg_img)}, Max value: {np.max(self.bg_img)}")
                
                # Auto adjust brightness/contrast by default
                self.auto_adjust_bg()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.bg_file_label.setText(f"Error: {str(e)}")
    
    def update_colormap(self):
        if self.mask_img is not None:
            # Normalize the mask to range 0-1
            mask_normalized = img_as_float(self.mask_img)
            
            # Apply colormap
            cmap = plt.get_cmap('viridis')
            colored = cmap(mask_normalized)
            
            # Set alpha channel based on mask values
            # Where mask is 0, make fully transparent
            colored[..., 3] = np.where(mask_normalized > 0, 1.0, 0.0)
            
            self.colored_mask = colored
            
            # If colormap is updated, reset the mask_plot to force redraw
            self.mask_plot = None
            
            self.update_display()
    
    def reset_bg_adjustment(self):
        """Reset background image to original state"""
        if self.bg_img is not None:
            self.processed_bg = self.bg_img.copy()
            self.update_display()
    
    def auto_adjust_bg(self):
        if self.bg_img is not None:
            try:
                # Apply adaptive histogram equalization for auto brightness/contrast
                if len(self.bg_img.shape) == 3 and self.bg_img.shape[2] >= 3:  # RGB/RGBA
                    # Process each channel
                    adjusted = np.zeros_like(self.bg_img)
                    for i in range(min(3, self.bg_img.shape[2])):  # Process up to 3 channels
                        # Use CLAHE for better results with microscopy images
                        adjusted[..., i] = exposure.equalize_adapthist(self.bg_img[..., i], clip_limit=0.03)
                    self.processed_bg = img_as_ubyte(adjusted)
                else:  # Grayscale
                    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    equalized = exposure.equalize_adapthist(self.bg_img, clip_limit=0.03)
                    self.processed_bg = img_as_ubyte(equalized)
                
                self.update_display()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Auto-adjust error: {str(e)}")
    
    def update_display(self):
        # Clear and recreate the figure only if needed
        if not hasattr(self, 'ax') or self.figure.axes == []:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_axis_off()
        
        # First ensure all previously created plots are removed
        if self.bg_plot is not None:
            try:
                self.bg_plot.remove()
            except:
                pass
            self.bg_plot = None
            
        if self.mask_plot is not None:
            try:
                self.mask_plot.remove()
            except:
                pass
            self.mask_plot = None
        
        # Display background if available - background is fixed
        if self.processed_bg is not None:
            bg_cmap = 'gray' if len(self.processed_bg.shape) == 2 else None
            self.bg_plot = self.ax.imshow(
                self.processed_bg, 
                cmap=bg_cmap,
                extent=[0, self.processed_bg.shape[1], self.processed_bg.shape[0], 0],
                zorder=1  # Lower zorder means it's drawn first (behind)
            )
        
        # Display mask overlay if available - mask moves with offset
        if self.colored_mask is not None:
            # Get mask opacity
            opacity = self.opacity_slider.value() / 100.0
            
            # Create a copy with adjusted opacity
            overlay = self.colored_mask.copy()
            
            # Highlight selected cell in yellow
            if self.selected_cell_id > 0 and self.mask_img is not None:
                selected_mask = (self.mask_img == self.selected_cell_id)
                overlay[selected_mask] = [1.0, 1.0, 0.0, 1.0]  # Yellow with full opacity
            
            # Only adjust opacity where mask isn't transparent (but keep selected cell bright)
            if self.selected_cell_id > 0 and self.mask_img is not None:
                selected_mask = (self.mask_img == self.selected_cell_id)
                overlay[..., 3] = np.where((overlay[..., 3] > 0) & (~selected_mask), opacity, overlay[..., 3])
            else:
                overlay[..., 3] = np.where(overlay[..., 3] > 0, opacity, 0)
            
            mask_h, mask_w = overlay.shape[:2]
            
            # Calculate the extent with the current offset
            # This is critical - extent defines the coordinates for the mask
            x_min = self.offset_x
            x_max = self.offset_x + mask_w
            y_min = self.offset_y
            y_max = self.offset_y + mask_h
            
            self.mask_plot = self.ax.imshow(
                overlay, 
                extent=[x_min, x_max, y_max, y_min],
                interpolation='nearest',
                zorder=2  # Higher zorder means it's drawn last (on top)
            )
            
            # Print debug info about the mask
            print(f"Mask dimensions: {mask_w}x{mask_h}")
            print(f"Mask displayed at: X={x_min:.1f}-{x_max:.1f}, Y={y_min:.1f}-{y_max:.1f}")
        
        # Set appropriate limits to see all content
        if self.processed_bg is not None:
            bg_h, bg_w = self.processed_bg.shape[:2]
            self.ax.set_xlim(0, bg_w)
            self.ax.set_ylim(bg_h, 0)  # Inverted y-axis for image coordinates
        
        self.canvas.draw()
    
    def on_press(self, event):
        if not event.inaxes or self.colored_mask is None:
            return
            
        # Get the mask dimensions
        mask_h, mask_w = self.colored_mask.shape[:2]
            
        # Calculate mask boundaries in the display
        x_min = self.offset_x
        x_max = self.offset_x + mask_w
        y_min = self.offset_y
        y_max = self.offset_y + mask_h
            
        # Check if click is within the mask boundaries
        if (x_min <= event.xdata <= x_max and 
            y_min <= event.ydata <= y_max):
                
            # Convert click position to mask array coordinates
            mask_x = int(event.xdata - self.offset_x)
            mask_y = int(event.ydata - self.offset_y)
                
            # Ensure we're within the mask array bounds
            if (0 <= mask_x < mask_w and 0 <= mask_y < mask_h):
                # Check if we clicked on a non-transparent part
                alpha = self.colored_mask[mask_y, mask_x, 3]
                if alpha > 0:
                    # Get cell ID at clicked position for selection
                    if self.mask_img is not None:
                        cell_id = self.mask_img[mask_y, mask_x]
                        if cell_id > 0:
                            self.selected_cell_id = cell_id
                            self.update_selected_cell_display()
                            print(f"Selected cell {cell_id}")
                    
                    self.drag_enabled = True
                    self.last_pos = (event.xdata, event.ydata)
                    # Print debug info
                    print(f"Drag started at: {event.xdata:.1f}, {event.ydata:.1f}")
                    print(f"Current mask offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
    
    def on_release(self, event):
        if self.drag_enabled:
            print(f"Drag ended. Final mask offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
        self.drag_enabled = False
    
    def on_motion(self, event):
        if self.drag_enabled and event.inaxes and self.last_pos:
            # Calculate the distance moved
            dx = event.xdata - self.last_pos[0]
            dy = event.ydata - self.last_pos[1]
            
            # Update the mask position by changing its offset
            self.offset_x += dx
            self.offset_y += dy
            
            # Debug info
            print(f"Drag delta: dx={dx:.1f}, dy={dy:.1f}")
            print(f"New offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
            
            # Update offset display
            self.update_offset_display()
            
            # Update the last position for the next movement calculation
            self.last_pos = (event.xdata, event.ydata)
            
            # Redraw the display with the new offset
            self.update_display()
    
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
    
    def apply_registration(self):
        """Open registration tool for batch processing."""
        try:
            # Create an instance of RegistrationTool without starting a new QApplication
            reg_tool = registration.RegistrationTool(parent=self, offset_x=self.offset_x, offset_y=self.offset_y)
            
            # Show the dialog non-modally (won't block the main window)
            reg_tool.show()
            
            # Optional: Connect the dialog's closed signal to a handler if needed
            reg_tool.finished.connect(lambda: print("Registration dialog closed"))
            
        except ImportError:
            QMessageBox.critical(self, "Error", "Registration module not found. Make sure registration.py is in the same directory or in your Python path.")
            return
    
    def save_shifted_mask(self):
        """Apply the current offset to the original mask and save it with original filename."""
        if self.mask_img is None or self.mask_file_path is None:
            QMessageBox.warning(self, "Warning", "Please load a mask file first.")
            return

        # Suggest a default output filename - use original filename
        original_filename = os.path.basename(self.mask_file_path)
        
        # Use output folder if in batch mode and output folder is set
        if self.is_batch_mode and self.output_folder:
            default_dir = self.output_folder
        else:
            default_dir = os.path.dirname(self.mask_file_path)
        
        default_save_path = os.path.join(default_dir, original_filename)

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Mask As", default_save_path,
                                                 "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")

        if not save_path:
            return # User cancelled

        try:
            # Use current mask (which may have deletions) instead of re-reading original file
            # This preserves any cell deletions that were made
            current_mask = self.mask_img.copy()
            
            # Apply the offset using the current values
            x_shift = int(round(self.offset_x))
            y_shift = int(round(self.offset_y))
            
            processed_mask = self.shift_without_warp(current_mask, x_shift, y_shift)
            
            # Save the mask using skimage.io.imsave to preserve data type
            imsave(save_path, processed_mask, check_contrast=False)
            
            QMessageBox.information(self, "Success", f"Mask saved to:\n{save_path}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save mask: {str(e)}")

    def showEvent(self, event):
        super().showEvent(event)
        # After showing, remove the always-on-top flag
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()
    
    def closeEvent(self, event):
        """Handle window close event - auto-save current mask if enabled"""
        if self.is_batch_mode:
            print(f"\n--- Application closing ---")
            self.auto_save_current_mask()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MaskVisualizationTool()
    window.show()
    sys.exit(app.exec_())
