import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import sys

# Safe OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"OpenCV import failed: {e}")
    print("Please reinstall OpenCV with: conda install opencv -c conda-forge")
    CV2_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes, label
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"SciPy import failed: {e}")
    print("Please install SciPy with: conda install scipy")
    SCIPY_AVAILABLE = False

class DividingLineUNet(nn.Module):
    """
    U-Net model for dividing line prediction
    
    Task: Binary cell mask → Binary dividing line prediction
    Input: Binary mask of budding cells
    Output: Binary mask of dividing line (1 = dividing line, 0 = background)
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        super(DividingLineUNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # For binary output
        
    def _double_conv(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        
        # Decoder path
        upconv4 = self.upconv4(bottleneck)
        concat4 = torch.cat([upconv4, enc4], dim=1)
        dec4 = self.dec4(concat4)
        
        upconv3 = self.upconv3(dec4)
        concat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(concat1)
        
        # Output
        out = self.out_conv(dec1)
        out = self.sigmoid(out)
        
        return out

class DividingLineDataset(Dataset):
    """
    Dataset class for dividing line prediction training data
    
    Data structure:
    - Input: Binary cell masks (budding cells)
    - Labels: Binary dividing line masks
    """
    
    def __init__(self, data_dir: str = "processed_dividing_line_data", transform=None, augment=True, verbose=True):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.transform = transform
        self.augment = augment
        
        # Get all sample files, excluding macOS hidden files
        self.sample_files = [f for f in self.images_dir.glob("*.npy") 
                           if not f.name.startswith("._")]
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No training samples found in {self.images_dir}")
        
        if verbose:
            print(f"Found {len(self.sample_files)} training samples")
        
        # Load generation summary for additional info
        summary_file = self.data_dir / "generation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)
            if verbose:
                print(f"Dataset generated on: {self.summary.get('generation_date', 'Unknown')}")
                print(f"Total budding pairs: {self.summary.get('total_budding_pairs', len(self.sample_files))}")
                print(f"Line thickness: {self.summary.get('line_thickness', 'Unknown')}")
        else:
            self.summary = None
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        # Load image and label
        image_file = self.sample_files[idx]
        label_file = self.labels_dir / image_file.name
        
        # Load data (already preprocessed)
        image = np.load(image_file, allow_pickle=True).astype(np.float32)
        label = np.load(label_file, allow_pickle=True).astype(np.float32)
        
        # Add channel dimension for image if needed
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # Add channel dimension for label if needed
        if len(label.shape) == 2:
            label = label[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        # Apply data augmentation if enabled
        if self.augment and self.transform:
            # Apply same transform to both image and label
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            label = self.transform(label)
        
        return image, label

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in dividing line prediction
    
    The dividing line pixels are much fewer than background pixels,
    so standard BCE loss will be dominated by easy background examples.
    Focal loss focuses learning on hard examples (like thin dividing lines).
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        
    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] - model predictions (after sigmoid)
        targets: [B, 1, H, W] - ground truth binary masks
        """
        # Flatten for easier computation
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""
    
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        inputs: [B, 1, H, W] - model predictions (after sigmoid)
        targets: [B, 1, H, W] - ground truth binary masks
        """
        # Flatten
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice

class CombinedDividingLineLoss(nn.Module):
    """
    Combined loss function optimized for dividing line prediction
    
    Combines:
    - Focal Loss: Handles class imbalance and focuses on hard examples
    - Dice Loss: Ensures good overlap for the thin dividing lines
    - BCE Loss: Provides stable gradients
    """
    
    def __init__(self, focal_weight=0.5, dice_weight=0.3, bce_weight=0.2, 
                 focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedDividingLineLoss, self).__init__()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        
        print(f"CombinedDividingLineLoss initialized:")
        print(f"   Focal: {focal_weight} (α={focal_alpha}, γ={focal_gamma})")
        print(f"   Dice: {dice_weight}")
        print(f"   BCE: {bce_weight}")
    
    def forward(self, inputs, targets):
        """
        Compute combined loss
        """
        # Compute individual losses
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        
        # Combine losses
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.bce_weight * bce)
        
        return total_loss, {
            'focal': focal.item() if hasattr(focal, 'item') else float(focal),
            'dice': dice.item() if hasattr(dice, 'item') else float(dice),
            'bce': bce.item() if hasattr(bce, 'item') else float(bce),
            'total': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        }

class CellSeparator:
    """
    Enhanced post-processing class for separating cells using predicted dividing lines
    
    Process:
    1. Take predicted dividing line mask
    2. Apply morphological operations for robust separation
    3. Remove dividing line from cell mask to create separation
    4. Identify connected components
    5. Assign mother/daughter based on area (larger = mother)
    6. Validate separation quality
    """
    
    def __init__(self, min_component_area=50, dilation_iterations=1, 
                 morphology_kernel_size=3, enable_validation=True):
        self.min_component_area = min_component_area
        self.dilation_iterations = dilation_iterations
        self.morphology_kernel_size = morphology_kernel_size
        self.enable_validation = enable_validation
        
        print(f"CellSeparator initialized:")
        print(f"   Min component area: {min_component_area} pixels")
        print(f"   Dilation iterations: {dilation_iterations}")
        print(f"   Morphology kernel size: {morphology_kernel_size}")
        print(f"   Validation enabled: {enable_validation}")
        
    def separate_cells(self, cell_mask: np.ndarray, dividing_line_mask: np.ndarray, 
                      threshold=0.5, return_assignments=True, return_debug_info=False):
        """
        Enhanced cell separation using predicted dividing line
        
        Args:
            cell_mask: [H, W] Binary mask of the complete cell
            dividing_line_mask: [H, W] Predicted dividing line (0-1 probabilities)
            threshold: Threshold for converting probabilities to binary
            return_assignments: Whether to return mother/daughter assignments
            return_debug_info: Whether to return debug visualization data
            
        Returns:
            separated_mask: [H, W] Mask with separated cells (1=daughter, 2=mother, 0=background)
            assignments: Dict with component info and quality metrics
            debug_info: Dict with intermediate processing steps (if return_debug_info=True)
        """
        
        # Input validation
        if cell_mask.shape != dividing_line_mask.shape:
            raise ValueError(f"Shape mismatch: cell_mask {cell_mask.shape} vs dividing_line_mask {dividing_line_mask.shape}")
        
        # Convert to binary masks
        cell_binary = (cell_mask > 0.5).astype(np.uint8)
        line_binary = (dividing_line_mask > threshold).astype(np.uint8)
        
        # Store original for debugging
        original_line = line_binary.copy()
        
        # Enhanced morphological processing for better separation
        if CV2_AVAILABLE and self.morphology_kernel_size > 0:
            # Create elliptical kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morphology_kernel_size, self.morphology_kernel_size))
            
            # Close small gaps in the dividing line
            line_binary = cv2.morphologyEx(line_binary, cv2.MORPH_CLOSE, kernel)
            
            # Optionally dilate the dividing line to ensure robust separation
            if self.dilation_iterations > 0:
                line_binary = cv2.dilate(line_binary, kernel, iterations=self.dilation_iterations)
        
        elif SCIPY_AVAILABLE and self.dilation_iterations > 0:
            # Use scipy for morphological operations if OpenCV not available
            from scipy.ndimage import binary_dilation, binary_closing
            
            # Create circular structure element
            y, x = np.ogrid[-1:2, -1:2]
            struct_elem = x*x + y*y <= 1
            
            # Close gaps and dilate
            line_binary = binary_closing(line_binary, structure=struct_elem).astype(np.uint8)
            if self.dilation_iterations > 0:
                for _ in range(self.dilation_iterations):
                    line_binary = binary_dilation(line_binary, structure=struct_elem).astype(np.uint8)
        
        # Remove dividing line from cell mask to create separation
        separated_regions = cell_binary.copy()
        separated_regions[line_binary > 0] = 0
        
        # Validate that we actually created a separation
        if self.enable_validation:
            line_pixels_removed = np.sum(line_binary & cell_binary)
            if line_pixels_removed == 0:
                print("Warning: No dividing line pixels overlap with cell mask")
        
        # Find connected components
        if CV2_AVAILABLE:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                separated_regions, connectivity=8)
        elif SCIPY_AVAILABLE:
            labels, num_labels = label(separated_regions)
            # Compute stats manually
            stats = []
            centroids = []
            for i in range(num_labels + 1):
                component_mask = (labels == i)
                area = np.sum(component_mask)
                if area > 0:
                    coords = np.argwhere(component_mask)
                    centroid = coords.mean(axis=0)
                    stats.append([0, 0, 0, 0, area])  # [left, top, width, height, area]
                    centroids.append(centroid)
                else:
                    stats.append([0, 0, 0, 0, 0])
                    centroids.append([0, 0])
            stats = np.array(stats)
            centroids = np.array(centroids)
        else:
            print("Neither OpenCV nor SciPy available. Using fallback method.")
            # Simple fallback: assume two regions exist
            separated_mask = np.zeros_like(cell_mask, dtype=np.uint8)
            separated_mask[cell_binary > 0] = 1  # Mark all as daughter initially
            return separated_mask, {'warning': 'Fallback method used'}
        
        # Filter components by minimum area
        valid_components = []
        for i in range(1, num_labels):  # Skip background (label 0)
            if CV2_AVAILABLE:
                area = stats[i, cv2.CC_STAT_AREA]
            else:
                area = stats[i, 4]  # Area is at index 4
                
            if area >= self.min_component_area:
                valid_components.append((i, area))
        
        # Create output mask
        separated_mask = np.zeros_like(cell_mask, dtype=np.uint8)
        
        # Calculate quality metrics
        original_cell_area = np.sum(cell_binary)
        separated_area = np.sum(separated_regions)
        area_preserved_ratio = separated_area / original_cell_area if original_cell_area > 0 else 0
        
        assignments = {
            'num_components': len(valid_components),
            'components': [],
            'mother_label': None,
            'daughter_label': None,
            'separation_success': False,
            'quality_metrics': {
                'original_area': int(original_cell_area),
                'separated_area': int(separated_area),
                'area_preserved_ratio': float(area_preserved_ratio),
                'line_pixels_removed': int(np.sum(line_binary & cell_binary)),
                'threshold_used': float(threshold)
            }
        }
        
        if len(valid_components) == 0:
            print("No valid components found after separation")
            print(f"   Original area: {original_cell_area}, Separated area: {separated_area}")
            print(f"   Try reducing min_component_area (current: {self.min_component_area})")
            
        elif len(valid_components) == 1:
            print("Only one component found after separation")
            component_id, area = valid_components[0]
            separated_mask[labels == component_id] = 2  # Assign as mother
            assignments['components'].append({'id': component_id, 'area': area, 'type': 'mother'})
            assignments['mother_label'] = 2
            print(f"   Single component area: {area} / {original_cell_area} ({area/original_cell_area:.1%})")
            
        else:
            # Sort components by area (largest first)
            valid_components.sort(key=lambda x: x[1], reverse=True)
            
            # Assign largest as mother, second largest as daughter
            mother_id, mother_area = valid_components[0]
            daughter_id, daughter_area = valid_components[1]
            
            separated_mask[labels == mother_id] = 2     # Mother = 2
            separated_mask[labels == daughter_id] = 1   # Daughter = 1
            
            assignments['components'].append({'id': mother_id, 'area': mother_area, 'type': 'mother'})
            assignments['components'].append({'id': daughter_id, 'area': daughter_area, 'type': 'daughter'})
            assignments['mother_label'] = 2
            assignments['daughter_label'] = 1
            assignments['separation_success'] = True
            
            # Calculate additional quality metrics
            total_assigned_area = mother_area + daughter_area
            assignments['quality_metrics'].update({
                'mother_area': int(mother_area),
                'daughter_area': int(daughter_area),
                'area_ratio_mother_daughter': float(mother_area / daughter_area) if daughter_area > 0 else float('inf'),
                'total_assigned_area': int(total_assigned_area),
                'assignment_efficiency': float(total_assigned_area / original_cell_area) if original_cell_area > 0 else 0
            })
            
            print(f"Successful separation:")
            print(f"   Mother: {mother_area:,} pixels ({mother_area/original_cell_area:.1%})")
            print(f"   Daughter: {daughter_area:,} pixels ({daughter_area/original_cell_area:.1%})")
            print(f"   Area ratio (M/D): {mother_area/daughter_area:.2f}")
            print(f"   Assignment efficiency: {total_assigned_area/original_cell_area:.1%}")
            
            # Handle additional components (if any)
            for i, (comp_id, comp_area) in enumerate(valid_components[2:], start=2):
                separated_mask[labels == comp_id] = i + 1  # Assign unique labels
                assignments['components'].append({'id': comp_id, 'area': comp_area, 'type': f'extra_{i-2}'})
                print(f"Extra component found with area {comp_area:,} pixels")
        
        # Prepare return values
        return_values = [separated_mask]
        
        if return_assignments:
            return_values.append(assignments)
        
        if return_debug_info:
            debug_info = {
                'original_cell_mask': cell_binary,
                'original_line_prediction': original_line,
                'processed_line_mask': line_binary,
                'separated_regions': separated_regions,
                'connected_components_labels': labels if 'labels' in locals() else None,
                'quality_metrics': assignments['quality_metrics']
            }
            return_values.append(debug_info)
        
        return return_values[0] if len(return_values) == 1 else tuple(return_values)
    
    def visualize_separation_process(self, cell_mask: np.ndarray, dividing_line_mask: np.ndarray, 
                                   separated_mask: np.ndarray, assignments: dict, 
                                   debug_info: dict = None, save_path: str = None, 
                                   title: str = "Cell Separation Process"):
        """
        Create a comprehensive visualization of the cell separation process
        
        Args:
            cell_mask: Original complete cell mask
            dividing_line_mask: Predicted dividing line
            separated_mask: Final separated result
            assignments: Assignment information from separate_cells
            debug_info: Debug information (optional)
            save_path: Path to save visualization (optional)
            title: Title for the visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import ListedColormap
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Original cell mask
        axes[0, 0].imshow(cell_mask, cmap='gray')
        axes[0, 0].set_title('Input: Complete Cell Mask')
        axes[0, 0].axis('off')
        
        # 2. Predicted dividing line
        axes[0, 1].imshow(cell_mask, cmap='gray', alpha=0.7)
        axes[0, 1].imshow(dividing_line_mask, cmap='Reds', alpha=0.8)
        axes[0, 1].set_title('Predicted Dividing Line')
        axes[0, 1].axis('off')
        
        # 3. Processed dividing line (if debug info available)
        if debug_info and 'processed_line_mask' in debug_info:
            axes[0, 2].imshow(cell_mask, cmap='gray', alpha=0.7)
            axes[0, 2].imshow(debug_info['processed_line_mask'], cmap='Reds', alpha=0.8)
            axes[0, 2].set_title('Processed Line (After Morphology)')
        else:
            axes[0, 2].imshow(dividing_line_mask, cmap='Reds')
            axes[0, 2].set_title('Dividing Line (Binary)')
        axes[0, 2].axis('off')
        
        # 4. Separated regions (before assignment)
        if debug_info and 'separated_regions' in debug_info:
            axes[1, 0].imshow(debug_info['separated_regions'], cmap='gray')
            axes[1, 0].set_title('After Line Removal')
        else:
            # Reconstruct separated regions
            line_binary = (dividing_line_mask > 0.5).astype(np.uint8)
            cell_binary = (cell_mask > 0.5).astype(np.uint8)
            separated_regions = cell_binary.copy()
            separated_regions[line_binary > 0] = 0
            axes[1, 0].imshow(separated_regions, cmap='gray')
            axes[1, 0].set_title('After Line Removal')
        axes[1, 0].axis('off')
        
        # 5. Final separation with color coding
        if assignments['separation_success']:
            # Create color-coded mask
            colored_mask = np.zeros((*separated_mask.shape, 3))
            colored_mask[separated_mask == 1] = [0, 1, 0]  # Daughter = Green
            colored_mask[separated_mask == 2] = [1, 0, 0]  # Mother = Red
            # Handle extra components
            for i in range(3, separated_mask.max() + 1):
                colored_mask[separated_mask == i] = [0, 0, 1]  # Extra = Blue
            
            axes[1, 1].imshow(colored_mask)
            axes[1, 1].set_title('Final Assignment\n(Red=Mother, Green=Daughter)')
        else:
            axes[1, 1].imshow(separated_mask, cmap='viridis')
            axes[1, 1].set_title('Separation Result\n(Failed)')
        axes[1, 1].axis('off')
        
        # 6. Statistics and quality metrics
        axes[1, 2].axis('off')
        
        # Prepare statistics text
        stats_text = f"""Separation Results:
        
Success: {'Yes' if assignments['separation_success'] else 'No'}
Components Found: {assignments['num_components']}

Quality Metrics:"""
        
        metrics = assignments.get('quality_metrics', {})
        if metrics:
            stats_text += f"""
Original Area: {metrics.get('original_area', 0):,} px
Separated Area: {metrics.get('separated_area', 0):,} px
Area Preserved: {metrics.get('area_preserved_ratio', 0):.1%}
Line Pixels Removed: {metrics.get('line_pixels_removed', 0):,}
Threshold Used: {metrics.get('threshold_used', 0):.2f}"""
            
            if 'mother_area' in metrics:
                stats_text += f"""

Mother Area: {metrics['mother_area']:,} px
Daughter Area: {metrics['daughter_area']:,} px
Area Ratio (M/D): {metrics['area_ratio_mother_daughter']:.2f}
Assignment Efficiency: {metrics['assignment_efficiency']:.1%}"""
        
        # Add component details
        if assignments['components']:
            stats_text += f"\n\nComponents:"
            for comp in assignments['components']:
                stats_text += f"\n• {comp['type'].title()}: {comp['area']:,} px"
        
        # Add recommendations
        stats_text += f"\n\nParameters Used:"
        stats_text += f"\nMin Component Area: {self.min_component_area}"
        stats_text += f"\nDilation Iterations: {self.dilation_iterations}"
        stats_text += f"\nMorphology Kernel: {self.morphology_kernel_size}"
        
        if not assignments['separation_success']:
            stats_text += f"\n\nSuggestions:"
            if assignments['num_components'] == 0:
                stats_text += f"\n• Reduce min_component_area"
                stats_text += f"\n• Lower threshold"
            elif assignments['num_components'] == 1:
                stats_text += f"\n• Increase dilation_iterations"
                stats_text += f"\n• Check dividing line quality"
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Separation visualization saved: {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig

def calculate_iou_binary(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Calculate IoU for binary masks"""
    pred_binary = (pred_mask > 0.5)
    true_binary = (true_mask > 0.5)
    
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_dice_binary(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Calculate Dice coefficient for binary masks"""
    pred_binary = (pred_mask > 0.5)
    true_binary = (true_mask > 0.5)
    
    intersection = np.logical_and(pred_binary, true_binary).sum()
    total = pred_binary.sum() + true_binary.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / total

def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-4, 
                device=None, save_dir="dividing_line_models", early_stopping_patience=25, criterion=None):
    """Train the dividing line U-Net model"""
    
    if device is None:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        else:
            device = torch.device('cpu')
    
    model = model.to(device)
    
    # Use provided criterion or create default
    if criterion is None:
        print("Using default CombinedDividingLineLoss")
        criterion = CombinedDividingLineLoss(
            focal_weight=0.5, dice_weight=0.3, bce_weight=0.2,
            focal_alpha=0.25, focal_gamma=2.0
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, min_lr=1e-7)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Training history
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    loss_components_history = []
    
    best_val_loss = float('inf')
    best_iou = 0.0
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        epoch_loss_components = {'focal': 0.0, 'dice': 0.0, 'bce': 0.0, 'total': 0.0}
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss, loss_components = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accumulate loss components
            for key in epoch_loss_components:
                epoch_loss_components[key] += loss_components[key]
            
            train_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Focal": f"{loss_components['focal']:.3f}",
                "Dice": f"{loss_components['dice']:.3f}",
                "BCE": f"{loss_components['bce']:.3f}"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {'focal': 0.0, 'dice': 0.0, 'bce': 0.0, 'total': 0.0}
        all_ious = []
        all_dices = []
        
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss, loss_components = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Accumulate validation loss components
                for key in val_loss_components:
                    val_loss_components[key] += loss_components[key]
                
                # Calculate IoU and Dice for validation
                predictions = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                for i in range(predictions.shape[0]):
                    pred = predictions[i, 0]  # Remove channel dimension
                    label = labels_np[i, 0]   # Remove channel dimension
                    
                    iou = calculate_iou_binary(pred, label)
                    dice = calculate_dice_binary(pred, label)
                    
                    all_ious.append(iou)
                    all_dices.append(dice)
                
                val_progress.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Focal": f"{loss_components['focal']:.3f}",
                    "Dice": f"{loss_components['dice']:.3f}"
                })
        
        # Calculate average losses and metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = np.mean(all_ious)
        avg_dice = np.mean(all_dices)
        
        # Calculate average loss components
        avg_train_components = {key: val / len(train_loader) for key, val in epoch_loss_components.items()}
        avg_val_components = {key: val / len(val_loader) for key, val in val_loss_components.items()}
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_iou)
        val_dices.append(avg_dice)
        loss_components_history.append({
            'epoch': epoch + 1,
            'train': avg_train_components,
            'val': avg_val_components
        })
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("New best model saved (lowest validation loss)")
        else:
            patience_counter += 1
        
        # Save best model based on IoU
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), save_dir / "best_iou_model.pth")
            print("New best IoU model saved")
        
        # Print progress with detailed metrics
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
        print(f"Loss Components (Train/Val):")
        print(f"   Focal: {avg_train_components['focal']:.4f}/{avg_val_components['focal']:.4f}")
        print(f"   Dice: {avg_train_components['dice']:.4f}/{avg_val_components['dice']:.4f}")
        print(f"   BCE: {avg_train_components['bce']:.4f}/{avg_val_components['bce']:.4f}")
        print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.1e} -> {new_lr:.1e}")
        
        # Early stopping checks
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping: No improvement for {early_stopping_patience} epochs")
            break
        
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("Early stopping: learning rate too small")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best IoU: {best_iou:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), save_dir / "final_model.pth")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_ious, val_dices, save_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_dices': val_dices,
        'loss_components_history': loss_components_history,
        'best_val_loss': best_val_loss,
        'best_iou': best_iou,
        'num_epochs_completed': len(train_losses),
        'final_lr': optimizer.param_groups[0]['lr'],
        'early_stopping_patience': early_stopping_patience,
        'final_patience_counter': patience_counter,
        'training_date': datetime.now().isoformat(),
        'task': 'dividing_line_prediction'
    }
    
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

def plot_training_history(train_losses, val_losses, val_ious, val_dices, save_dir):
    """Plot training and validation curves"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IoU curve
    ax2.plot(epochs, val_ious, label='Validation IoU', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Dice curve
    ax3.plot(epochs, val_dices, label='Validation Dice', color='orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Dice Coefficient')
    ax3.set_title('Validation Dice Coefficient')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Combined metrics
    ax4.plot(epochs, val_ious, label='IoU', color='green', linewidth=2)
    ax4.plot(epochs, val_dices, label='Dice', color='orange', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Metric Value')
    ax4.set_title('Validation Metrics Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")

def predict_and_separate(model, cell_mask, device=None, threshold=0.5, 
                        min_component_area=50, dilation_iterations=1, 
                        morphology_kernel_size=3, enable_validation=True,
                        return_visualization=False, save_visualization=None):
    """
    Enhanced complete pipeline: predict dividing line and separate cells
    
    Args:
        model: Trained dividing line U-Net model
        cell_mask: [H, W] Binary cell mask
        device: Device for inference
        threshold: Threshold for dividing line prediction
        min_component_area: Minimum area for valid components
        dilation_iterations: Number of dilation iterations for line processing
        morphology_kernel_size: Size of morphological operations kernel
        enable_validation: Whether to enable validation checks
        return_visualization: Whether to return visualization data
        save_visualization: Path to save visualization (if provided)
        
    Returns:
        separated_mask: [H, W] Separated cell mask (1=daughter, 2=mother, 0=background)
        assignments: Dict with separation details and quality metrics
        Additional returns (if requested):
        - dividing_line_pred: [H, W] Predicted dividing line
        - debug_info: Dict with debug information
        - visualization: matplotlib figure object
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    print(f"Starting enhanced cell separation pipeline...")
    print(f"   Input shape: {cell_mask.shape}")
    print(f"   Threshold: {threshold}")
    print(f"   Min component area: {min_component_area}")
    
    # Prepare input
    if len(cell_mask.shape) == 2:
        input_tensor = torch.from_numpy(cell_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        input_tensor = torch.from_numpy(cell_mask.astype(np.float32))
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    input_tensor = input_tensor.to(device)
    
    # Predict dividing line
    print("Predicting dividing line...")
    with torch.no_grad():
        dividing_line_pred = model(input_tensor)
        dividing_line_pred = dividing_line_pred[0, 0].cpu().numpy()  # Remove batch and channel dims
    
    print(f"   Line prediction range: [{dividing_line_pred.min():.3f}, {dividing_line_pred.max():.3f}]")
    print(f"   Pixels above threshold: {np.sum(dividing_line_pred > threshold):,}")
    
    # Separate cells using enhanced separator
    print("Separating cells...")
    separator = CellSeparator(
        min_component_area=min_component_area,
        dilation_iterations=dilation_iterations,
        morphology_kernel_size=morphology_kernel_size,
        enable_validation=enable_validation
    )
    
    # Get separation results with debug info
    separated_mask, assignments, debug_info = separator.separate_cells(
        cell_mask, dividing_line_pred, threshold=threshold, 
        return_assignments=True, return_debug_info=True
    )
    
    # Prepare return values
    return_values = [separated_mask, assignments]
    
    if return_visualization or save_visualization:
        # Create visualization
        print("Creating visualization...")
        fig = separator.visualize_separation_process(
            cell_mask, dividing_line_pred, separated_mask, assignments, 
            debug_info, save_path=save_visualization, 
            title=f"Cell Separation (Success: {assignments['separation_success']})"
        )
        
        if return_visualization:
            return_values.extend([dividing_line_pred, debug_info, fig])
        else:
            return_values.extend([dividing_line_pred, debug_info])
    else:
        if return_visualization:  # Only return line prediction if specifically requested
            return_values.append(dividing_line_pred)
    
    print(f"Pipeline complete!")
    return tuple(return_values) if len(return_values) > 2 else return_values[0] if len(return_values) == 1 else tuple(return_values)

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(
        description="Dividing Line Prediction U-Net Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_dividing_line_unet.py --lr 1e-4 --batch-size 8
  python train_dividing_line_unet.py --epochs 150 --device cuda
  python train_dividing_line_unet.py --focal-weight 0.6 --dice-weight 0.3
  
Loss Function Strategy:
  - Focal Loss: Handles class imbalance (few dividing line pixels vs many background)
  - Dice Loss: Ensures good overlap for thin lines
  - BCE Loss: Provides stable gradients

Post-processing:
  Use predict_and_separate() for complete pipeline:
  cell_mask → dividing_line_prediction → separated_cells → mother/daughter_assignment
        """
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='cuda', help='Device to use')
    
    parser.add_argument('--focal-weight', type=float, default=0.5, help='Focal loss weight')
    parser.add_argument('--dice-weight', type=float, default=0.3, help='Dice loss weight')
    parser.add_argument('--bce-weight', type=float, default=0.2, help='BCE loss weight')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    
    parser.add_argument('--data-dir', default='processed_dividing_line_data', help='Training data directory')
    parser.add_argument('--save-dir', default='dividing_line_models', help='Model save directory')
    parser.add_argument('--early-stopping', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--no-augmentation', default=False, action='store_true', help='Disable data augmentation')
    parser.add_argument('--split-ratio', nargs=3, type=float, default=[0.7, 0.2, 0.1], help='Train/Val/Test split')
    
    args = parser.parse_args()
    
    print("Dividing Line Prediction U-Net Training")
    print("Task: Cell mask -> Dividing line prediction -> Cell separation")
    print("=" * 60)
    
    # Check if processed data exists
    processed_data_dir = Path(args.data_dir)
    if not processed_data_dir.exists() or not (processed_data_dir / "images").exists():
        print(f"No processed data found in {args.data_dir}!")
        print("   Please run 'python data_processor_dividing_line.py' first to generate training data.")
        return
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data augmentation transforms
    if not args.no_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ])
    else:
        train_transform = None
    
    # Create datasets
    print("\nLoading datasets...")
    full_dataset = DividingLineDataset(args.data_dir, transform=train_transform, augment=not args.no_augmentation)
    
    # Split into train/val/test
    train_ratio, val_ratio, test_ratio = args.split_ratio
    train_size = int(train_ratio * len(full_dataset))
    val_size = int(val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create validation and test datasets without augmentation
    dataset_no_aug = DividingLineDataset(args.data_dir, transform=None, augment=False, verbose=False)
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(dataset_no_aug, val_indices)
    test_dataset_clean = torch.utils.data.Subset(dataset_no_aug, test_indices)
    
    # Create data loaders
    num_workers = min(8, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    val_loader = DataLoader(val_dataset_clean, batch_size=args.batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    test_loader = DataLoader(test_dataset_clean, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = DividingLineUNet(in_channels=1, out_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = CombinedDividingLineLoss(
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping,
        criterion=criterion
    )
    
    print("\nTraining completed successfully!")
    print(f"Models saved to: {args.save_dir}/")
    print(f"Best IoU: {history['best_iou']:.4f}")
    print(f"Best Loss: {history['best_val_loss']:.4f}")
    
    print(f"\nUsage tips:")
    print(f"   Use predict_and_separate() for complete pipeline")
    print(f"   Example: separated_mask, assignments = predict_and_separate(model, cell_mask)")
    print(f"   For inference: python -c \"import train_dividing_line_unet; model = train_dividing_line_unet.DividingLineUNet(); model.load_state_dict(torch.load('{args.save_dir}/best_iou_model.pth')); ...\"")

if __name__ == "__main__":
    main()
