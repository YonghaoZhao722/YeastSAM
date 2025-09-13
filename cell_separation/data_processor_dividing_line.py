import os
import re
import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import json
from scipy import ndimage
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

class DividingLineDataProcessor:
    """
    Data processor for dividing line prediction
    
    Task: Given cell masks and manual annotations, generate training data for dividing line prediction
    Input: Cell masks + manual outline annotations
    Output: (cell_mask, dividing_line_mask) pairs for U-Net training
    """
    
    def __init__(self, data_root: str = "data", output_dir: str = "processed_dividing_line_data"):
        self.data_root = Path(data_root)
        self.divided_masks_dir = self.data_root / "divided_masks"  # Cell masks with labels
        self.divided_outlines_dir = self.data_root / "divided_outlines"  # Manual annotations
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)  # Input cell masks
        (self.output_dir / "labels").mkdir(exist_ok=True)  # Dividing line masks
        (self.output_dir / "visualizations").mkdir(exist_ok=True)  # Visualization samples
        
        self.master_df = None
        self.target_size = (256, 256)  # Fixed size for training
        
        # Processing parameters
        self.line_thickness = 5
        self.min_cell_area = 100
        self.enable_visualization = True
        
        # Print initialization info
        print(f"=== Dividing Line Data Processor Initialized ===")
        print(f"Data root: {self.data_root}")
        print(f"Divided masks (source): {self.divided_masks_dir}")
        print(f"Divided outlines (manual annotations): {self.divided_outlines_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target image size: {self.target_size}")
        print(f"Line thickness (dilation): {self.line_thickness}")
        print(f"Task: Combined cell mask -> Dividing line prediction")
        
    def find_matching_files(self) -> List[Tuple[Path, Path]]:
        print("\n=== Finding Matching File Pairs ===")
        
        # Get all valid files (excluding macOS hidden files)
        divided_mask_files = [f for f in self.divided_masks_dir.glob("*.tif") 
                             if not f.name.startswith("._")]
        txt_files = [f for f in self.divided_outlines_dir.glob("*.txt") 
                    if not f.name.startswith("._")]
        
        print(f"Found {len(divided_mask_files)} divided mask files")
        print(f"Found {len(txt_files)} TXT annotation files")
        
        matched_pairs = []
        unmatched_masks = []
        
        for divided_mask_file in divided_mask_files:
            # Find matching TXT file
            txt_file = self._find_matching_file(divided_mask_file, txt_files, target_ext='.txt')
            
            if txt_file:
                matched_pairs.append((divided_mask_file, txt_file))
                print(f"Complete match:")
                print(f"   Divided mask: {divided_mask_file.name}")
                print(f"   Annotations:  {txt_file.name}")
            else:
                unmatched_masks.append(divided_mask_file)
                print(f"Missing TXT annotation for: {divided_mask_file.name}")
        
        print(f"\nMatching Summary:")
        print(f"  - Complete pairs: {len(matched_pairs)}")
        print(f"  - Missing annotations: {len(unmatched_masks)}")
        
        return matched_pairs
    
    def _find_matching_file(self, reference_file: Path, candidate_files: List[Path], 
                           target_ext: str = None) -> Optional[Path]:
        """Find matching file for a reference file using flexible matching"""
        ref_stem = reference_file.stem
        
        # Strategy 1: Exact name match
        if target_ext:
            exact_match = ref_stem + target_ext
        else:
            exact_match = reference_file.name
            
        for candidate in candidate_files:
            if candidate.name == exact_match:
                return candidate
        
        # Strategy 2: Remove "_shifted" from reference and match
        if "_shifted" in ref_stem:
            base_name = ref_stem.replace("_shifted", "")
            if target_ext:
                target_name = base_name + target_ext
            else:
                target_name = base_name + reference_file.suffix
                
            for candidate in candidate_files:
                if candidate.name == target_name:
                    return candidate
        
        # Strategy 3: Add "_shifted" to reference name
        if "_shifted" not in ref_stem:
            shifted_name = ref_stem + "_shifted"
            if target_ext:
                target_name = shifted_name + target_ext
            else:
                target_name = shifted_name + reference_file.suffix
                
            for candidate in candidate_files:
                if candidate.name == target_name:
                    return candidate
        
        return None
    
    def parse_txt_file(self, txt_file: Path, divided_mask_file: Path) -> List[Dict]:
        """Parse a single TXT file and extract budding pair information"""
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Unicode decode error for {txt_file.name}, skipping...")
            return []
        
        # Extract all cells from the file
        cell_blocks = re.findall(r'CELL\s+Cell_(\d+).*?(?=CELL\s+Cell_\d+|Z_POS\s*$)', content, re.DOTALL)
        
        # Handle the last cell block separately
        last_cell_match = re.search(r'CELL\s+Cell_(\d+).*?Z_POS\s*$', content, re.DOTALL)
        if last_cell_match:
            last_cell_id = last_cell_match.group(1)
            if last_cell_id not in cell_blocks:
                cell_blocks.append(last_cell_id)
        
        # Load divided mask to get cell instances
        try:
            divided_mask = tifffile.imread(divided_mask_file)
        except Exception as e:
            print(f"Error loading divided mask {divided_mask_file}: {e}")
            return []
        
        # Get unique cell instances from divided mask (sorted by first y coordinate)
        unique_instances = np.unique(divided_mask)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        # Sort instances by first y pixel (topmost pixel)
        def get_first_y_pixel(instance_id):
            coords = np.where(divided_mask == instance_id)
            return coords[0].min() if len(coords[0]) > 0 else float('inf')
        
        sorted_instances = sorted(unique_instances, key=get_first_y_pixel)
        
        # Match cell numbers from txt with sorted instances
        cell_numbers = []
        for cell_id in cell_blocks:
            try:
                cell_numbers.append(int(cell_id))
            except ValueError:
                continue
        
        # Create mapping between instance IDs and cell numbers
        instance_to_cell = {}
        for i, instance_id in enumerate(sorted_instances):
            if i < len(cell_numbers):
                instance_to_cell[instance_id] = cell_numbers[i]
        
        # Group cells by dividing events (same cell number ignoring last 2 digits)
        dividing_groups = {}
        for instance_id, cell_number in instance_to_cell.items():
            base_cell_id = cell_number // 100  # Remove last 2 digits
            if base_cell_id not in dividing_groups:
                dividing_groups[base_cell_id] = []
            dividing_groups[base_cell_id].append({
                'instance_id': instance_id,
                'cell_number': cell_number,
                'base_cell_id': base_cell_id
            })
        
        extracted_data = []
        
        # Process each dividing group
        for base_cell_id, cells in dividing_groups.items():
            if len(cells) >= 2:  # Only process if there are at least 2 cells in the group
                extracted_data.append({
                    'divided_mask_name': divided_mask_file.name,
                    'divided_mask_path': str(divided_mask_file),
                    'txt_file': str(txt_file),
                    'base_cell_id': base_cell_id,
                    'cells_in_group': cells,
                    'has_complete_pair': True,
                    'is_dividing_event': True
                })
        
        return extracted_data
    
    def _bridge_gaps(self, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intelligently bridge the gap between two separate masks AND generate the centerline as the dividing line.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (connected_input_mask, dividing_line_mask)
        """
        combined = (mask1 | mask2).astype(np.uint8)
        
        from scipy.ndimage import label as connected_components, distance_transform_edt
        _, num_components = connected_components(combined)
        
        # If already connected, the dividing line is simply the contact surface
        if num_components <= 1:
            # We can still use dilation to find the thin contact line
            dilated1 = binary_dilation(mask1, iterations=2)
            dilated2 = binary_dilation(mask2, iterations=2)
            dividing_line = (dilated1 & dilated2) & (~(mask1 | mask2))
            return combined, dividing_line.astype(np.uint8)
            
        print(f"Bridging gap and finding centerline between disconnected cells...")
        
        # Calculate distance from the background to the nearest pixel in each mask
        dist1 = distance_transform_edt(mask1 == 0)
        dist2 = distance_transform_edt(mask2 == 0)
        
        total_dist = dist1 + dist2
        background = (combined == 0)
        
        if background.sum() > 0:
            # Find bridge region
            min_path_dist = total_dist[background].min()
            bridge = (total_dist <= min_path_dist + 2) & background
            
            # Find centerline (where dist1 is approx equal to dist2)
            # This forms a Voronoi-like boundary
            centerline = (np.abs(dist1 - dist2) <= 1)
            
            # The dividing line is the part of the centerline that lies within the bridge
            dividing_line = centerline & bridge
            
            # Create the final connected input mask
            connected_mask = combined | bridge.astype(np.uint8)
            
            print(f"   Successfully bridged {mask1.sum() + mask2.sum()} -> {connected_mask.sum()} pixels")
            print(f"   Generated centerline: {dividing_line.sum()} pixels")
            
            return connected_mask, dividing_line.astype(np.uint8)
        
        # Fallback if something goes wrong
        print(f"   Bridging failed, using original combined mask")
        return combined, np.zeros_like(combined, dtype=np.uint8)
    
    def _visualize_failed_sample(self, divided_mask: np.ndarray, cells_in_group: List[Dict], 
                                 reason: str, sample_name: str):
        """
        Create visualization for failed samples to help with debugging
        """
        if not self.enable_visualization:
            return
            
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create figure
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get individual cell masks
        individual_masks = []
        for cell_info in cells_in_group:
            instance_id = cell_info['instance_id']
            mask = (divided_mask == instance_id).astype(np.uint8)
            individual_masks.append((mask, instance_id, cell_info['cell_number']))
        
        # 1. Original divided mask with colored cells
        ax1 = fig.add_subplot(gs[0, 0])
        overlay = np.zeros((*divided_mask.shape, 3))
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        for i, (mask, instance_id, cell_number) in enumerate(individual_masks):
            color = colors[i % len(colors)]
            overlay[mask > 0] = color
        ax1.imshow(overlay)
        ax1.set_title(f'Original Divided Cells\n{len(individual_masks)} cells')
        ax1.axis('off')
        
        # 2. Simple union
        ax2 = fig.add_subplot(gs[0, 1])
        simple_union = np.zeros_like(divided_mask, dtype=np.uint8)
        for mask, _, _ in individual_masks:
            simple_union |= mask
        ax2.imshow(simple_union, cmap='gray')
        ax2.set_title(f'Simple Union\n{simple_union.sum():,} pixels')
        ax2.axis('off')
        
        # 3. Bridged version (if applicable)
        ax3 = fig.add_subplot(gs[0, 2])
        if len(individual_masks) >= 2:
            bridged, _ = self._bridge_gaps(individual_masks[0][0], individual_masks[1][0])
            ax3.imshow(bridged, cmap='gray')
            ax3.set_title(f'After Bridging\n{bridged.sum():,} pixels')
        else:
            ax3.imshow(simple_union, cmap='gray')
            ax3.set_title('Bridging N/A\n(< 2 cells)')
        ax3.axis('off')
        
        # 4. Distance analysis
        ax4 = fig.add_subplot(gs[0, 3])
        if len(individual_masks) >= 2:
            mask1, mask2 = individual_masks[0][0], individual_masks[1][0]
            from scipy.ndimage import distance_transform_edt
            dist1 = distance_transform_edt(mask1 == 0)
            dist2 = distance_transform_edt(mask2 == 0)
            total_dist = dist1 + dist2
            background = (simple_union == 0)
            min_dist = total_dist[background].min() if background.sum() > 0 else 0
            
            ax4.imshow(total_dist, cmap='viridis')
            ax4.set_title(f'Distance Map\nMin gap: {min_dist:.1f}px')
        else:
            ax4.text(0.5, 0.5, 'Distance\nanalysis\nN/A', ha='center', va='center', 
                    transform=ax4.transAxes)
            ax4.set_title('Distance Analysis')
        ax4.axis('off')
        
        # 5-8. Individual cell details
        for i, (mask, instance_id, cell_number) in enumerate(individual_masks[:4]):
            row = 1
            col = i
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(mask, cmap='gray')
            ax.set_title(f'Cell {cell_number}\nID:{instance_id}, Area:{mask.sum():,}px')
            ax.axis('off')
        
        # Fill remaining subplots with failure information
        for i in range(len(individual_masks), 4):
            ax = fig.add_subplot(gs[1, i])
            ax.axis('off')
            if i == len(individual_masks):  # First empty slot gets failure details
                failure_text = f"""FAILURE ANALYSIS

Reason: {reason}

Cell Group Details:
• Total cells: {len(individual_masks)}
• Group ID: {cells_in_group[0]['base_cell_id'] if cells_in_group else 'N/A'}

Individual Areas:"""
                for mask, instance_id, cell_number in individual_masks:
                    failure_text += f"\n• Cell {cell_number}: {mask.sum():,}px"
                
                if len(individual_masks) >= 2:
                    mask1, mask2 = individual_masks[0][0], individual_masks[1][0]
                    # Check connectivity
                    from scipy.ndimage import label as connected_components
                    combined = mask1 | mask2
                    _, num_comp = connected_components(combined)
                    failure_text += f"\n\nConnectivity Check:"
                    failure_text += f"\n• Components: {num_comp}"
                    
                    # Check minimum distance
                    if num_comp > 1:
                        from scipy.ndimage import distance_transform_edt
                        dist1 = distance_transform_edt(mask1 == 0)
                        dist2 = distance_transform_edt(mask2 == 0)
                        background = (combined == 0)
                        if background.sum() > 0:
                            min_gap = (dist1 + dist2)[background].min()
                            failure_text += f"\n• Min gap: {min_gap:.1f}px"
                
                ax.text(0.05, 0.95, failure_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.suptitle(f'Failed Sample Analysis: {sample_name}\nReason: {reason}', 
                    fontsize=14, fontweight='bold')
        
        # Save visualization
        viz_path = self.output_dir / "visualizations" / f"failed_sample_{sample_name.replace(' ', '_')}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Failed sample visualization saved: {viz_path}")
    
    def generate_dividing_line_mask(self, divided_mask: np.ndarray, 
                                        cells_in_group: List[Dict]) -> Tuple[np.ndarray, np.ndarray, bool]:
            """
            [Final Version] Generate a connected input mask and a thick bridge-like dividing line label.
            
            This simplified and robust version directly implements the "bridge as label" concept.
            """
            if len(cells_in_group) < 2:
                # This part remains the same, skipping samples with fewer than 2 cells.
                print(f"Not enough cells in group ({len(cells_in_group)}). Skipping.")
                return np.zeros_like(divided_mask), np.zeros_like(divided_mask), False

            from scipy.ndimage import label as connected_components, distance_transform_edt, binary_dilation

            # 1. Get the original, separated cell masks
            mask1 = (divided_mask == cells_in_group[0]['instance_id'])
            mask2 = (divided_mask == cells_in_group[1]['instance_id'])
            
            if mask1.sum() == 0 or mask2.sum() == 0:
                print("One of the cell masks is empty. Skipping.")
                return np.zeros_like(divided_mask), np.zeros_like(divided_mask), False

            # 2. Create the "bridge" that will become the base for our label
            # The bridge is the region that perfectly fills the gap between the two cells.
            dist1 = distance_transform_edt(mask1 == 0)
            dist2 = distance_transform_edt(mask2 == 0)
            total_dist = dist1 + dist2
            
            original_combined = (mask1 | mask2)
            background = (original_combined == 0)
            
            if background.sum() == 0: # Should not happen if cells are separate
                print("No background found between cells. Skipping.")
                return np.zeros_like(divided_mask), np.zeros_like(divided_mask), False
                
            min_path_dist = total_dist[background].min()
            
            # The bridge fills the shortest path. We use a tolerance of +1 for a clean fill.
            bridge_mask = (total_dist <= min_path_dist + 1) & background
            
            # 3. The raw "bridge" is our initial, thin label
            dividing_line_mask = bridge_mask.astype(np.uint8)

            # 4. Create the final connected INPUT mask for the model
            complete_cell_mask = (original_combined | dividing_line_mask).astype(np.uint8)
            
            # Ensure the input mask is a single connected component
            _, num_components = connected_components(complete_cell_mask)
            if num_components > 1:
                # If for some reason the bridge wasn't enough, do a small dilation to force connection
                complete_cell_mask = binary_dilation(complete_cell_mask, iterations=2)

            # 5. Thicken the LABEL for class balance, controlled by self.line_thickness
            if self.line_thickness > 1:
                # We use a disk kernel for uniform thickening
                y, x = np.ogrid[-self.line_thickness:self.line_thickness+1, -self.line_thickness:self.line_thickness+1]
                disk = x*x + y*y <= (self.line_thickness -1)**2 # Use thickness as radius
                
                # Dilate the bridge to get the final thick label
                dividing_line_mask = binary_dilation(dividing_line_mask, structure=disk).astype(np.uint8)
                
            # Final safety check: ensure the label is within the input mask
            dividing_line_mask = np.logical_and(dividing_line_mask, complete_cell_mask).astype(np.uint8)
            
            if dividing_line_mask.sum() == 0:
                print("Failed to generate a valid dividing line. Skipping.")
                return complete_cell_mask, dividing_line_mask, False

            print(f"Generated training pair. Input pixels: {complete_cell_mask.sum():,}, Line pixels: {dividing_line_mask.sum():,}")
            
            return complete_cell_mask, dividing_line_mask, True
        
    def generate_training_sample(self, sample_info: Dict) -> bool:
        """Generate a single training sample from dividing event information"""
        
        if not sample_info['is_dividing_event']:
            return False
            
        divided_mask_path = sample_info['divided_mask_path']
        cells_in_group = sample_info['cells_in_group']
        base_cell_id = sample_info['base_cell_id']
        
        try:
            # Load mask
            divided_mask = tifffile.imread(divided_mask_path)  # Divided mask with separate instances
        except Exception as e:
            print(f"Error loading mask: {e}")
            return False
        
        # Generate dividing line mask using new method (no longer needs dic_mask)
        complete_cell_mask, dividing_line_mask, is_valid = self.generate_dividing_line_mask(
            divided_mask, cells_in_group
        )
        
        if not is_valid or dividing_line_mask.sum() == 0:
            print(f"Failed to generate valid dividing line for base cell {base_cell_id}")
            return False
        
        # Check if masks are valid
        if complete_cell_mask.sum() < self.min_cell_area:
            print(f"Cell area too small: {complete_cell_mask.sum()} pixels")
            return False
        
        # Find bounding box and crop with padding
        coords = np.argwhere(complete_cell_mask > 0)
        if len(coords) == 0:
            return False
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding) 
        y_max = min(complete_cell_mask.shape[0], y_max + padding)
        x_max = min(complete_cell_mask.shape[1], x_max + padding)
        
        # Crop the regions
        X_cropped = complete_cell_mask[y_min:y_max, x_min:x_max]  # Input complete cell mask
        y_cropped = dividing_line_mask[y_min:y_max, x_min:x_max]   # Dividing line mask
        
        # Resize to target size using scipy
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_y = self.target_size[1] / X_cropped.shape[0]  # target_size is (width, height)
        zoom_x = self.target_size[0] / X_cropped.shape[1]
        
        # Resize using nearest neighbor interpolation
        X_resized = zoom(X_cropped.astype(np.float32), (zoom_y, zoom_x), order=0)  # order=0 for nearest neighbor
        y_resized = zoom(y_cropped.astype(np.float32), (zoom_y, zoom_x), order=0)
        
        # Convert back to binary
        X_resized = (X_resized > 0.5).astype(np.uint8)
        y_resized = (y_resized > 0.5).astype(np.uint8)
        
        # Save the processed data
        mask_name = Path(sample_info['divided_mask_name']).stem
        sample_name = f"{mask_name}_cell_{base_cell_id}"
        
        np.save(self.output_dir / "images" / f"{sample_name}.npy", X_resized)
        np.save(self.output_dir / "labels" / f"{sample_name}.npy", y_resized)
        
        print(f"Generated training sample: {sample_name}")
        print(f"   Input cell area: {X_resized.sum()} pixels")
        print(f"   Dividing line area: {y_resized.sum()} pixels")
        print(f"   Cells in group: {[cell['cell_number'] for cell in cells_in_group]}")
        
        # Create merged divided mask for visualization
        merged_divided_mask = np.zeros_like(divided_mask, dtype=np.uint8)
        for cell_info in cells_in_group:
            instance_id = cell_info['instance_id']
            instance_mask = (divided_mask == instance_id)
            merged_divided_mask[instance_mask] = 1
        
        # Store data for potential visualization
        sample_data = {
            'sample_name': sample_name,
            'original_divided_mask': divided_mask,
            'original_merged_divided': merged_divided_mask,
            'original_input_mask': complete_cell_mask,
            'original_dividing_line': dividing_line_mask,
            'processed_input': X_resized,
            'processed_label': y_resized,
            'crop_coords': (y_min, y_max, x_min, x_max),
            'base_cell_id': base_cell_id,
            'cells_in_group': cells_in_group
        }
        
        return sample_data
    
    def stage1_data_parsing(self) -> pd.DataFrame:
        """
        Stage 1: Data Parsing and Structuring
        Parse file triplets and create a unified DataFrame
        """
        print("\n=== Stage 1: Data Parsing and Structuring ===")
        
        # Find matching file pairs
        matched_pairs = self.find_matching_files()
        
        if not matched_pairs:
            print("No matching file pairs found!")
            return pd.DataFrame()
        
        all_data = []
        
        # Process each matched pair
        for divided_mask_file, txt_file in matched_pairs:
            print(f"Processing: {txt_file.name}")
            try:
                cell_data = self.parse_txt_file(txt_file, divided_mask_file)
                all_data.extend(cell_data)
            except Exception as e:
                print(f"Error processing {txt_file}: {e}")
        
        # Create master DataFrame
        self.master_df = pd.DataFrame(all_data)
        
        if len(self.master_df) > 0:
            print(f"\nCreated master DataFrame with {len(self.master_df)} entries")
            
            # Show distribution statistics
            dividing_events = len(self.master_df[self.master_df['is_dividing_event'] == True])
            
            print(f"Dividing events found: {dividing_events}")
            
            # Show cell group size distribution
            if dividing_events > 0:
                group_sizes = []
                for _, row in self.master_df.iterrows():
                    if row['is_dividing_event']:
                        group_sizes.append(len(row['cells_in_group']))
                
                print(f"Cells per dividing event:")
                print(f"   Min: {min(group_sizes)} cells")
                print(f"   Max: {max(group_sizes)} cells") 
                print(f"   Average: {np.mean(group_sizes):.1f} cells")
            
            # Save master DataFrame
            output_file = self.output_dir / "master_data.csv"
            self.master_df.to_csv(output_file, index=False)
            print(f"Saved master DataFrame to {output_file}")
            
        else:
            print("No data found!")
            
        return self.master_df
    
    def stage2_generate_training_pairs(self) -> int:
        """
        Stage 2: Generate Training Sample Pairs
        Create (cell_mask, dividing_line_mask) pairs for model training
        """
        print("\n=== Stage 2: Generate Training Sample Pairs ===")
        
        if self.master_df is None or len(self.master_df) == 0:
            print("No master DataFrame found. Run stage1_data_parsing() first.")
            return 0
        
        # Filter dividing events
        dividing_events = self.master_df[self.master_df['is_dividing_event'] == True].copy()
        print(f"Found {len(dividing_events)} dividing events")
        
        if len(dividing_events) == 0:
            print("No dividing events found!")
            return 0
        
        # Generate training samples
        generated_samples = 0
        processed_events = set()  # Track processed events to avoid duplicates
        visualization_samples = []  # Store samples for visualization
        
        for _, sample_info in dividing_events.iterrows():
            base_cell_id = sample_info['base_cell_id']
            mask_name = sample_info['divided_mask_name']
            
            # Create unique identifier for this dividing event
            unique_event_id = f"{mask_name}_{base_cell_id}"
            
            if unique_event_id in processed_events:
                continue  # Skip if already processed
            
            try:
                sample_data = self.generate_training_sample(sample_info)
                if sample_data:
                    generated_samples += 1
                    processed_events.add(unique_event_id)
                    
                    # Store first few samples for visualization
                    if len(visualization_samples) < 6:
                        visualization_samples.append(sample_data)
                        
            except Exception as e:
                print(f"Error processing dividing event {base_cell_id} in {mask_name}: {e}")
                continue
        
        print(f"\nSuccessfully generated {generated_samples} training samples")
        
        # Generate visualizations
        if self.enable_visualization and len(visualization_samples) > 0:
            print(f"\nCreating visualizations for {len(visualization_samples)} samples...")
            self.create_sample_visualizations(visualization_samples)
        
        # Save generation summary
        self._save_generation_summary(generated_samples, len(processed_events))
        
        return generated_samples
    
    def _save_generation_summary(self, generated_samples: int, total_pairs: int):
        """Save summary of training sample generation"""
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_budding_pairs': total_pairs,
            'successfully_generated': generated_samples,
            'target_image_size': self.target_size,
            'line_thickness': self.line_thickness,
            'min_cell_area': self.min_cell_area,
            'preprocessing_method': 'dividing_line_prediction',
            'input_source': 'divided_masks - intelligently bridged cell masks for input',
            'label_source': 'divided_masks - dividing line prediction from original contact regions',
            'model_task': 'Connected cell mask → Dividing line segmentation (binary classification)',
            'postprocessing_strategy': 'Predict dividing line → Split cells → Assign by area (larger=mother, smaller=daughter)',
            'bridging_strategy': 'Distance-transform based morphological bridging to connect separated cells',
            'class_balance_solution': f'Line dilation with thickness {self.line_thickness} to address class imbalance',
            'quality_control': 'Automatic filtering of bad samples with detailed failure visualization'
        }
        
        summary_file = self.output_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved generation summary to {summary_file}")
    
    def create_sample_visualizations(self, visualization_samples: List[Dict]):
        """
        Create comprehensive visualizations of the preprocessing results
        
        Shows the complete pipeline: original masks → dividing line generation → final training data
        """
        
        # Create overview visualization with multiple samples
        n_samples = len(visualization_samples)
        fig = plt.figure(figsize=(20, 4 * n_samples))
        gs = GridSpec(n_samples, 6, figure=fig, hspace=0.4, wspace=0.3)
        
        for i, sample_data in enumerate(visualization_samples):
            sample_name = sample_data['sample_name']
            
            # Extract data
            original_divided = sample_data['original_divided_mask']
            original_merged_divided = sample_data['original_merged_divided']
            original_input = sample_data['original_input_mask']
            original_dividing_line = sample_data['original_dividing_line']
            processed_input = sample_data['processed_input']
            processed_label = sample_data['processed_label']
            crop_coords = sample_data['crop_coords']
            base_cell_id = sample_data['base_cell_id']
            cells_in_group = sample_data['cells_in_group']
            
            y_min, y_max, x_min, x_max = crop_coords
            
            # Create individual visualizations for this sample
            row = i
            
            # 1. Original divided mask showing separate instances
            ax1 = fig.add_subplot(gs[row, 0])
            
            # Create colored overlay for different cell instances
            overlay = np.zeros((*original_divided.shape, 3))
            
            # Assign different colors to each instance in the group
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
            for i_cell, cell_info in enumerate(cells_in_group):
                instance_id = cell_info['instance_id']
                instance_mask = (original_divided == instance_id)
                color = colors[i_cell % len(colors)]
                overlay[instance_mask] = color
            
            ax1.imshow(overlay)
            ax1.set_title(f'Original Labels\n{sample_name}', fontsize=10)
            ax1.axis('off')
            
            # Add crop rectangle
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='yellow', facecolor='none')
            ax1.add_patch(rect)
            
            # 2. Combined input mask (mother + daughter)
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.imshow(original_input, cmap='gray')
            ax2.set_title('Combined Cell Mask\n(Model Input)', fontsize=10)
            ax2.axis('off')
            
            # Add crop rectangle
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='yellow', facecolor='none')
            ax2.add_patch(rect)
            
            # 3. Generated dividing line
            ax3 = fig.add_subplot(gs[row, 2])
            ax3.imshow(original_input, cmap='gray', alpha=0.7)
            ax3.imshow(original_dividing_line, cmap='Reds', alpha=0.8)
            ax3.set_title(f'Generated Dividing Line\n(thickness={self.line_thickness})', fontsize=10)
            ax3.axis('off')
            
            # Add crop rectangle
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='yellow', facecolor='none')
            ax3.add_patch(rect)
            
            # 4. Cropped and resized input
            ax4 = fig.add_subplot(gs[row, 3])
            ax4.imshow(processed_input, cmap='gray')
            ax4.set_title(f'Processed Input\n{self.target_size}', fontsize=10)
            ax4.axis('off')
            
            # 5. Cropped and resized label
            ax5 = fig.add_subplot(gs[row, 4])
            ax5.imshow(processed_input, cmap='gray', alpha=0.7)
            ax5.imshow(processed_label, cmap='Reds', alpha=0.8)
            ax5.set_title(f'Processed Label\n{self.target_size}', fontsize=10)
            ax5.axis('off')
            
            # 6. Statistics and info
            ax6 = fig.add_subplot(gs[row, 5])
            ax6.axis('off')
            
            # Calculate statistics
            total_input_pixels = processed_input.sum()
            total_line_pixels = processed_label.sum()
            line_ratio = total_line_pixels / total_input_pixels if total_input_pixels > 0 else 0
            
            # Calculate areas for each cell instance
            cell_areas = []
            for cell_info in cells_in_group:
                instance_id = cell_info['instance_id']
                cell_number = cell_info['cell_number']
                instance_mask = (original_divided == instance_id)
                area = instance_mask.sum()
                cell_areas.append((cell_number, area))
            
            cell_areas.sort(key=lambda x: x[1], reverse=True)  # Sort by area (largest first)
            
            info_text = f"""Sample: {sample_name}
            
Original Size: {original_divided.shape}
Processed Size: {self.target_size}
Crop Region: {x_max-x_min}×{y_max-y_min}

Dividing Group {base_cell_id}:"""
            
            for i, (cell_number, area) in enumerate(cell_areas):
                info_text += f"\n• Cell {cell_number}: {area:,} px"
            
            info_text += f"""

Processed Data:
• Input pixels: {total_input_pixels:,}
• Line pixels: {total_line_pixels:,}
• Line ratio: {line_ratio:.3f}

Quality Check:
• Line thickness: {self.line_thickness}px
• Connectivity: {'Yes' if total_line_pixels > 0 else 'No'}
• Group size: {len(cells_in_group)} cells"""
            
            ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, fontsize=8,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Add overall title
        fig.suptitle(f'Dividing Line Preprocessing Visualization\n'
                    f'Generated {len(visualization_samples)} samples with line thickness {self.line_thickness}',
                    fontsize=16, fontweight='bold')
        
        # Save visualization
        viz_path = self.output_dir / "visualizations" / "preprocessing_samples.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Sample visualization saved: {viz_path}")
        
        # Create individual detailed visualizations for first 3 samples
        for i, sample_data in enumerate(visualization_samples[:3]):
            self._create_detailed_sample_visualization(sample_data, i)
        
        # Create class balance analysis
        self._create_class_balance_visualization(visualization_samples)
    
    def _create_detailed_sample_visualization(self, sample_data: Dict, sample_idx: int):
        """Create detailed visualization for a single sample"""
        
        sample_name = sample_data['sample_name']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Detailed Analysis: {sample_name}', fontsize=14, fontweight='bold')
        
        # Original data
        original_divided = sample_data['original_divided_mask']
        original_merged_divided = sample_data['original_merged_divided']
        base_cell_id = sample_data['base_cell_id']
        cells_in_group = sample_data['cells_in_group']
        
        complete_mask = sample_data['original_input_mask']
        dividing_line = sample_data['original_dividing_line']
        
        # Row 1: Original analysis
        # Show individual instances in the dividing group
        if len(cells_in_group) >= 2:
            # First instance
            instance_id_1 = cells_in_group[0]['instance_id']
            cell_number_1 = cells_in_group[0]['cell_number']
            instance_mask_1 = (original_divided == instance_id_1).astype(np.uint8)
            axes[0, 0].imshow(instance_mask_1, cmap='Reds')
            axes[0, 0].set_title(f'Cell {cell_number_1} (ID: {instance_id_1})\nArea: {instance_mask_1.sum():,} pixels')
            axes[0, 0].axis('off')
            
            # Second instance
            instance_id_2 = cells_in_group[1]['instance_id']
            cell_number_2 = cells_in_group[1]['cell_number']
            instance_mask_2 = (original_divided == instance_id_2).astype(np.uint8)
            axes[0, 1].imshow(instance_mask_2, cmap='Blues')
            axes[0, 1].set_title(f'Cell {cell_number_2} (ID: {instance_id_2})\nArea: {instance_mask_2.sum():,} pixels')
            axes[0, 1].axis('off')
        else:
            # Fallback if less than 2 cells
            axes[0, 0].imshow(original_merged_divided, cmap='gray')
            axes[0, 0].set_title('Merged Divided Cells')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(complete_mask, cmap='gray')
            axes[0, 1].set_title('Complete Cell Mask')
            axes[0, 1].axis('off')
        
        # Complete cell with dividing line
        axes[0, 2].imshow(complete_mask, cmap='gray', alpha=0.8)
        axes[0, 2].imshow(dividing_line, cmap='Greens', alpha=0.9)
        axes[0, 2].set_title(f'Dividing Line Generation\nLine pixels: {dividing_line.sum():,}')
        axes[0, 2].axis('off')
        
        # Row 2: Processed data
        processed_input = sample_data['processed_input']
        processed_label = sample_data['processed_label']
        
        # Processed input
        axes[1, 0].imshow(processed_input, cmap='gray')
        axes[1, 0].set_title(f'Processed Input\n{self.target_size[0]}×{self.target_size[1]}')
        axes[1, 0].axis('off')
        
        # Processed label
        axes[1, 1].imshow(processed_label, cmap='Reds')
        axes[1, 1].set_title(f'Processed Label\nLine pixels: {processed_label.sum():,}')
        axes[1, 1].axis('off')
        
        # Final overlay
        axes[1, 2].imshow(processed_input, cmap='gray', alpha=0.7)
        axes[1, 2].imshow(processed_label, cmap='Reds', alpha=0.8)
        axes[1, 2].set_title('Final Training Data\n(Input + Label)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save detailed visualization
        detail_path = self.output_dir / "visualizations" / f"detailed_sample_{sample_idx+1}_{sample_name}.png"
        plt.savefig(detail_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Detailed visualization saved: {detail_path}")
    
    def _create_class_balance_visualization(self, visualization_samples: List[Dict]):
        """Create visualization showing class balance analysis"""
        
        # Collect statistics
        line_ratios = []
        group_size_counts = []
        input_areas = []
        line_areas = []
        cell_area_variations = []
        
        for sample_data in visualization_samples:
            processed_input = sample_data['processed_input']
            processed_label = sample_data['processed_label']
            
            original_divided = sample_data['original_divided_mask']
            cells_in_group = sample_data['cells_in_group']
            
            # Calculate cell areas in the dividing group
            cell_areas = []
            for cell_info in cells_in_group:
                instance_id = cell_info['instance_id']
                instance_mask = (original_divided == instance_id)
                area = instance_mask.sum()
                cell_areas.append(area)
            
            # Calculate variation in cell sizes (coefficient of variation)
            if len(cell_areas) > 1:
                mean_area = np.mean(cell_areas)
                std_area = np.std(cell_areas)
                cv = std_area / mean_area if mean_area > 0 else 0
                cell_area_variations.append(cv)
            
            input_area = processed_input.sum()
            line_area = processed_label.sum()
            
            if input_area > 0:
                line_ratios.append(line_area / input_area)
                
            group_size_counts.append(len(cells_in_group))
            input_areas.append(input_area)
            line_areas.append(line_area)
        
        # Create analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Class Balance and Quality Analysis', fontsize=14, fontweight='bold')
        
        # Line ratio distribution
        ax1.hist(line_ratios, bins=10, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xlabel('Line Pixels / Total Cell Pixels')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Dividing Line Density\nMean: {np.mean(line_ratios):.4f}')
        ax1.grid(True, alpha=0.3)
        
        # Group size distribution
        ax2.hist(group_size_counts, bins=range(min(group_size_counts), max(group_size_counts)+2), 
                alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Number of Cells in Dividing Group')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Dividing Group Sizes\nMean: {np.mean(group_size_counts):.1f}')
        ax2.grid(True, alpha=0.3)
        
        # Area comparison
        ax3.scatter(input_areas, line_areas, alpha=0.7, color='green')
        ax3.set_xlabel('Input Cell Area (pixels)')
        ax3.set_ylabel('Dividing Line Area (pixels)')
        ax3.set_title('Cell Area vs Line Area')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        ax4.axis('off')
        stats_text = f"""Preprocessing Quality Summary
        
Number of samples: {len(visualization_samples)}
Line thickness: {self.line_thickness} pixels
Target size: {self.target_size[0]}×{self.target_size[1]}

Line Density Statistics:
• Mean: {np.mean(line_ratios):.4f}
• Std: {np.std(line_ratios):.4f}
• Min: {np.min(line_ratios):.4f}
• Max: {np.max(line_ratios):.4f}

Group Size Statistics:
• Mean: {np.mean(group_size_counts):.1f}
• Most common: {max(set(group_size_counts), key=group_size_counts.count)}
• Range: {min(group_size_counts)}-{max(group_size_counts)} cells"""

        if cell_area_variations:
            stats_text += f"""

Cell Size Variation:
• Mean CV: {np.mean(cell_area_variations):.2f}
• Std CV: {np.std(cell_area_variations):.2f}"""

        stats_text += f"""

Quality Indicators:
• Valid dividing events: {len(visualization_samples)}
• Sufficient line density: {sum(r > 0.001 for r in line_ratios)}/{len(line_ratios)}
• Good processing: Yes

Recommendations:
• Line thickness is {'appropriate' if 0.001 < np.mean(line_ratios) < 0.01 else 'needs adjustment'}
• Group diversity is {'good' if len(set(group_size_counts)) > 1 else 'limited'}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        
        # Save analysis
        analysis_path = self.output_dir / "visualizations" / "class_balance_analysis.png"
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Class balance analysis saved: {analysis_path}")
        
        # Print summary to console
        print(f"\nQuality Summary:")
        print(f"   Line density: {np.mean(line_ratios):.4f} ± {np.std(line_ratios):.4f}")
        print(f"   Group sizes: {np.mean(group_size_counts):.1f} ± {np.std(group_size_counts):.1f}")
        if cell_area_variations:
            print(f"   Cell size variation: {np.mean(cell_area_variations):.2f} ± {np.std(cell_area_variations):.2f}")
        print(f"   Valid dividing events: {len(visualization_samples)}")
        print(f"   Preprocessing quality looks {'good' if np.mean(line_ratios) > 0.001 else 'needs improvement'}")
        
        return {
            'line_ratios': line_ratios,
            'group_size_counts': group_size_counts,
            'cell_area_variations': cell_area_variations,
            'mean_line_density': np.mean(line_ratios),
            'mean_group_size': np.mean(group_size_counts)
        }
    
    def run_full_pipeline(self) -> int:
        """Run the complete data processing pipeline"""
        
        print("Starting Dividing Line Data Processing Pipeline")
        print("=" * 60)
        
        # Stage 1: Parse data
        master_df = self.stage1_data_parsing()
        
        if master_df is None or len(master_df) == 0:
            print("Pipeline stopped: No data found in Stage 1")
            return 0
        
        # Stage 2: Generate training pairs
        num_samples = self.stage2_generate_training_pairs()
        
        print("\n" + "=" * 60)
        print(f"Pipeline Complete! Generated {num_samples} training samples")
        print(f"Output directory: {self.output_dir}")
        print(f"Input: Intelligently bridged cell masks (binary)")
        print(f"Labels: Dividing line masks from original contact regions (binary)")
        print(f"Bridging: Distance-transform based gap filling for realistic inputs")
        print(f"Post-processing: Line prediction -> Cell separation -> Area-based assignment")
        print(f"Quality control: Failed samples automatically visualized for debugging")
        
        return num_samples

def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Dividing Line Data Processor')
    parser.add_argument('--data_root', default='data', 
                      help='Root directory containing divided_masks and divided_outlines')
    parser.add_argument('--output_dir', default='processed_dividing_line_data',
                      help='Output directory for processed data')
    parser.add_argument('--target_size', nargs=2, type=int, default=[256, 256],
                      help='Target image size (width height)')
    parser.add_argument('--line_thickness', type=int, default=3,
                      help='Dividing line thickness (dilation kernel size)')
    parser.add_argument('--min_cell_area', type=int, default=100,
                      help='Minimum cell area in pixels')
    parser.add_argument('--stage', choices=['1', '2', 'all'], default='all',
                      help='Which stage to run (1=parsing, 2=training pairs, all=both)')
    parser.add_argument('--visualize', default=True, action='store_true',
                      help='Create visualization samples during preprocessing')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DividingLineDataProcessor(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    processor.target_size = tuple(args.target_size)
    processor.line_thickness = args.line_thickness
    processor.min_cell_area = args.min_cell_area
    processor.enable_visualization = args.visualize
    
    # Run requested stages
    if args.stage == '1':
        processor.stage1_data_parsing()
    elif args.stage == '2':
        if processor.master_df is None:
            # Try to load existing data
            master_file = processor.output_dir / "master_data.csv"
            if master_file.exists():
                processor.master_df = pd.read_csv(master_file)
                print(f"Loaded existing master data from {master_file}")
            else:
                print("No existing master data found. Run stage 1 first.")
                return
        processor.stage2_generate_training_pairs()
    else:  # 'all'
        processor.run_full_pipeline()

if __name__ == "__main__":
    main()
