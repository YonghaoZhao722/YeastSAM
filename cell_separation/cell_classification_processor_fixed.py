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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class CellClassificationProcessorFixed:
    """
    CORRECTED Data processor for binary cell classification (Normal vs Budding)
    
    Enhanced Logic:
    - Ground Truth: dic_masks (contains ALL cells)
    - Helper: divided_masks (contains only budding cells, used for identification)
    - Uses Hungarian algorithm for better instance matching
    - Filters out boundary cells (incomplete masks)
    - For each cell in dic_masks:
      * If it matches with divided_masks -> Budding (1)
      * If no match -> Normal (0)
    - Each budding event = ONE sample (mother+daughter combined)
    """
    
    def __init__(self, data_root: str = "data", output_dir: str = "classification_data_fixed"):
        self.data_root = Path(data_root)
        self.dic_masks_dir = self.data_root / "dic_masks"           # Ground Truth (ALL cells)
        self.divided_masks_dir = self.data_root / "divided_masks"   # Helper (only budding cells)
        self.divided_outlines_dir = self.data_root / "divided_outlines"  # Annotations
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        
        self.master_df = None
        self.target_size = (256, 256)
        
        print(f"=== ENHANCED Cell Classification Processor ===")
        print(f"Ground Truth (all cells): {self.dic_masks_dir}")
        print(f"Helper (budding only): {self.divided_masks_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target image size: {self.target_size}")
        print(f"Task: Binary classification (Normal=0, Budding=1)")
        print(f"Enhanced with Hungarian algorithm matching & boundary filtering")
    
    def _is_boundary_cell(self, cell_mask: np.ndarray, tolerance: int = 30) -> bool:
        """
        Check if a cell is at the boundary (incomplete mask)
        
        IMPORTANT: Images were globally shifted by ~25 pixels, so cells that were 
        originally at boundaries are now ~25 pixels inside. Using tolerance=30 
        to catch these shifted boundary cells.
        
        Args:
            cell_mask: Binary mask of the cell
            tolerance: Tolerance for boundary detection (pixels) - increased to 30 
                      to account for the ~25 pixel global shift
            
        Returns:
            True if cell is at boundary (should be filtered out)
        """
        
        # Get cell coordinates
        coords = np.argwhere(cell_mask > 0)
        if len(coords) == 0:
            return True  # Empty mask should be filtered
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        h, w = cell_mask.shape
        
        # Check if cell is near image boundaries (accounting for ~25px shift)
        near_top = np.any(y_coords <= tolerance)
        near_bottom = np.any(y_coords >= h - tolerance - 1)
        near_left = np.any(x_coords <= tolerance)
        near_right = np.any(x_coords >= w - tolerance - 1)
        
        # If cell doesn't approach any boundary, it's definitely complete
        if not (near_top or near_bottom or near_left or near_right):
            return False
        
        # Calculate cell area and boundary proximity ratio
        cell_area = len(coords)
        
        # Count pixels near each boundary
        pixels_near_top = np.sum(y_coords <= tolerance)
        pixels_near_bottom = np.sum(y_coords >= h - tolerance - 1)
        pixels_near_left = np.sum(x_coords <= tolerance)
        pixels_near_right = np.sum(x_coords >= w - tolerance - 1)
        
        # If a significant portion of the cell is near any boundary, filter it
        max_boundary_ratio = max(pixels_near_top, pixels_near_bottom, 
                                pixels_near_left, pixels_near_right) / cell_area
        
        # More aggressive filtering for shifted images
        if max_boundary_ratio > 0.15:  # If >15% of cell is near boundary
            return True
        
        # Check for straight edges along boundaries (with larger tolerance)
        if near_top and pixels_near_top > 5:
            top_cells = coords[y_coords <= tolerance]
            y_range = np.max(top_cells[:, 0]) - np.min(top_cells[:, 0])
            if y_range <= tolerance // 3:  # Relatively flat edge
                return True
        
        if near_bottom and pixels_near_bottom > 5:
            bottom_cells = coords[y_coords >= h - tolerance - 1]
            y_range = np.max(bottom_cells[:, 0]) - np.min(bottom_cells[:, 0])
            if y_range <= tolerance // 3:
                return True
        
        if near_left and pixels_near_left > 5:
            left_cells = coords[x_coords <= tolerance]
            x_range = np.max(left_cells[:, 1]) - np.min(left_cells[:, 1])
            if x_range <= tolerance // 3:
                return True
        
        if near_right and pixels_near_right > 5:
            right_cells = coords[x_coords >= w - tolerance - 1]
            x_range = np.max(right_cells[:, 1]) - np.min(right_cells[:, 1])
            if x_range <= tolerance // 3:
                return True
        
        # Check for cells cut off at corners (accounting for shift)
        corner_checks = [
            (near_top and near_left and pixels_near_top + pixels_near_left > 0.25 * cell_area),
            (near_top and near_right and pixels_near_top + pixels_near_right > 0.25 * cell_area),
            (near_bottom and near_left and pixels_near_bottom + pixels_near_left > 0.25 * cell_area),
            (near_bottom and near_right and pixels_near_bottom + pixels_near_right > 0.25 * cell_area),
        ]
        
        if any(corner_checks):
            return True
        
        return False
    
    def _merge_mother_daughter_pairs(self, divided_mask: np.ndarray, 
                                   cell_annotations: Dict) -> np.ndarray:
        """
        Merge mother-daughter pairs in divided_mask to single instances
        
        Args:
            divided_mask: Original divided mask with separate mother/daughter IDs
            cell_annotations: Cell annotation information
            
        Returns:
            Merged mask where each budding pair has the same ID
        """
        
        merged_mask = np.zeros_like(divided_mask)
        pair_mapping = {}
        next_pair_id = 1
        
        # Group cells by budding pair
        budding_pairs = {}
        for cell_id, info in cell_annotations.items():
            pair_id = info.get('budding_pair_id')
            if pair_id is not None:
                if pair_id not in budding_pairs:
                    budding_pairs[pair_id] = []
                budding_pairs[pair_id].append(cell_id)
        
        # Assign same ID to each mother-daughter pair
        for pair_id, cell_ids in budding_pairs.items():
            merged_id = next_pair_id
            for cell_id in cell_ids:
                cell_region = (divided_mask == cell_id)
                merged_mask[cell_region] = merged_id
                pair_mapping[cell_id] = merged_id
            next_pair_id += 1
        
        # Handle cells without pair information
        unique_ids = np.unique(divided_mask)
        for cell_id in unique_ids:
            if cell_id > 0 and cell_id not in pair_mapping:
                cell_region = (divided_mask == cell_id)
                merged_mask[cell_region] = next_pair_id
                pair_mapping[cell_id] = next_pair_id
                next_pair_id += 1
        
        return merged_mask
    
    def _compute_instance_features(self, mask: np.ndarray, instance_id: int) -> Dict:
        """
        Compute features for an instance (centroid, area, bounding box)
        
        Args:
            mask: Labeled mask
            instance_id: ID of the instance
            
        Returns:
            Dictionary with instance features
        """
        
        instance_mask = (mask == instance_id)
        coords = np.argwhere(instance_mask)
        
        if len(coords) == 0:
            return None
        
        # Centroid
        centroid_y, centroid_x = coords.mean(axis=0)
        
        # Area
        area = len(coords)
        
        # Bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return {
            'centroid': (centroid_y, centroid_x),
            'area': area,
            'bbox': (y_min, x_min, y_max, x_max),
            'coords': coords
        }
    
    def _match_instances_hungarian(self, dic_mask: np.ndarray, 
                                 merged_divided_mask: np.ndarray) -> Dict[int, int]:
        """
        Use Hungarian algorithm to match instances between dic_mask and divided_mask
        
        Args:
            dic_mask: Ground truth mask (all cells)
            merged_divided_mask: Merged divided mask (budding pairs)
            
        Returns:
            Dictionary mapping dic_cell_id -> divided_cell_id (None if no match)
        """
        
        # Get instance features from both masks
        dic_ids = np.unique(dic_mask)
        dic_ids = dic_ids[dic_ids > 0]
        
        divided_ids = np.unique(merged_divided_mask)
        divided_ids = divided_ids[divided_ids > 0]
        
        if len(dic_ids) == 0 or len(divided_ids) == 0:
            return {int(cell_id): None for cell_id in dic_ids}
        
        # Compute features for all instances
        dic_features = {}
        for cell_id in dic_ids:
            features = self._compute_instance_features(dic_mask, cell_id)
            if features is not None:
                dic_features[cell_id] = features
        
        divided_features = {}
        for cell_id in divided_ids:
            features = self._compute_instance_features(merged_divided_mask, cell_id)
            if features is not None:
                divided_features[cell_id] = features
        
        # Build cost matrix based on centroid distances and area differences
        dic_list = list(dic_features.keys())
        divided_list = list(divided_features.keys())
        
        if len(dic_list) == 0 or len(divided_list) == 0:
            return {int(cell_id): None for cell_id in dic_ids}
        
        cost_matrix = np.zeros((len(dic_list), len(divided_list)))
        
        for i, dic_id in enumerate(dic_list):
            dic_centroid = dic_features[dic_id]['centroid']
            dic_area = dic_features[dic_id]['area']
            
            for j, div_id in enumerate(divided_list):
                div_centroid = divided_features[div_id]['centroid']
                div_area = divided_features[div_id]['area']
                
                # Distance cost (normalized)
                distance = np.sqrt((dic_centroid[0] - div_centroid[0])**2 + 
                                 (dic_centroid[1] - div_centroid[1])**2)
                
                # Area difference cost (normalized)
                area_diff = abs(dic_area - div_area) / max(dic_area, div_area)
                
                # Overlap cost (IoU)
                dic_region = (dic_mask == dic_id)
                div_region = (merged_divided_mask == div_id)
                intersection = np.sum(dic_region & div_region)
                union = np.sum(dic_region | div_region)
                iou = intersection / union if union > 0 else 0
                
                # Combined cost (lower is better)
                cost = distance / 100.0 + area_diff + (1 - iou)
                cost_matrix[i, j] = cost
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping with threshold for valid matches
        matching = {}
        for i, dic_id in enumerate(dic_ids):
            matching[int(dic_id)] = None  # Default to no match
        
        threshold = 1.0  # Adjust threshold as needed
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < threshold:
                dic_id = dic_list[i]
                div_id = divided_list[j]
                matching[int(dic_id)] = int(div_id)
        
        return matching
    
    def find_all_cells_corrected(self) -> List[Dict]:
        """
        ENHANCED: Find all cells from Ground Truth and classify them properly
        """
        print("\n=== Finding All Cells (ENHANCED Logic) ===")
        
        all_cells = []
        boundary_filtered_count = 0
        
        # Get all DIC mask files (Ground Truth)
        dic_mask_files = [f for f in self.dic_masks_dir.glob("*.tif") 
                         if not f.name.startswith("._")]
        
        print(f"Found {len(dic_mask_files)} Ground Truth files")
        
        for dic_mask_file in dic_mask_files:
            try:
                # Load Ground Truth mask
                dic_mask = tifffile.imread(dic_mask_file)
                unique_ids = np.unique(dic_mask)
                unique_ids = unique_ids[unique_ids > 0]  # Remove background
                
                print(f"Processing {dic_mask_file.name}: {len(unique_ids)} cells")
                
                # Find corresponding divided mask file
                divided_mask_file = self._find_corresponding_divided_mask(dic_mask_file)
                divided_mask = None
                merged_divided_mask = None
                
                if divided_mask_file and divided_mask_file.exists():
                    divided_mask = tifffile.imread(divided_mask_file)
                    
                    # Get annotation info for merging
                    txt_file = self._find_corresponding_txt_file(dic_mask_file)
                    cell_annotations = {}
                    if txt_file and txt_file.exists():
                        cell_annotations = self._parse_txt_file(txt_file)
                    
                    # Merge mother-daughter pairs
                    merged_divided_mask = self._merge_mother_daughter_pairs(divided_mask, cell_annotations)
                    
                    # Use Hungarian algorithm for matching
                    matching = self._match_instances_hungarian(dic_mask, merged_divided_mask)
                else:
                    matching = {int(cell_id): None for cell_id in unique_ids}
                
                # Process each cell in Ground Truth
                for cell_id in unique_ids:
                    cell_mask = (dic_mask == cell_id).astype(np.uint8)
                    
                    # Filter boundary cells
                    if self._is_boundary_cell(cell_mask):
                        boundary_filtered_count += 1
                        continue
                    
                    # Check if this cell matches a budding cell
                    matched_divided_id = matching.get(int(cell_id))
                    is_budding = matched_divided_id is not None
                    
                    # Create cell data
                    cell_data = {
                        'cell_id': int(cell_id),
                        'mask_file': str(dic_mask_file),
                        'is_budding': is_budding,
                        'classification_label': 1 if is_budding else 0,
                        'cell_type': 'budding' if is_budding else 'normal',
                        'matched_divided_id': matched_divided_id,
                        'detection_method': 'hungarian_algorithm'
                    }
                    
                    all_cells.append(cell_data)
                
            except Exception as e:
                print(f"Error processing {dic_mask_file}: {e}")
        
        print(f"\nTotal cells processed: {len(all_cells)}")
        print(f"Boundary cells filtered: {boundary_filtered_count}")
        
        # Count and report distribution
        budding_count = sum(1 for cell in all_cells if cell['is_budding'])
        normal_count = len(all_cells) - budding_count
        
        print(f"Classification Results:")
        print(f"   Normal cells: {normal_count}")
        print(f"   Budding cells: {budding_count}")
        print(f"   Budding ratio: {budding_count/(budding_count+normal_count):.3f}")
        
        return all_cells
    
    def _find_corresponding_divided_mask(self, dic_mask_file: Path) -> Optional[Path]:
        """Find corresponding divided_mask file"""
        base_name = dic_mask_file.stem
        
        # Try different naming patterns
        for pattern in [f"{base_name}.tif", f"{base_name}_shifted.tif"]:
            divided_file = self.divided_masks_dir / pattern
            if divided_file.exists():
                return divided_file
        
        # Try without _shifted
        base_name_no_shifted = base_name.replace("_shifted", "")
        for pattern in [f"{base_name_no_shifted}.tif", f"{base_name_no_shifted}_shifted.tif"]:
            divided_file = self.divided_masks_dir / pattern
            if divided_file.exists():
                return divided_file
        
        return None
    
    def _find_corresponding_txt_file(self, mask_file: Path) -> Optional[Path]:
        """Find corresponding TXT annotation file"""
        base_name = mask_file.stem
        
        # Try multiple naming patterns
        txt_files_to_try = [
            self.divided_outlines_dir / f"{base_name}.txt",  # Exact match
            self.divided_outlines_dir / f"{base_name.replace('_shifted', '')}.txt",  # Without _shifted
        ]
        
        for txt_file in txt_files_to_try:
            if txt_file.exists() and not txt_file.name.startswith('._'):
                return txt_file
        
        return None
    
    def _parse_txt_file(self, txt_file: Path) -> Dict[int, Dict]:
        """Parse TXT file and extract cell information"""
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Unicode decode error for {txt_file.name}")
            return {}
        
        # Extract all cells from the file
        cell_blocks = re.findall(r'CELL\s+Cell_(\d+).*?(?=CELL\s+Cell_\d+|Z_POS\s*$)', content, re.DOTALL)
        
        # Handle the last cell block separately
        last_cell_match = re.search(r'CELL\s+Cell_(\d+).*?Z_POS\s*$', content, re.DOTALL)
        if last_cell_match:
            last_cell_id = last_cell_match.group(1)
            if last_cell_id not in cell_blocks:
                cell_blocks.append(last_cell_id)
        
        cell_info = {}
        
        for cell_id in cell_blocks:
            try:
                full_cell_id = int(cell_id)
                
                # Parse according to naming rules
                budding_pair_id = full_cell_id // 100
                suffix = full_cell_id % 100
                
                # Determine cell type based on suffix
                if suffix == 1:
                    cell_type = 'daughter'
                elif suffix == 2:
                    cell_type = 'mother'
                else:
                    cell_type = 'normal'
                
                cell_info[full_cell_id] = {
                    'txt_file': str(txt_file),
                    'full_cell_id': full_cell_id,
                    'budding_pair_id': budding_pair_id,
                    'cell_type': cell_type,
                    'suffix': suffix
                }
                
            except ValueError as e:
                print(f"Could not parse cell ID '{cell_id}' in {txt_file.name}: {e}")
                continue
        
        return cell_info
    
    def generate_training_sample(self, cell_data: Dict, sample_id: int) -> bool:
        """Generate a single training sample from cell data"""
        
        try:
            # Load the Ground Truth mask
            mask_file = Path(cell_data['mask_file'])
            mask = tifffile.imread(mask_file)
            cell_id = cell_data['cell_id']
            
            # Extract individual cell mask from Ground Truth
            cell_mask = (mask == cell_id).astype(np.uint8)
            
            # Check if mask is non-empty
            if np.sum(cell_mask) == 0:
                print(f"Empty mask found for cell {cell_id}")
                return False
            
            # Find bounding box and crop with padding
            coords = np.argwhere(cell_mask > 0)
            if len(coords) == 0:
                return False
                
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding
            padding = 10
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(cell_mask.shape[0], y_max + padding)
            x_max = min(cell_mask.shape[1], x_max + padding)
            
            # Crop the region
            cropped = cell_mask[y_min:y_max, x_min:x_max]
            
            # Resize to target size
            resized = cv2.resize(cropped.astype(np.float32), self.target_size, 
                               interpolation=cv2.INTER_NEAREST)
            
            # Get classification label
            label = cell_data['classification_label']
            
            # Save the processed data
            sample_name = f"cell_{sample_id}"
            
            np.save(self.output_dir / "images" / f"{sample_name}.npy", resized)
            np.save(self.output_dir / "labels" / f"{sample_name}.npy", label)
            
            return True
            
        except Exception as e:
            print(f"Error processing cell {cell_data.get('cell_id', 'unknown')}: {e}")
            return False
    
    def stage1_extract_all_cells(self) -> pd.DataFrame:
        """Stage 1: Extract all individual cells and create labels"""
        print("\n=== Stage 1: Extract All Individual Cells (ENHANCED) ===")
        
        # Find all cells with enhanced logic
        all_cells = self.find_all_cells_corrected()
        
        if not all_cells:
            print("No cells found!")
            return pd.DataFrame()
        
        # Create DataFrame
        self.master_df = pd.DataFrame(all_cells)
        
        print(f"\nCreated master DataFrame with {len(self.master_df)} cells")
        print(f"📋 Columns: {list(self.master_df.columns)}")
        
        # Show classification statistics
        if 'classification_label' in self.master_df.columns:
            label_counts = self.master_df['classification_label'].value_counts()
            print(f"📊 Classification distribution:")
            print(f"   Normal cells (0): {label_counts.get(0, 0)}")
            print(f"   Budding cells (1): {label_counts.get(1, 0)}")
            
            # Show class balance
            if len(label_counts) > 1:
                ratio = label_counts.get(1, 0) / label_counts.get(0, 1)  # budding/normal
                print(f"📈 Budding/Normal ratio: {ratio:.3f}")
        
        # Save master DataFrame
        output_file = self.output_dir / "classification_master_data_enhanced.csv"
        self.master_df.to_csv(output_file, index=False)
        print(f"💾 Saved master DataFrame to {output_file}")
        
        return self.master_df
    
    def stage2_generate_training_samples(self) -> int:
        """Stage 2: Generate training samples for classification"""
        print("\n=== Stage 2: Generate Classification Training Samples ===")
        
        if self.master_df is None or len(self.master_df) == 0:
            print("❌ No master DataFrame found. Run stage1_extract_all_cells() first.")
            return 0
        
        generated_samples = 0
        
        for idx, cell_data in self.master_df.iterrows():
            try:
                if self.generate_training_sample(cell_data.to_dict(), idx + 1):
                    generated_samples += 1
                    if (generated_samples) % 100 == 0:
                        print(f"Generated {generated_samples} samples...")
                        
            except Exception as e:
                print(f"❌ Error processing sample {idx}: {e}")
                continue
        
        print(f"\n✅ Successfully generated {generated_samples} training samples")
        
        # Save generation summary
        self._save_generation_summary(generated_samples)
        
        return generated_samples
    
    def _save_generation_summary(self, generated_samples: int):
        """Save summary of training sample generation"""
        
        normal_count = len(self.master_df[self.master_df['classification_label'] == 0])
        budding_count = len(self.master_df[self.master_df['classification_label'] == 1])
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_cells_processed': len(self.master_df),
            'successfully_generated': generated_samples,
            'normal_cells': normal_count,
            'budding_cells': budding_count,
            'class_ratio_budding_to_normal': budding_count / max(normal_count, 1),
            'target_image_size': self.target_size,
            'task': 'ENHANCED Binary cell classification (Normal=0, Budding=1)',
            'input_source': 'Ground Truth dic_masks with divided_masks as helper',
            'model_architecture': 'Lightweight CNN for binary classification',
            'enhancement_notes': [
                'Uses Hungarian algorithm for precise instance matching',
                'Filters out boundary cells (incomplete masks)',
                'Merges mother-daughter pairs before matching',
                'Uses combined distance, area, and IoU cost function',
                'Proper overlap-based classification with geometric constraints'
            ]
        }
        
        summary_file = self.output_dir / "classification_generation_summary_enhanced.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Saved enhanced generation summary to {summary_file}")
    
    def run_full_pipeline(self) -> int:
        """Run the complete enhanced classification data processing pipeline"""
        
        print("🚀 Starting ENHANCED Classification Data Processing Pipeline")
        print("=" * 70)
        
        # Stage 1: Extract all cells with enhanced logic
        master_df = self.stage1_extract_all_cells()
        
        if master_df is None or len(master_df) == 0:
            print("❌ Pipeline stopped: No cells found in Stage 1")
            return 0
        
        # Stage 2: Generate training samples
        num_samples = self.stage2_generate_training_samples()
        
        print("\n" + "=" * 70)
        print(f"✅ ENHANCED Pipeline Complete! Generated {num_samples} training samples")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📝 Input: Ground Truth dic_masks")
        print(f"🏷️  Labels: Enhanced binary classification (0=Normal, 1=Budding)")
        print(f"🔧 Features: Hungarian matching + Boundary filtering")
        
        return num_samples


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='ENHANCED Cell Classification Data Processor')
    parser.add_argument('--data_root', default='data', 
                      help='Root directory containing dic_masks, divided_masks, and divided_outlines')
    parser.add_argument('--output_dir', default='classification_data_enhanced',
                      help='Output directory for enhanced classification data')
    parser.add_argument('--target_size', nargs=2, type=int, default=[256, 256],
                      help='Target image size (width height)')
    
    args = parser.parse_args()
    
    # Initialize enhanced processor
    processor = CellClassificationProcessorFixed(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    processor.target_size = tuple(args.target_size)
    
    # Run enhanced pipeline
    processor.run_full_pipeline()


if __name__ == "__main__":
    main() 