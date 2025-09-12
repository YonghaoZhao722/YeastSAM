import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import seaborn as sns

class LightweightCNN(nn.Module):
    """
    Lightweight CNN for binary cell classification (Normal vs Budding)
    
    Architecture optimized for binary shape classification of cell masks
    """
    
    def __init__(self, input_size=128, num_classes=2, dropout=0.3):
        super(LightweightCNN, self).__init__()
        
        self.input_size = input_size
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.7),
            
            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            
            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )
        
        # Add adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Always output 8x8 feature maps
        
        # Fixed feature size after adaptive pooling: 256 channels * 8 * 8 = 16384
        self.feature_size = 256 * 8 * 8
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CellClassificationDataset(Dataset):
    """
    Dataset class for cell classification training data
    
    Input: Individual cell masks (binary images)
    Output: Binary labels (0=Normal, 1=Budding)
    
    Enhanced to handle missing files gracefully - useful when manually removing samples
    """
    
    def __init__(self, data_dir: str = "classification_data", transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.transform = transform
        self.augment = augment
        
        # Get all potential sample files from images directory
        potential_files = [f for f in self.images_dir.glob("cell_*.npy") 
                          if not f.name.startswith("._")]
        
        # Filter to only include samples that have both image and label files
        self.sample_files = []
        missing_labels = []
        missing_images = []
        
        for image_file in potential_files:
            label_file = self.labels_dir / image_file.name
            
            # Check if both files exist
            if image_file.exists() and label_file.exists():
                self.sample_files.append(image_file)
            else:
                if not label_file.exists():
                    missing_labels.append(image_file.name)
                if not image_file.exists():
                    missing_images.append(image_file.name)
        
        # Also check for orphaned label files
        potential_label_files = [f for f in self.labels_dir.glob("cell_*.npy") 
                               if not f.name.startswith("._")]
        orphaned_labels = []
        
        for label_file in potential_label_files:
            image_file = self.images_dir / label_file.name
            if not image_file.exists():
                orphaned_labels.append(label_file.name)
        
        # Report statistics
        total_potential = len(potential_files) + len(orphaned_labels)
        valid_samples = len(self.sample_files)
        
        print(f"Dataset Loading Summary:")
        print(f"   Valid samples: {valid_samples}")
        print(f"   Total potential files found: {total_potential}")
        
        if missing_labels:
            print(f"   Missing label files: {len(missing_labels)}")
            if len(missing_labels) <= 5:
                print(f"      Files: {missing_labels}")
            else:
                print(f"      First 5: {missing_labels[:5]}...")
        
        if missing_images:
            print(f"   Missing image files: {len(missing_images)}")
            if len(missing_images) <= 5:
                print(f"      Files: {missing_images}")
            else:
                print(f"      First 5: {missing_images[:5]}...")
        
        if orphaned_labels:
            print(f"   Orphaned label files: {len(orphaned_labels)}")
            if len(orphaned_labels) <= 5:
                print(f"      Files: {orphaned_labels}")
            else:
                print(f"      First 5: {orphaned_labels[:5]}...")
        
        if valid_samples == 0:
            raise ValueError(f"No valid training samples found in {self.data_dir}. "
                           f"Ensure both images and labels directories contain matching .npy files.")
        
        # Load generation summary for additional info
        summary_files = [
            self.data_dir / "classification_generation_summary_enhanced.json",
            self.data_dir / "classification_generation_summary.json"
        ]
        
        self.summary = None
        for summary_file in summary_files:
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.summary = json.load(f)
                print(f"Dataset generated on: {self.summary.get('generation_date', 'Unknown')}")
                original_normal = self.summary.get('normal_cells', 'Unknown')
                original_budding = self.summary.get('budding_cells', 'Unknown')
                print(f"Original: Normal={original_normal}, Budding={original_budding}")
                
                if isinstance(original_normal, int) and isinstance(original_budding, int):
                    original_total = original_normal + original_budding
                    retention_rate = valid_samples / original_total if original_total > 0 else 0
                    print(f"Retention rate: {retention_rate:.1%} ({valid_samples}/{original_total})")
                break
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        # Load image and label
        image_file = self.sample_files[idx]
        label_file = self.labels_dir / image_file.name
        
        try:
            # Load data with error handling
            image = np.load(image_file, allow_pickle=True).astype(np.float32)
            label = np.load(label_file, allow_pickle=True).astype(np.int64)
            
            # Add channel dimension for image if needed
            if len(image.shape) == 2:
                image = image[np.newaxis, ...]  # (H, W) -> (1, H, W)
            
            # Convert to tensors
            image = torch.from_numpy(image)
            label = torch.tensor(label, dtype=torch.long)
            
            # Apply data augmentation if enabled
            if self.augment and self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading sample {idx} ({image_file.name}): {e}")
            print(f"    This file may have been deleted during training.")
            
            # Return a dummy sample to avoid crashing
            # This should rarely happen since we filter files in __init__
            dummy_image = torch.zeros((1, 256, 256), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            
            return dummy_image, dummy_label

def calculate_class_weights(dataset):
    """Calculate class weights for balanced training"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.item())
    
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Calculate weights inversely proportional to class frequency
    total_samples = len(labels)
    weights = total_samples / (len(unique_labels) * counts)
    
    class_weights = torch.FloatTensor(weights)
    
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    print(f"Class weights: {dict(zip(unique_labels, weights))}")
    
    return class_weights

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, 
                device=None, save_dir="classification_models", class_weights=None):
    """Train the classification model"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Loss function and optimizer
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, min_lr=1e-6)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    print(f"Training on device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    print("Starting training...")
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / "best_f1_model.pth")
            print("New best F1 model saved")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "best_loss_model.pth")
            print("New best loss model saved")
        
        # Print progress
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.1e} -> {new_lr:.1e}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping: no improvement for {early_stopping_patience} epochs")
            break
        
        # Stop if learning rate becomes too small
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Early stopping: learning rate too small")
            break
    
    print(f"\nTraining completed!")
    print(f"Best F1 score: {best_val_f1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), save_dir / "final_model.pth")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores, save_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': best_val_f1,
        'best_val_loss': best_val_loss,
        'num_epochs_completed': len(train_losses),
        'final_lr': optimizer.param_groups[0]['lr'],
        'training_date': datetime.now().isoformat()
    }
    
    with open(save_dir / "classification_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

def evaluate_test_set(model, test_loader, device=None, save_dir="classification_models"):
    """Evaluate model on test set with detailed metrics"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing", leave=False)
        
        for images, labels in test_progress:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate detailed metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    # Calculate average metrics
    avg_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    avg_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    avg_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print(f"\nTest Set Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision (Normal/Budding): {precision[0]:.4f} / {precision[1]:.4f}")
    print(f"   Recall (Normal/Budding): {recall[0]:.4f} / {recall[1]:.4f}")
    print(f"   F1 Score (Normal/Budding): {f1[0]:.4f} / {f1[1]:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Average Recall: {avg_recall:.4f}")
    print(f"   Average F1 Score: {avg_f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, save_dir)
    
    results = {
        'test_accuracy': accuracy,
        'test_precision_normal': precision[0],
        'test_precision_budding': precision[1],
        'test_recall_normal': recall[0],
        'test_recall_budding': recall[1],
        'test_f1_normal': f1[0],
        'test_f1_budding': f1[1],
        'test_avg_precision': avg_precision,
        'test_avg_recall': avg_recall,
        'test_avg_f1': avg_f1,
        'confusion_matrix': cm.tolist()
    }
    
    return results

def plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores, save_dir):
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
    
    # Accuracy curve
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # F1 Score curve
    ax3.plot(epochs, val_f1_scores, label='Validation F1 Score', color='purple', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Validation F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Learning curves comparison
    ax4.plot(epochs, val_accuracies, label='Accuracy', color='green', linewidth=2)
    ax4.plot(epochs, val_f1_scores, label='F1 Score', color='purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.set_title('Validation Metrics Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / "classification_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'classification_training_curves.png'}")

def plot_confusion_matrix(cm, save_dir):
    """Plot confusion matrix"""
    
    save_dir = Path(save_dir)  # Ensure it's a Path object
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Budding'], 
                yticklabels=['Normal', 'Budding'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")

def analyze_dataset_before_training(data_dir="classification_data"):
    """Analyze dataset characteristics before training"""
    
    print("Analyzing classification dataset before training...")
    
    labels_dir = Path(data_dir) / "labels"
    label_files = list(labels_dir.glob("cell_*.npy"))
    
    if not label_files:
        print("No label files found for analysis")
        return
    
    # Sample some labels to analyze distribution
    sample_size = min(100, len(label_files))
    sample_files = label_files[:sample_size]
    
    labels = []
    for label_file in sample_files:
        label = np.load(label_file)
        labels.append(label)
    
    labels = np.array(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"Label distribution (from {len(sample_files)} samples):")
    for label_id, name in [(0, "Normal"), (1, "Budding")]:
        if label_id in unique_labels:
            idx = np.where(unique_labels == label_id)[0][0]
            count = counts[idx]
            percentage = 100 * count / len(labels)
            print(f"   {name}: {percentage:.1f}% ({count} samples)")
    
    # Calculate class weights for balanced training
    if len(unique_labels) > 1:
        total_samples = len(labels)
        weights = total_samples / (len(unique_labels) * counts)
        print(f"Suggested class weights: Normal={weights[0]:.2f}, Budding={weights[1]:.2f}")

def main():
    """Main training function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train cell classification model')
    parser.add_argument('--data_dir', default='classification_data', 
                       help='Directory containing processed classification data')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--save_dir', default='classification_models', 
                       help='Directory to save trained models')
    
    args = parser.parse_args()
    
    print("Cell Binary Classification Model Training")
    print("Task: Normal vs Budding Cell Classification")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Check if processed data exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not (data_dir / "images").exists():
        print("No processed classification data found!")
        print(f"   Please ensure {data_dir} contains 'images' and 'labels' directories.")
        return
    
    # Analyze dataset
    analyze_dataset_before_training(data_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        # Note: No normalization needed as input is binary masks
    ])
    
    # Create datasets
    print("\nLoading datasets...")
    full_dataset = CellClassificationDataset(data_dir, transform=train_transform, augment=True)
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(full_dataset)
    
    # Split into train/val/test (70%/15%/15% split)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create validation and test datasets without augmentation
    dataset_no_aug = CellClassificationDataset(data_dir, transform=None, augment=False)
    
    # Use the same indices for validation and test
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(dataset_no_aug, val_indices)
    test_dataset_clean = torch.utils.data.Subset(dataset_no_aug, test_indices)
    
    batch_size = args.batch_size
    print(f"Selected batch size: {batch_size}")
    
    # Create data loaders
    num_workers = min(4, os.cpu_count() if os.cpu_count() else 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset_clean, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset_clean, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Initialize model
    model = LightweightCNN(input_size=256, num_classes=2, dropout=0.3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Use command line epochs or determine based on dataset size
    if args.epochs != 100:  # If user specified epochs
        num_epochs = args.epochs
        print(f"Using specified epochs: {num_epochs}")
    else:
        # Auto-determine epochs based on dataset size
        if len(full_dataset) < 100:
            num_epochs = 150
        elif len(full_dataset) < 500:
            num_epochs = 100
        else:
            num_epochs = 80
        print(f"Auto-determined epochs: {num_epochs}")
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        class_weights=class_weights
    )
    
    print("\nTraining completed successfully!")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    # Load best model for testing
    model.load_state_dict(torch.load("classification_models/best_f1_model.pth"))
    test_results = evaluate_test_set(model, test_loader, device, "classification_models")
    
    # Save complete results including test performance
    history.update(test_results)
    
    with open("classification_models/classification_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModels saved to: classification_models/")
    print(f"Training history saved")
    print(f"Best Validation F1: {history['best_val_f1']:.4f}")
    print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['test_avg_f1']:.4f}")

if __name__ == "__main__":
    main() 