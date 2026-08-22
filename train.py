"""
PyTorch Deep Learning Training Pipeline for PlantVillage Crop Disease Classification
Trains MobileNetV3 vision backbone with data augmentation, learning rate scheduling, validation checkpoints, and weight exports.
"""

import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

# Windows UTF-8 stream safety
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from backend.classifier import CropDiseaseClassifier, CLASS_NAMES, IMAGE_SIZE
from backend.dataset_downloader import create_fast_plantvillage_subset, DEFAULT_DATA_DIR

def get_data_transforms():
    """Build train and validation data augmentation pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(
    data_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    output_weights_path: str = "backend/model_weights.pth",
    device_name: str = "auto"
):
    """
    Main training execution function.
    """
    print("=" * 65)
    print(" [AgroAI] Deep Vision Classifier Training Pipeline")
    print("=" * 65)
    
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        
    print(f"[Training] Target Compute Device: {device}")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # If train/val subfolders don't exist, check if data_dir contains class folders directly
    if not os.path.exists(train_dir):
        if os.path.exists(data_dir) and any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)):
            train_dir = data_dir
            val_dir = data_dir
        else:
            print(f"[Training] Dataset not found at {data_dir}. Generating fast PlantVillage subset...")
            create_fast_plantvillage_subset(data_dir)
            train_dir = os.path.join(data_dir, "train")
            val_dir = os.path.join(data_dir, "val")

    train_tf, val_tf = get_data_transforms()
    
    print(f"[Training] Loading datasets from: {train_dir}")
    train_dataset = ImageFolder(train_dir, transform=train_tf)
    val_dataset = ImageFolder(val_dir, transform=val_tf) if os.path.exists(val_dir) else train_dataset
    
    print(f"[Training] Found {len(train_dataset)} training samples across {len(train_dataset.classes)} classes.")
    print(f"[Training] Found {len(val_dataset)} validation samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize Model
    model = CropDiseaseClassifier(num_classes=len(CLASS_NAMES))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print("\n[Training] Beginning Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)
            
        scheduler.step()
        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = (correct_train / total_train * 100.0) if total_train > 0 else 0
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = (correct_val / total_val * 100.0) if total_val > 0 else 0
        
        print(f" Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Checkpoint saving
        if epoch_val_acc >= best_val_acc or epoch == epochs:
            best_val_acc = epoch_val_acc
            os.makedirs(os.path.dirname(os.path.abspath(output_weights_path)), exist_ok=True)
            torch.save(model.state_dict(), output_weights_path)
            
    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"[Training Complete] Best Validation Accuracy: {best_val_acc:.2f}% in {elapsed:.1f}s")
    print(f"[Training] Saved optimal model weights to: {output_weights_path}")
    print("=" * 65)
    return output_weights_path

def main():
    parser = argparse.ArgumentParser(description="AgroAI PlantVillage Training Pipeline")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output", type=str, default="backend/model_weights.pth", help="Path to save weights")
    parser.add_argument("--fast-subset", action="store_true", help="Generate fast local subset across 41 classes")
    
    args = parser.parse_args()
    
    if args.fast_subset or not os.path.exists(args.data_dir):
        create_fast_plantvillage_subset(args.data_dir)
        
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_weights_path=args.output
    )

if __name__ == "__main__":
    main()

