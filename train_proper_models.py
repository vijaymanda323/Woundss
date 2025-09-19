#!/usr/bin/env python3
"""
Train proper wound classification models with corrected labels.
This script will create actual PyTorch models for wound type classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoundDataset(Dataset):
    """Custom dataset for wound images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), label
            return Image.new('RGB', (224, 224), (0, 0, 0)), label

class WoundClassifier(nn.Module):
    """CNN model for wound classification."""
    
    def __init__(self, num_classes):
        super(WoundClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_dataset_with_labels(dataset_path):
    """Load dataset with proper labels."""
    images = []
    labels = []
    
    labels_file = os.path.join(dataset_path, 'labels.csv')
    images_dir = os.path.join(dataset_path, 'images')
    
    if not os.path.exists(labels_file):
        logger.warning(f"No labels.csv found in {dataset_path}")
        return [], []
    
    if not os.path.exists(images_dir):
        logger.warning(f"No images directory found in {dataset_path}")
        return [], []
    
    try:
        df = pd.read_csv(labels_file)
        
        for _, row in df.iterrows():
            filename = row['filename']
            wound_type = row['wound_type']
            
            image_path = os.path.join(images_dir, filename)
            if os.path.exists(image_path):
                images.append(image_path)
                labels.append(wound_type)
            else:
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Loaded {len(images)} images from {dataset_path}")
        return images, labels
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_path}: {e}")
        return [], []

def train_classification_model(dataset_paths, output_dir='models'):
    """Train a comprehensive wound classification model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all datasets
    all_images = []
    all_labels = []
    
    for dataset_path in dataset_paths:
        images, labels = load_dataset_with_labels(dataset_path)
        all_images.extend(images)
        all_labels.extend(labels)
    
    if len(all_images) == 0:
        logger.error("No images found in any dataset!")
        return False
    
    logger.info(f"Total images loaded: {len(all_images)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {label_encoder.classes_}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WoundDataset(X_train, y_train, train_transform)
    val_dataset = WoundDataset(X_val, y_val, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = WoundClassifier(num_classes).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] - '
                   f'Train Loss: {train_loss/len(train_loader):.4f}, '
                   f'Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss/len(val_loader):.4f}, '
                   f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'num_classes': num_classes,
                'classes': label_encoder.classes_
            }, os.path.join(output_dir, 'wound_classification_model.pth'))
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save label encoder separately
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return True

def main():
    """Main training function."""
    logger.info("🔥 Training Proper Wound Classification Models")
    logger.info("=" * 60)
    
    # Define dataset paths
    datasets_dir = 'datasets'
    dataset_paths = []
    
    # Get all dataset directories
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path):
            labels_file = os.path.join(item_path, 'labels.csv')
            images_dir = os.path.join(item_path, 'images')
            if os.path.exists(labels_file) and os.path.exists(images_dir):
                dataset_paths.append(item_path)
    
    if not dataset_paths:
        logger.error("No valid datasets found!")
        return
    
    logger.info(f"Found {len(dataset_paths)} valid datasets:")
    for path in dataset_paths:
        logger.info(f"  - {path}")
    
    # Train the model
    success = train_classification_model(dataset_paths)
    
    if success:
        logger.info("✅ Training completed successfully!")
        logger.info("🎯 Model saved as: models/wound_classification_model.pth")
        logger.info("📊 Label encoder saved as: models/label_encoder.pkl")
        logger.info("🚀 The model is now ready for accurate wound classification!")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main()


