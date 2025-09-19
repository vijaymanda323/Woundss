#!/usr/bin/env python3
"""
Train the wound classification model with real wound data.
This script will train the model to achieve accurate predictions.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWoundClassifier(nn.Module):
    """Simple CNN for wound classification."""
    
    def __init__(self, num_classes):
        super(SimpleWoundClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class WoundDataset(Dataset):
    """Dataset for wound images."""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_wound_data():
    """Load wound data from all datasets."""
    logger.info("📂 Loading wound data from datasets...")
    
    images = []
    labels = []
    dataset_paths = []
    
    # Find all dataset directories
    datasets_dir = Path('datasets')
    if not datasets_dir.exists():
        logger.error("❌ Datasets directory not found!")
        return None, None, None
    
    # Get all dataset folders
    dataset_folders = [d for d in datasets_dir.iterdir() if d.is_dir()]
    logger.info(f"📁 Found {len(dataset_folders)} dataset folders: {[d.name for d in dataset_folders]}")
    
    for dataset_folder in dataset_folders:
        images_dir = dataset_folder / 'images'
        labels_file = dataset_folder / 'labels.csv'
        
        if not images_dir.exists():
            logger.warning(f"⚠️ Images directory not found in {dataset_folder.name}")
            continue
        
        if not labels_file.exists():
            logger.warning(f"⚠️ Labels file not found in {dataset_folder.name}")
            continue
        
        # Load labels
        try:
            labels_df = pd.read_csv(labels_file)
            logger.info(f"📋 Loaded {len(labels_df)} labels from {dataset_folder.name}")
        except Exception as e:
            logger.error(f"❌ Error loading labels from {dataset_folder.name}: {e}")
            continue
        
        # Load images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        logger.info(f"🖼️ Found {len(image_files)} images in {dataset_folder.name}")
        
        for image_file in image_files:
            try:
                # Load image
                image = Image.open(image_file).convert('RGB')
                
                # Find corresponding label
                filename = image_file.name
                label_row = labels_df[labels_df['filename'] == filename]
                
                if len(label_row) > 0:
                    wound_type = label_row.iloc[0]['wound_type']
                    images.append(image)
                    labels.append(wound_type)
                    dataset_paths.append(str(dataset_folder.name))
                else:
                    logger.warning(f"⚠️ No label found for {filename}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading image {image_file}: {e}")
                continue
    
    logger.info(f"✅ Loaded {len(images)} images with labels")
    logger.info(f"📊 Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return images, labels, dataset_paths

def train_model():
    """Train the wound classification model."""
    logger.info("🚀 Starting model training...")
    
    # Load data
    images, labels, dataset_paths = load_wound_data()
    if images is None or len(images) == 0:
        logger.error("❌ No data loaded, cannot train model")
        return False
    
    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    logger.info(f"🏷️ Encoded labels: {len(label_encoder.classes_)} classes")
    logger.info(f"📋 Classes: {list(label_encoder.classes_)}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    logger.info(f"📊 Training set: {len(X_train)} images")
    logger.info(f"📊 Validation set: {len(X_val)} images")
    
    # Create datasets
    train_dataset = WoundDataset(X_train, y_train, transform=transform)
    val_dataset = WoundDataset(X_val, y_val, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    num_classes = len(label_encoder.classes_)
    model = SimpleWoundClassifier(num_classes)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0
    
    logger.info(f"🎯 Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        # Calculate accuracies
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {train_loss/len(train_loader):.4f}, "
                   f"Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss/len(val_loader):.4f}, "
                   f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"🎉 New best validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    logger.info(f"🏆 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'classes': list(label_encoder.classes_),
        'model_type': 'classification'
    }
    
    torch.save(model_data, 'models/wound_classification_model.pth')
    
    # Save label encoder separately
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    logger.info("💾 Model saved successfully!")
    logger.info(f"📁 Saved to: models/wound_classification_model.pth")
    logger.info(f"📁 Saved to: models/label_encoder.pkl")
    
    return True

def test_trained_model():
    """Test the trained model with burn images."""
    logger.info("🧪 Testing trained model...")
    
    try:
        # Load model
        model_path = 'models/wound_classification_model.pth'
        encoder_path = 'models/label_encoder.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            logger.error("❌ Model files not found")
            return False
        
        # Load model data
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        num_classes = model_data['num_classes']
        model = SimpleWoundClassifier(num_classes)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Test with burn images
        burn_images_dir = Path('datasets/Burns/images')
        if burn_images_dir.exists():
            burn_images = list(burn_images_dir.glob('*.jpg'))[:5]  # Test first 5
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            correct_predictions = 0
            total_predictions = 0
            
            for burn_image in burn_images:
                try:
                    # Load and preprocess image
                    image = Image.open(burn_image).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = model(image_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class_idx].item()
                        
                        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
                        
                        logger.info(f"🖼️ {burn_image.name}: {predicted_class} (confidence: {confidence:.3f})")
                        
                        if predicted_class == 'burn':
                            correct_predictions += 1
                        total_predictions += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error testing {burn_image}: {e}")
            
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            logger.info(f"🎯 Burn prediction accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            
            return accuracy > 80  # Consider successful if >80% accuracy
        
        else:
            logger.warning("⚠️ No burn images found for testing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        return False

def main():
    """Main function."""
    logger.info("🚀 Wound Classification Model Training")
    logger.info("=" * 50)
    
    # Train model
    success = train_model()
    
    if success:
        # Test model
        test_success = test_trained_model()
        
        if test_success:
            logger.info("🎉 SUCCESS! Model trained and tested successfully!")
            logger.info("🔥 Burn wounds should now be predicted accurately!")
        else:
            logger.warning("⚠️ Model trained but testing showed issues")
    else:
        logger.error("❌ Training failed")

if __name__ == "__main__":
    main()


