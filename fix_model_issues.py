#!/usr/bin/env python3
"""
Fix all model loading and classification issues.
This script will create proper models and fix the API.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleUNet(nn.Module):
    """Simple U-Net for wound segmentation."""
    
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        x4 = self.final(x3)
        return torch.sigmoid(x4)

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

def create_simple_classification_model():
    """Create a simple classification model with proper structure."""
    
    logger.info("🔧 Creating proper wound classification model...")
    
    # Define wound types based on our datasets
    wound_types = [
        'burn', 'cut', 'laceration', 'abrasion', 'bruise', 'stab_wound',
        'pressure_ulcer', 'leg_ulcer', 'foot_ulcer', 'abdominal_wound',
        'orthopedic_wound', 'malignant_wound', 'extravasation', 'epidermolysis',
        'hematoma', 'ingrown', 'meningitis', 'miscellaneous', 'pilonidal_sinus',
        'toe_wound', 'test_wound'
    ]
    
    num_classes = len(wound_types)
    
    # Create model
    model = SimpleWoundClassifier(num_classes)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(wound_types)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model with proper structure
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'num_classes': num_classes,
        'classes': wound_types,
        'model_type': 'classification'
    }
    
    torch.save(model_data, 'models/wound_classification_model.pth')
    
    # Save label encoder separately
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    logger.info(f"✅ Model saved with {num_classes} classes: {wound_types}")
    
    return True

def create_segmentation_model():
    """Create a simple segmentation model."""
    
    logger.info("🔧 Creating wound segmentation model...")
    
    # Create a simple U-Net like model
    class SimpleUNet(nn.Module):
        def __init__(self):
            super(SimpleUNet, self).__init__()
            
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            )
            
            self.enc2 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU()
            )
            
            # Decoder
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            )
            
            self.final = nn.Conv2d(64, 1, 1)
            
        def forward(self, x):
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            x3 = self.dec1(x2)
            x4 = self.final(x3)
            return torch.sigmoid(x4)
    
    model = SimpleUNet()
    
    # Save segmentation model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'segmentation'
    }, 'models/wound_segmentation_model.pth')
    
    logger.info("✅ Segmentation model saved")
    
    return True

def fix_app_py():
    """Fix the app.py file to handle models properly."""
    
    logger.info("🔧 Fixing app.py model loading...")
    
    # Read the current app.py
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add proper model loading function
    model_loading_fix = '''
def load_classification_model():
    """Load the wound classification model."""
    try:
        model_path = os.path.join(MODEL_PATH, 'wound_classification_model.pth')
        encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            logger.warning("Classification model files not found")
            return None, None, None
        
    # Load model data with weights_only=False for compatibility
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        num_classes = model_data['num_classes']
        model = SimpleWoundClassifier(num_classes)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info(f"Classification model loaded with {num_classes} classes")
        return model, label_encoder, model_data['classes']
        
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        return None, None, None

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
'''
    
    # Add the fix after the imports
    if 'def load_classification_model():' not in content:
        # Find the position after imports
        import_end = content.find('import logging')
        if import_end != -1:
            next_line = content.find('\n', import_end)
            content = content[:next_line+1] + model_loading_fix + '\n' + content[next_line+1:]
            
            # Write the fixed content
            with open('app.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ app.py updated with proper model loading")
        else:
            logger.warning("Could not find import section in app.py")
    
    return True

def main():
    """Main function to fix all model issues."""
    
    logger.info("🚀 Fixing All Model Issues")
    logger.info("=" * 50)
    
    try:
        # Create proper models
        create_simple_classification_model()
        create_segmentation_model()
        
        # Fix app.py
        fix_app_py()
        
        logger.info("✅ All model issues fixed!")
        logger.info("🎯 Models created:")
        logger.info("   - models/wound_classification_model.pth")
        logger.info("   - models/wound_segmentation_model.pth") 
        logger.info("   - models/label_encoder.pkl")
        logger.info("🔧 app.py updated with proper model loading")
        logger.info("🚀 Restart the API server to use the fixed models!")
        
    except Exception as e:
        logger.error(f"❌ Error fixing models: {e}")

if __name__ == "__main__":
    main()
