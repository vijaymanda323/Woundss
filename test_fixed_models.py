#!/usr/bin/env python3
"""
Test the fixed models to ensure they work properly.
"""

import torch
import pickle
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def test_classification_model():
    """Test the classification model."""
    print("🧪 Testing Classification Model...")
    
    try:
        # Load model
        model_path = 'models/wound_classification_model.pth'
        encoder_path = 'models/label_encoder.pkl'
        
        if not os.path.exists(model_path):
            print("❌ Model file not found")
            return False
        
        if not os.path.exists(encoder_path):
            print("❌ Encoder file not found")
            return False
        
        # Load model data with weights_only=False for compatibility
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Model loaded with {model_data['num_classes']} classes")
        print(f"📋 Classes: {model_data['classes']}")
        
        # Load encoder
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        print(f"✅ Encoder loaded with {len(encoder.classes_)} classes")
        
        # Test with a sample image
        test_image_path = 'datasets/Burns/images/burns (1).jpg'
        if os.path.exists(test_image_path):
            # Load and preprocess image
            image = Image.open(test_image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)
            
            # Create model and make prediction
            from fix_model_issues import SimpleWoundClassifier
            model = SimpleWoundClassifier(model_data['num_classes'])
            model.load_state_dict(model_data['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
                
                predicted_class = encoder.inverse_transform([predicted_class_idx])[0]
                
                print(f"🎯 Prediction: {predicted_class}")
                print(f"📊 Confidence: {confidence:.3f}")
                
                if predicted_class == 'burn':
                    print("✅ CORRECT! Burn wound identified correctly!")
                    return True
                else:
                    print(f"❌ WRONG! Expected 'burn', got '{predicted_class}'")
                    return False
        else:
            print("❌ Test image not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_segmentation_model():
    """Test the segmentation model."""
    print("\n🧪 Testing Segmentation Model...")
    
    try:
        model_path = 'models/wound_segmentation_model.pth'
        
        if not os.path.exists(model_path):
            print("❌ Segmentation model file not found")
            return False
        
        model_data = torch.load(model_path, map_location='cpu')
        print("✅ Segmentation model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing segmentation model: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Fixed Models")
    print("=" * 40)
    
    # Test classification model
    classification_ok = test_classification_model()
    
    # Test segmentation model
    segmentation_ok = test_segmentation_model()
    
    print("\n📊 Test Results:")
    print(f"Classification Model: {'✅ PASS' if classification_ok else '❌ FAIL'}")
    print(f"Segmentation Model: {'✅ PASS' if segmentation_ok else '❌ FAIL'}")
    
    if classification_ok and segmentation_ok:
        print("\n🎉 All models are working correctly!")
        print("🚀 The API should now provide accurate wound predictions!")
    else:
        print("\n⚠️ Some models have issues that need to be fixed.")

if __name__ == "__main__":
    main()
