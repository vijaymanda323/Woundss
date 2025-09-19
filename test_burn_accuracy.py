#!/usr/bin/env python3
"""
Test burn wound prediction accuracy with corrected labels.
This script will test the model's ability to correctly identify burn wounds.
"""

import requests
import os
import json
from pathlib import Path

def test_burn_prediction(image_path, expected_type="burn"):
    """Test prediction for a single burn image."""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'patient_id': 'test_burn_patient',
                'timestamp': '2024-01-01T00:00:00Z'
            }
            
            response = requests.post('http://localhost:5000/analyze', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                predicted_type = result.get('wound_classification', {}).get('wound_type', 'unknown')
                confidence = result.get('wound_classification', {}).get('confidence', 0)
                days_to_cure = result.get('wound_classification', {}).get('estimated_days_to_cure', 0)
                
                is_correct = predicted_type.lower() == expected_type.lower()
                
                return {
                    'image': os.path.basename(image_path),
                    'expected': expected_type,
                    'predicted': predicted_type,
                    'confidence': confidence,
                    'days_to_cure': days_to_cure,
                    'correct': is_correct,
                    'area_cm2': result.get('area_cm2', 0)
                }
            else:
                return {
                    'image': os.path.basename(image_path),
                    'error': f'HTTP {response.status_code}: {response.text}',
                    'correct': False
                }
                
    except Exception as e:
        return {
            'image': os.path.basename(image_path),
            'error': str(e),
            'correct': False
        }

def main():
    """Test burn wound prediction accuracy."""
    print("🔥 Testing Burn Wound Prediction Accuracy")
    print("=" * 50)
    
    # Test burn images from the Burns dataset
    burn_dataset_path = "datasets/Burns/images"
    test_results = []
    
    if not os.path.exists(burn_dataset_path):
        print(f"❌ Burn dataset not found at {burn_dataset_path}")
        return
    
    # Get first 10 burn images for testing
    burn_images = []
    for file in os.listdir(burn_dataset_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            burn_images.append(os.path.join(burn_dataset_path, file))
            if len(burn_images) >= 10:
                break
    
    if not burn_images:
        print("❌ No burn images found for testing")
        return
    
    print(f"🧪 Testing {len(burn_images)} burn images...")
    print()
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, image_path in enumerate(burn_images, 1):
        print(f"📸 Test {i}/{len(burn_images)}: {os.path.basename(image_path)}")
        
        result = test_burn_prediction(image_path, "burn")
        test_results.append(result)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            total_predictions += 1
            if result['correct']:
                correct_predictions += 1
                print(f"   ✅ Correct! Predicted: {result['predicted']} (Confidence: {result['confidence']:.2f})")
            else:
                print(f"   ❌ Wrong! Expected: burn, Predicted: {result['predicted']} (Confidence: {result['confidence']:.2f})")
            
            area = result.get('area_cm2', 0) or 0
            days = result.get('days_to_cure', 0) or 0
            print(f"   📊 Area: {area:.2f} cm², Days to cure: {days}")
        
        print()
    
    # Calculate accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print("=" * 50)
        print(f"📊 BURN PREDICTION ACCURACY RESULTS:")
        print(f"✅ Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"🎯 Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("🎉 EXCELLENT! Model is highly accurate for burn prediction!")
        elif accuracy >= 70:
            print("👍 GOOD! Model shows good accuracy for burn prediction.")
        elif accuracy >= 50:
            print("⚠️ MODERATE! Model needs improvement for burn prediction.")
        else:
            print("❌ POOR! Model needs significant improvement for burn prediction.")
        
        print()
        print("🔍 Detailed Results:")
        for result in test_results:
            if 'error' not in result:
                status = "✅" if result['correct'] else "❌"
                print(f"   {status} {result['image']}: {result['predicted']} (conf: {result['confidence']:.2f})")
    
    else:
        print("❌ No successful predictions to analyze")
    
    # Test with other wound types to ensure specificity
    print("\n" + "=" * 50)
    print("🔍 Testing Specificity (should NOT predict burn for other wounds):")
    
    # Test a cut image
    cut_dataset_path = "datasets/Cut/images"
    if os.path.exists(cut_dataset_path):
        cut_images = [f for f in os.listdir(cut_dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if cut_images:
            cut_image_path = os.path.join(cut_dataset_path, cut_images[0])
            print(f"📸 Testing cut image: {os.path.basename(cut_image_path)}")
            
            result = test_burn_prediction(cut_image_path, "cut")
            if 'error' not in result:
                if result['predicted'].lower() != 'burn':
                    print(f"   ✅ Good! Correctly identified as: {result['predicted']}")
                else:
                    print(f"   ❌ Problem! Incorrectly identified burn as: {result['predicted']}")
            else:
                print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
