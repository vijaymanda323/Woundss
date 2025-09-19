#!/usr/bin/env python3
"""
Simple Test for Wound Analysis
=============================

Test your wound analysis system with sample images.
"""

import requests
import json
from pathlib import Path

def test_wound_analysis():
    """Test the wound analysis with sample images."""
    
    print("🔍 Testing Wound Analysis System")
    print("=" * 50)
    
    # Test images from your datasets
    test_images = [
        ("datasets/test_wounds/images/burn_wound_01.jpg", "burn_patient"),
        ("datasets/test_wounds/images/chronic_wound_01.jpg", "chronic_patient"),
        ("datasets/test_wounds/images/surgical_wound_01.jpg", "surgical_patient"),
        ("datasets/Abrasions/images/abrasions (1).jpg", "abrasion_patient"),
        ("datasets/Burns/images/burns (1).jpg", "burn_patient_2")
    ]
    
    print("🚀 Starting tests...")
    print()
    
    for i, (image_path, patient_id) in enumerate(test_images, 1):
        print(f"📁 Test {i}: {Path(image_path).name}")
        
        if not Path(image_path).exists():
            print(f"   ❌ Image not found: {image_path}")
            continue
        
        try:
            # Send request to API
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'patient_id': patient_id}
                
                response = requests.post('http://localhost:5000/analyze', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                wound_info = result['wound_classification']
                print(f"   ✅ Analysis successful!")
                print(f"   🏥 Type: {wound_info['wound_type']}")
                print(f"   ⏱️  Days to cure: {wound_info['estimated_days_to_cure']}")
                print(f"   📏 Area: {result['area_cm2']:.2f} cm²")
                print(f"   🎯 Confidence: {result['model_confidence']:.1%}")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Cannot connect to API server")
            print("   💡 Make sure server is running: python app.py")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    print("🎉 Testing completed!")

if __name__ == "__main__":
    test_wound_analysis()





