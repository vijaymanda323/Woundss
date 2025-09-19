#!/usr/bin/env python3
"""
Test Dynamic Healing Prediction
==============================

Test the improved healing time prediction system.
"""

import requests
import json
from pathlib import Path
import time

def test_dynamic_healing():
    """Test the dynamic healing prediction with multiple images."""
    
    print("🔍 Testing Dynamic Healing Prediction")
    print("=" * 50)
    
    # Test images
    test_images = [
        ("datasets/test_wounds/images/burn_wound_01.jpg", "burn_patient"),
        ("datasets/test_wounds/images/chronic_wound_01.jpg", "chronic_patient"),
        ("datasets/test_wounds/images/surgical_wound_01.jpg", "surgical_patient")
    ]
    
    print("🚀 Testing with different wound types...")
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
                print(f"   📊 Category: {wound_info['healing_time_category']}")
                print(f"   🎯 Confidence: {wound_info['healing_time_confidence']:.1%}")
                print(f"   📏 Area: {result['area_cm2']:.2f} cm²")
                
                # Show healing progress if available
                if result.get('healing_pct') is not None:
                    print(f"   📈 Healing progress: {result['healing_pct']:.1f}%")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Cannot connect to API server")
            print("   💡 Make sure server is running: python app.py")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
        
        # Wait a bit between requests
        time.sleep(1)
    
    print("🎉 Dynamic healing prediction test completed!")
    print()
    print("💡 Key improvements:")
    print("   • Healing time adjusts based on wound characteristics")
    print("   • Considers healing progress from previous images")
    print("   • Accounts for wound size, shape, and severity")
    print("   • Provides more accurate predictions over time")

def test_healing_progress():
    """Test healing progress tracking with multiple uploads."""
    
    print("\n📈 Testing Healing Progress Tracking")
    print("=" * 50)
    
    # Simulate multiple uploads for the same patient
    patient_id = "progress_test_patient"
    test_image = "datasets/test_wounds/images/burn_wound_01.jpg"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🔄 Simulating healing progress for patient: {patient_id}")
    print()
    
    for day in range(1, 4):  # Simulate 3 days of progress
        print(f"📅 Day {day}:")
        
        try:
            with open(test_image, 'rb') as f:
                files = {'image': f}
                data = {
                    'patient_id': patient_id,
                    'notes': f'Day {day} measurement'
                }
                
                response = requests.post('http://localhost:5000/analyze', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                wound_info = result['wound_classification']
                
                print(f"   🏥 Wound type: {wound_info['wound_type']}")
                print(f"   ⏱️  Days to cure: {wound_info['estimated_days_to_cure']}")
                print(f"   📊 Category: {wound_info['healing_time_category']}")
                
                if result.get('healing_pct') is not None:
                    print(f"   📈 Healing progress: {result['healing_pct']:.1f}%")
                else:
                    print(f"   📈 Healing progress: First measurement")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
        time.sleep(0.5)
    
    print("🎉 Healing progress tracking test completed!")

if __name__ == "__main__":
    test_dynamic_healing()
    test_healing_progress()




