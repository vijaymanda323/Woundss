#!/usr/bin/env python3
"""
Test Improved Dynamic Healing Prediction
========================================

Demonstrate the enhanced healing time prediction system.
"""

import requests
import json
from pathlib import Path
import time
from datetime import datetime, timedelta

def simulate_patient_progress():
    """Simulate a patient's healing progress over time."""
    
    print("🏥 Simulating Patient Healing Progress")
    print("=" * 50)
    
    patient_id = "healing_test_patient"
    test_image = "datasets/test_wounds/images/burn_wound_01.jpg"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"👤 Patient ID: {patient_id}")
    print(f"📷 Using image: {Path(test_image).name}")
    print()
    
    # Simulate multiple visits over time
    visits = [
        {"day": 0, "notes": "Initial injury - burn wound"},
        {"day": 5, "notes": "5 days post-injury - healing progressing"},
        {"day": 10, "notes": "10 days post-injury - significant improvement"},
        {"day": 15, "notes": "15 days post-injury - almost healed"}
    ]
    
    for visit in visits:
        print(f"📅 Day {visit['day']}: {visit['notes']}")
        
        try:
            # Simulate timestamp for this visit
            visit_date = datetime.now() - timedelta(days=visit['day'])
            timestamp = visit_date.isoformat()
            
            with open(test_image, 'rb') as f:
                files = {'image': f}
                data = {
                    'patient_id': patient_id,
                    'timestamp': timestamp,
                    'notes': visit['notes']
                }
                
                response = requests.post('http://localhost:5000/analyze', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                print(f"   ✅ Analysis successful!")
                print(f"   📏 Wound area: {result['area_cm2']:.2f} cm²")
                
                # Show healing progress
                if result.get('healing_pct') is not None:
                    print(f"   📈 Healing progress: {result['healing_pct']:.1f}%")
                else:
                    print(f"   📈 Healing progress: First measurement")
                
                # Show healing prediction
                days_to_heal = result.get('days_to_heal')
                if days_to_heal is not None:
                    print(f"   ⏱️  Days to complete healing: {days_to_heal}")
                else:
                    print(f"   ⏱️  Days to complete healing: Not yet predictable")
                
                # Show wound classification
                wound_info = result['wound_classification']
                print(f"   🏥 Wound type: {wound_info['wound_type']}")
                print(f"   📊 Healing category: {wound_info['healing_time_category']}")
                confidence = wound_info.get('healing_time_confidence', 0)
                print(f"   🎯 Confidence: {confidence:.1%}")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                if response.status_code == 500:
                    error_data = response.json()
                    print(f"   Error details: {error_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
        time.sleep(0.5)

def test_different_wound_types():
    """Test different wound types with dynamic prediction."""
    
    print("\n🔍 Testing Different Wound Types")
    print("=" * 50)
    
    # Test different wound images
    test_cases = [
        ("datasets/test_wounds/images/burn_wound_01.jpg", "burn_patient", "Burn wound"),
        ("datasets/test_wounds/images/chronic_wound_01.jpg", "chronic_patient", "Chronic wound"),
        ("datasets/test_wounds/images/surgical_wound_01.jpg", "surgical_patient", "Surgical wound"),
        ("datasets/Abrasions/images/abrasions (1).jpg", "abrasion_patient", "Abrasion"),
        ("datasets/Burns/images/burns (1).jpg", "burn_patient_2", "Burn injury")
    ]
    
    for image_path, patient_id, wound_description in test_cases:
        print(f"🏥 {wound_description}")
        
        if not Path(image_path).exists():
            print(f"   ❌ Image not found: {image_path}")
            continue
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'patient_id': patient_id}
                
                response = requests.post('http://localhost:5000/analyze', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ Analysis successful!")
                print(f"   📏 Wound area: {result['area_cm2']:.2f} cm²")
                print(f"   📐 Perimeter: {result['perimeter']:.0f} pixels")
                
                # Show wound classification
                wound_info = result['wound_classification']
                print(f"   🏥 Type: {wound_info['wound_type']}")
                print(f"   ⏱️  Days to cure: {wound_info['estimated_days_to_cure']}")
                print(f"   📊 Category: {wound_info['healing_time_category']}")
                confidence = wound_info.get('healing_time_confidence', 0)
                print(f"   🎯 Confidence: {confidence:.1%}")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def demonstrate_improvements():
    """Demonstrate the key improvements in the healing prediction system."""
    
    print("\n🚀 Key Improvements in Dynamic Healing Prediction")
    print("=" * 60)
    
    improvements = [
        "✅ **Dynamic Time Prediction**: Healing time adjusts based on wound characteristics",
        "✅ **Progress Tracking**: Considers healing progress from previous images",
        "✅ **Wound Analysis**: Analyzes size, shape, severity, and edge clarity",
        "✅ **Multi-factor Assessment**: Accounts for wound type, location, and patient factors",
        "✅ **Confidence Scoring**: Provides confidence levels for predictions",
        "✅ **Fallback System**: Uses OpenCV segmentation when ML models unavailable",
        "✅ **Real-time Updates**: Predictions improve as more data becomes available"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print()
    print("💡 **How It Works:**")
    print("   1. Analyzes wound characteristics (size, shape, edges)")
    print("   2. Tracks healing progress from previous measurements")
    print("   3. Applies multiple adjustment factors")
    print("   4. Provides dynamic healing time prediction")
    print("   5. Updates predictions as wound heals")
    
    print()
    print("📊 **Example Scenarios:**")
    print("   • Small, regular wound → Faster healing prediction")
    print("   • Large, irregular wound → Slower healing prediction")
    print("   • Good healing progress → Reduced remaining time")
    print("   • Poor healing progress → Increased remaining time")
    print("   • Sharp wound edges → Better healing prognosis")

if __name__ == "__main__":
    print("🔬 Testing Improved Dynamic Healing Prediction System")
    print("=" * 60)
    
    # Test patient progress simulation
    simulate_patient_progress()
    
    # Test different wound types
    test_different_wound_types()
    
    # Demonstrate improvements
    demonstrate_improvements()
    
    print("\n🎉 Testing completed!")
    print("\n💡 **Next Steps:**")
    print("   • Upload more wound images to improve model accuracy")
    print("   • Train classification models for better wound type detection")
    print("   • Add patient demographic data for more personalized predictions")
    print("   • Implement real-time monitoring and alerts")
