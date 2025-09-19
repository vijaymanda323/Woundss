#!/usr/bin/env python3
"""
Dynamic Healing Prediction Solution
===================================

This script demonstrates how to implement dynamic healing time prediction
that adjusts based on wound progress and patient history.
"""

import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def simulate_dynamic_healing():
    """Simulate dynamic healing prediction with different scenarios."""
    
    print("🏥 Dynamic Healing Prediction Solution")
    print("=" * 60)
    print()
    
    # Test scenarios showing the difference between fixed and dynamic predictions
    scenarios = [
        {
            "name": "Fresh Burn Wound (Day 0)",
            "image": "datasets/test_wounds/images/burn_wound_01.jpg",
            "patient_id": "burn_patient_fresh",
            "days_since_injury": 0,
            "fixed_prediction": 30,  # Old system: always 30 days
            "dynamic_prediction": 21,  # New system: based on wound characteristics
            "explanation": "Fresh burn → Dynamic system analyzes wound size, depth, and type"
        },
        {
            "name": "Healing Burn Wound (Day 10)",
            "image": "datasets/test_wounds/images/burn_wound_01.jpg", 
            "patient_id": "burn_patient_healing",
            "days_since_injury": 10,
            "fixed_prediction": 30,  # Old system: still 30 days
            "dynamic_prediction": 11,  # New system: 21 - 10 = 11 days remaining
            "explanation": "10 days later → Dynamic system accounts for healing progress"
        },
        {
            "name": "Slow-Healing Burn (Day 15)",
            "image": "datasets/test_wounds/images/burn_wound_01.jpg",
            "patient_id": "burn_patient_slow",
            "days_since_injury": 15,
            "fixed_prediction": 30,  # Old system: still 30 days
            "dynamic_prediction": 20,  # New system: detects slow healing, adjusts prediction
            "explanation": "Slow healing detected → Dynamic system extends prediction"
        },
        {
            "name": "Chronic Wound (Day 30)",
            "image": "datasets/test_wounds/images/chronic_wound_01.jpg",
            "patient_id": "chronic_patient",
            "days_since_injury": 30,
            "fixed_prediction": 30,  # Old system: still 30 days
            "dynamic_prediction": 60,  # New system: recognizes chronic condition
            "explanation": "Chronic wound → Dynamic system predicts longer healing time"
        }
    ]
    
    print("📊 Comparison: Fixed vs Dynamic Healing Predictions")
    print("-" * 60)
    print(f"{'Scenario':<25} {'Fixed':<8} {'Dynamic':<8} {'Improvement':<12}")
    print("-" * 60)
    
    for scenario in scenarios:
        improvement = scenario['fixed_prediction'] - scenario['dynamic_prediction']
        improvement_str = f"{improvement:+d} days" if improvement != 0 else "Same"
        
        print(f"{scenario['name']:<25} {scenario['fixed_prediction']:<8} {scenario['dynamic_prediction']:<8} {improvement_str:<12}")
    
    print()
    print("🎯 Key Benefits of Dynamic Prediction:")
    print("   • More accurate predictions based on actual healing progress")
    print("   • Personalized to each patient's healing rate")
    print("   • Accounts for wound characteristics and type")
    print("   • Adjusts predictions as healing progresses")
    print("   • Better treatment planning and patient communication")
    print()
    
    # Test the actual API
    print("🔍 Testing Current API Response:")
    print("-" * 40)
    
    test_image = "datasets/test_wounds/images/burn_wound_01.jpg"
    if Path(test_image).exists():
        try:
            response = requests.post(
                "http://localhost:5000/analyze",
                files={"image": open(test_image, 'rb')},
                data={"patient_id": "test_patient"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Current API Response:")
                area_cm2 = result.get('area_cm2', 0)
                if area_cm2 is not None:
                    print(f"   📏 Wound Area: {area_cm2:.2f} cm²")
                else:
                    print(f"   📏 Wound Area: Not calculated")
                print(f"   🏥 Wound Type: {result['wound_classification']['wound_type']}")
                print(f"   ⏱️  Days to Cure: {result['wound_classification']['estimated_days_to_cure']}")
                print(f"   📊 Category: {result['wound_classification']['healing_time_category']}")
                print(f"   🎯 Model Available: {result['wound_classification']['model_available']}")
                
                if not result['wound_classification']['model_available']:
                    print()
                    print("⚠️  Note: Classification model not available")
                    print("   The dynamic healing prediction requires the trained model")
                    print("   to be loaded and available in the system.")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Test image not found: {test_image}")

def show_implementation_details():
    """Show how the dynamic healing prediction would be implemented."""
    
    print("\n🔧 Implementation Details:")
    print("=" * 50)
    print()
    
    print("1. **Wound Analysis**:")
    print("   • Analyze wound size, shape, and appearance")
    print("   • Detect wound type (burn, cut, chronic, etc.)")
    print("   • Calculate healing progress from previous images")
    print()
    
    print("2. **Dynamic Calculation**:")
    print("   • Base healing time by wound type")
    print("   • Adjust based on current wound state")
    print("   • Factor in healing progress over time")
    print("   • Account for patient-specific healing rate")
    print()
    
    print("3. **Example Algorithm**:")
    print("   ```python")
    print("   def predict_dynamic_healing(wound_type, current_area, days_since_injury, healing_progress):")
    print("       base_time = get_base_healing_time(wound_type)")
    print("       progress_factor = calculate_healing_progress(current_area, healing_progress)")
    print("       remaining_days = base_time - days_since_injury")
    print("       adjusted_days = remaining_days * progress_factor")
    print("       return max(adjusted_days, 1)  # At least 1 day")
    print("   ```")
    print()
    
    print("4. **Benefits Over Fixed Prediction**:")
    print("   • Fresh burn (Day 0): 21 days (vs fixed 30)")
    print("   • Healing burn (Day 10): 11 days remaining (vs fixed 30)")
    print("   • Slow-healing burn (Day 15): 20 days remaining (vs fixed 30)")
    print("   • Chronic wound (Day 30): 60+ days (vs fixed 30)")

if __name__ == "__main__":
    simulate_dynamic_healing()
    show_implementation_details()
