#!/usr/bin/env python3
"""
Demo: Dynamic Healing Prediction
===============================

Demonstrate how the improved healing time prediction works.
"""

import requests
import json
from pathlib import Path
from datetime import datetime, timedelta

def demo_dynamic_healing():
    """Demonstrate dynamic healing prediction scenarios."""
    
    print("🏥 Dynamic Healing Prediction Demo")
    print("=" * 50)
    print()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Fresh Burn Wound",
            "image": "datasets/test_wounds/images/burn_wound_01.jpg",
            "patient_id": "burn_patient_fresh",
            "days_since_injury": 0,
            "expected": "Should predict ~21 days initially"
        },
        {
            "name": "Healing Burn Wound (Day 10)",
            "image": "datasets/test_wounds/images/burn_wound_01.jpg", 
            "patient_id": "burn_patient_healing",
            "days_since_injury": 10,
            "expected": "Should predict ~11 days remaining"
        },
        {
            "name": "Chronic Wound",
            "image": "datasets/test_wounds/images/chronic_wound_01.jpg",
            "patient_id": "chronic_patient",
            "days_since_injury": 30,
            "expected": "Should predict longer healing time"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        print(f"   📷 Image: {Path(scenario['image']).name}")
        print(f"   👤 Patient: {scenario['patient_id']}")
        print(f"   📅 Days since injury: {scenario['days_since_injury']}")
        print(f"   🎯 Expected: {scenario['expected']}")
        print()
        
        # Test the API
        try:
            response = requests.post(
                "http://localhost:5000/analyze",
                files={"image": open(scenario['image'], 'rb')},
                data={
                    "patient_id": scenario['patient_id'],
                    "timestamp": (datetime.now() - timedelta(days=scenario['days_since_injury'])).isoformat()
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("   ✅ Analysis Results:")
                print(f"      📏 Area: {result['metrics']['area_cm2']:.2f} cm²")
                print(f"      🏥 Wound type: {result['wound_classification']['wound_type']}")
                print(f"      ⏱️  Days to cure: {result['wound_classification']['estimated_days_to_cure']}")
                print(f"      📊 Category: {result['wound_classification']['healing_time_category']}")
                
                # Show healing progress if available
                if result.get('healing_metrics'):
                    healing = result['healing_metrics']
                    if healing.get('healing_pct') is not None:
                        print(f"      📈 Healing progress: {healing['healing_pct']:.1f}%")
                    if healing.get('days_to_heal') is not None:
                        print(f"      ⏰ Days to complete healing: {healing['days_to_heal']}")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
        print("-" * 50)
        print()

def show_improvement_explanation():
    """Explain how the dynamic healing prediction improves the system."""
    
    print("🚀 How Dynamic Healing Prediction Improves the System")
    print("=" * 60)
    print()
    
    print("❌ OLD SYSTEM (Fixed Predictions):")
    print("   • Burn wound → Always predicts 30 days")
    print("   • Patient uploads after 15 days → Still shows 30 days")
    print("   • No consideration of healing progress")
    print()
    
    print("✅ NEW SYSTEM (Dynamic Predictions):")
    print("   • Analyzes current wound state (size, appearance)")
    print("   • Considers healing progress from previous images")
    print("   • Adjusts prediction based on actual healing rate")
    print("   • Accounts for wound type and characteristics")
    print()
    
    print("📊 Example Scenarios:")
    print("   • Fresh burn (Day 0) → Predicts 21 days")
    print("   • Healing burn (Day 10) → Predicts 11 days remaining")
    print("   • Slow-healing burn (Day 15) → Predicts 20 days remaining")
    print("   • Chronic wound (Day 30) → Predicts 60+ days")
    print()
    
    print("🎯 Benefits:")
    print("   • More accurate predictions")
    print("   • Personalized to patient's healing rate")
    print("   • Better treatment planning")
    print("   • Improved patient communication")

if __name__ == "__main__":
    demo_dynamic_healing()
    print()
    show_improvement_explanation()




