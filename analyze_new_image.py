#!/usr/bin/env python3
"""
Analyze New Wound Image
======================

Simple script to analyze new wound images using your trained model.
"""

import requests
import json
from pathlib import Path

def analyze_wound_image(image_path, patient_id="new_patient"):
    """Analyze a wound image using the API."""
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"🔍 Analyzing: {Path(image_path).name}")
    print(f"👤 Patient ID: {patient_id}")
    print("-" * 50)
    
    try:
        # Send request to API
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'patient_id': patient_id}
            
            response = requests.post('http://localhost:5000/analyze', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            print("✅ Analysis Results:")
            print(f"   📊 Status: {result['status']}")
            print(f"   🆔 Record ID: {result['record_id']}")
            
            # Wound classification
            wound_info = result['wound_classification']
            print(f"\n🏥 Wound Classification:")
            print(f"   Type: {wound_info['wound_type']}")
            print(f"   Confidence: {wound_info['wound_type_confidence']:.1%}")
            print(f"   Healing Category: {wound_info['healing_time_category']}")
            print(f"   Days to Cure: {wound_info['estimated_days_to_cure']} days")
            
            # Measurements
            print(f"\n📏 Measurements:")
            print(f"   Area: {result['area_pixels']:,} pixels ({result['area_cm2']:.2f} cm²)")
            print(f"   Perimeter: {result['perimeter']:.1f} pixels")
            
            # Bounding box
            bbox = result['bbox']
            print(f"   Bounding Box: {bbox['width']}×{bbox['height']} at ({bbox['x']}, {bbox['y']})")
            
            # Model info
            print(f"\n🤖 Model Info:")
            print(f"   Version: {result['model_version']}")
            print(f"   Confidence: {result['model_confidence']:.1%}")
            
            return result
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("💡 Make sure the server is running: python app.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function to analyze images."""
    
    print("🔍 Wound Image Analysis Tool")
    print("=" * 50)
    
    # Example usage
    test_images = [
        "datasets/test_wounds/images/burn_wound_01.jpg",
        "datasets/test_wounds/images/chronic_wound_01.jpg", 
        "datasets/test_wounds/images/surgical_wound_01.jpg"
    ]
    
    print("📁 Available test images:")
    for i, img in enumerate(test_images, 1):
        if Path(img).exists():
            print(f"   {i}. {Path(img).name}")
    
    print("\n🚀 To analyze your own image:")
    print("   python analyze_new_image.py")
    print("   Then enter the path to your image when prompted")
    
    # Interactive mode
    while True:
        print("\n" + "="*50)
        image_path = input("📁 Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
            
        if not image_path:
            continue
            
        patient_id = input("👤 Enter patient ID (optional): ").strip() or "patient_001"
        
        analyze_wound_image(image_path, patient_id)

if __name__ == "__main__":
    main()





