#!/usr/bin/env python3
"""
Test Enhanced Cut Detection
===========================

Test the enhanced cut detection capabilities of the intelligent agent.
"""

import requests
import os
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"

def test_cut_detection():
    """Test cut detection with various cut images."""
    print("🔪 Testing Enhanced Cut Detection")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Backend is not running")
            return False
        print("✅ Backend is running")
    except:
        print("❌ Backend is not running")
        return False
    
    # Find cut images
    cut_images = []
    cut_dataset_path = "datasets/Cut/images"
    if os.path.exists(cut_dataset_path):
        cut_images = [f"{cut_dataset_path}/{f}" for f in os.listdir(cut_dataset_path) if f.endswith('.jpg')][:5]
    
    if not cut_images:
        print("❌ No cut images found")
        return False
    
    print(f"📸 Found {len(cut_images)} cut images")
    print()
    
    results = []
    for i, image_path in enumerate(cut_images):
        print(f"🔍 Testing {i+1}/{len(cut_images)}: {os.path.basename(image_path)}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{API_BASE_URL}/analyze-intelligent", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']
                
                result = {
                    'image': os.path.basename(image_path),
                    'prediction': analysis['prediction'],
                    'confidence': analysis['confidence'],
                    'enhanced_analysis': analysis.get('enhanced_analysis', False),
                    'cut_analysis': analysis.get('cut_analysis', {}),
                    'search_results': analysis.get('search_results', {})
                }
                results.append(result)
                
                print(f"   ✅ Prediction: {analysis['prediction']}")
                print(f"   🎯 Confidence: {analysis['confidence']:.3f}")
                print(f"   🔧 Enhanced: {analysis.get('enhanced_analysis', False)}")
                
                # Show cut-specific analysis
                if 'cut_analysis' in analysis:
                    cut_analysis = analysis['cut_analysis']
                    print(f"   🔪 Cut Probability: {cut_analysis.get('cut_probability', 0):.3f}")
                    print(f"   📏 Linear Structure: {cut_analysis.get('linear_structure', False)}")
                    print(f"   🎨 Color Match: {cut_analysis.get('color_match', False)}")
                
                # Show search results
                if 'search_results' in analysis:
                    search_results = analysis['search_results']
                    print(f"   🔍 Search Query: {search_results.get('query', 'N/A')}")
                    print(f"   📈 Confidence Boost: {search_results.get('confidence_boost', 0)}")
                
                print()
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("📊 CUT DETECTION SUMMARY:")
    print("-" * 40)
    correct_cuts = 0
    total_tests = len(results)
    
    for result in results:
        is_cut = result['prediction'] == 'cut'
        if is_cut:
            correct_cuts += 1
        
        status = "✅ CORRECT" if is_cut else "❌ INCORRECT"
        print(f"   {result['image']}: {result['prediction']} ({result['confidence']:.3f}) - {status}")
    
    accuracy = (correct_cuts / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 Cut Detection Accuracy: {accuracy:.1f}% ({correct_cuts}/{total_tests})")
    
    return accuracy > 80

def test_non_cut_images():
    """Test with non-cut images to ensure they're not misclassified as cuts."""
    print("\n🚫 Testing Non-Cut Images (Should NOT be classified as cuts)")
    print("=" * 60)
    
    # Find non-cut images
    non_cut_images = []
    for dataset in ['Burns', 'Abrasions']:
        dataset_path = f"datasets/{dataset}/images"
        if os.path.exists(dataset_path):
            images = [f"{dataset_path}/{f}" for f in os.listdir(dataset_path) if f.endswith('.jpg')][:2]
            non_cut_images.extend(images)
    
    if not non_cut_images:
        print("❌ No non-cut images found")
        return False
    
    print(f"📸 Found {len(non_cut_images)} non-cut images")
    print()
    
    results = []
    for i, image_path in enumerate(non_cut_images):
        print(f"🔍 Testing {i+1}/{len(non_cut_images)}: {os.path.basename(image_path)}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{API_BASE_URL}/analyze-intelligent", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']
                
                result = {
                    'image': os.path.basename(image_path),
                    'prediction': analysis['prediction'],
                    'confidence': analysis['confidence'],
                    'is_cut': analysis['prediction'] == 'cut'
                }
                results.append(result)
                
                status = "❌ MISCLASSIFIED" if result['is_cut'] else "✅ CORRECT"
                print(f"   {analysis['prediction']} ({analysis['confidence']:.3f}) - {status}")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n📊 NON-CUT CLASSIFICATION SUMMARY:")
    print("-" * 40)
    misclassified = sum(1 for r in results if r['is_cut'])
    total_tests = len(results)
    
    for result in results:
        status = "❌ MISCLASSIFIED" if result['is_cut'] else "✅ CORRECT"
        print(f"   {result['image']}: {result['prediction']} - {status}")
    
    accuracy = ((total_tests - misclassified) / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 Non-Cut Classification Accuracy: {accuracy:.1f}% ({total_tests - misclassified}/{total_tests})")
    
    return misclassified == 0

if __name__ == "__main__":
    print("🔪 Enhanced Cut Detection Test")
    print("=" * 60)
    
    # Test cut detection
    success1 = test_cut_detection()
    
    # Test non-cut images
    success2 = test_non_cut_images()
    
    if success1 and success2:
        print("\n🎉 All cut detection tests passed!")
        print("✅ Cut images are correctly identified as cuts")
        print("✅ Non-cut images are not misclassified as cuts")
    else:
        print("\n❌ Some cut detection tests failed.")
        if not success1:
            print("❌ Cut detection accuracy is below 80%")
        if not success2:
            print("❌ Non-cut images are being misclassified as cuts")

