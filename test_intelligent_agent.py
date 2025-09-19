#!/usr/bin/env python3
"""
Test Intelligent Wound Analysis Agent
=====================================

Test the intelligent agent with sample images.
"""

import requests
import os
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"
TEST_IMAGE = "datasets/Burns/images/burns (1).jpg"

def test_intelligent_analysis():
    """Test the intelligent analysis endpoint."""
    print("🤖 Testing Intelligent Wound Analysis Agent")
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
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    print(f"📸 Testing with image: {TEST_IMAGE}")
    print()
    
    try:
        # Test intelligent analysis
        with open(TEST_IMAGE, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{API_BASE_URL}/analyze-intelligent", files=files)
        
        if response.status_code == 200:
            data = response.json()
            analysis = data['analysis']
            
            print("🎯 INTELLIGENT ANALYSIS RESULTS:")
            print("-" * 40)
            print(f"📊 Prediction: {analysis['prediction']}")
            print(f"🎯 Confidence: {analysis['confidence']:.3f}")
            print(f"⏰ Timestamp: {analysis['timestamp']}")
            print()
            
            # Display detailed analysis
            if 'features' in analysis:
                features = analysis['features']
                print("🔍 DETAILED FEATURE ANALYSIS:")
                print("-" * 40)
                
                # Color analysis
                if 'color_analysis' in features:
                    color = features['color_analysis']
                    print(f"🎨 Color Analysis:")
                    print(f"   - Mean HSV: {color['mean_hsv']}")
                    print(f"   - Color Variance: {color['color_variance']:.3f}")
                
                # Texture analysis
                if 'texture_analysis' in features:
                    texture = features['texture_analysis']
                    print(f"📐 Texture Analysis:")
                    print(f"   - Edge Density: {texture['edge_density']:.3f}")
                    print(f"   - Smoothness: {texture['smoothness']:.3f}")
                
                # Shape analysis
                if 'shape_analysis' in features:
                    shape = features['shape_analysis']
                    print(f"🔷 Shape Analysis:")
                    print(f"   - Area: {shape['area']:.0f} pixels")
                    print(f"   - Circularity: {shape['circularity']:.3f}")
                    print(f"   - Aspect Ratio: {shape['aspect_ratio']:.3f}")
                
                # Size analysis
                if 'size_analysis' in features:
                    size = features['size_analysis']
                    print(f"📏 Size Analysis:")
                    print(f"   - Wound Percentage: {size['wound_percentage']:.3f}")
                    print(f"   - Estimated Area: {size['estimated_area_cm2']:.2f} cm²")
            
            # Display reasoning
            if 'reasoning' in analysis:
                reasoning = analysis['reasoning']
                print()
                print("🧠 INTELLIGENT REASONING:")
                print("-" * 40)
                
                if 'primary_indicators' in reasoning:
                    print("🎯 Primary Indicators:")
                    for indicator in reasoning['primary_indicators']:
                        print(f"   • {indicator}")
                
                if 'supporting_evidence' in reasoning:
                    print("📋 Supporting Evidence:")
                    for evidence in reasoning['supporting_evidence']:
                        print(f"   • {evidence}")
                
                if 'confidence_factors' in reasoning:
                    print("✅ Confidence Factors:")
                    for factor in reasoning['confidence_factors']:
                        print(f"   • {factor}")
            
            # Display severity assessment
            if 'severity_assessment' in analysis:
                severity = analysis['severity_assessment']
                print()
                print("⚠️ SEVERITY ASSESSMENT:")
                print("-" * 40)
                print(f"📊 Severity Level: {severity['level']}")
                print(f"🎯 Severity Score: {severity['score']:.3f}")
                print(f"📋 Factors: {severity['factors']}")
            
            # Display treatment recommendations
            if 'treatment_recommendations' in analysis:
                recommendations = analysis['treatment_recommendations']
                print()
                print("💊 TREATMENT RECOMMENDATIONS:")
                print("-" * 40)
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            # Display healing timeline
            if 'healing_timeline' in analysis:
                timeline = analysis['healing_timeline']
                print()
                print("⏰ HEALING TIMELINE:")
                print("-" * 40)
                print(f"📅 Estimated Days: {timeline['estimated_days']}")
                print(f"📊 Range: {timeline['range_days']} days")
                print(f"🎯 Confidence: {timeline['confidence']}")
            
            # Display risk factors
            if 'risk_factors' in analysis:
                risks = analysis['risk_factors']
                if risks:
                    print()
                    print("⚠️ RISK FACTORS:")
                    print("-" * 40)
                    for risk in risks:
                        print(f"   • {risk}")
            
            print()
            print("🎉 Intelligent analysis completed successfully!")
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def test_multiple_images():
    """Test with multiple images."""
    print("\n🔄 Testing Multiple Images")
    print("=" * 60)
    
    # Find test images
    test_images = []
    for dataset in ['Burns', 'Cut', 'Abrasions']:
        dataset_path = f"datasets/{dataset}/images"
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')][:2]
            test_images.extend([f"{dataset_path}/{img}" for img in images])
    
    if not test_images:
        print("❌ No test images found")
        return False
    
    print(f"📸 Found {len(test_images)} test images")
    print()
    
    results = []
    for i, image_path in enumerate(test_images[:5]):  # Test first 5 images
        print(f"🔍 Testing {i+1}/{min(5, len(test_images))}: {os.path.basename(image_path)}")
        
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
                    'severity': analysis.get('severity_assessment', {}).get('level', 'unknown')
                }
                results.append(result)
                
                print(f"   ✅ {analysis['prediction']} (confidence: {analysis['confidence']:.3f})")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print()
    print("📊 SUMMARY:")
    print("-" * 40)
    for result in results:
        print(f"   {result['image']}: {result['prediction']} ({result['confidence']:.3f}) - {result['severity']}")
    
    return len(results) > 0

if __name__ == "__main__":
    print("🤖 Intelligent Wound Analysis Agent Test")
    print("=" * 60)
    
    # Test single image
    success1 = test_intelligent_analysis()
    
    # Test multiple images
    success2 = test_multiple_images()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The intelligent agent is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

