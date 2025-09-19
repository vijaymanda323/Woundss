#!/usr/bin/env python3
"""
Test script to compare local wound analysis with ChatGPT analysis.
This helps identify discrepancies between your local model and ChatGPT.
"""

import requests
import json
import os
from pathlib import Path

def test_analysis_comparison():
    """Test the analysis comparison endpoint."""
    
    print("🔍 Wound Analysis Comparison Test")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not running")
            return
        print("✅ Backend is running")
    except requests.exceptions.RequestException:
        print("❌ Backend is not running")
        return
    
    # Find test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("❌ No test_images directory found")
        print("📁 Please create a 'test_images' directory and add some wound images")
        return
    
    # Get list of images
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_files:
        print("❌ No images found in test_images directory")
        print("📁 Please add some .jpg or .png wound images to the test_images directory")
        return
    
    print(f"📸 Found {len(image_files)} test images")
    
    # Test each image
    for i, image_path in enumerate(image_files[:3]):  # Test first 3 images
        print(f"\n🔍 Testing image {i+1}: {image_path.name}")
        print("-" * 30)
        
        try:
            # Prepare the request
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'ai_service': 'openai',
                    'api_key': 'test_key_123'  # This will fail but show the comparison structure
                }
                
                # Send request
                response = requests.post(
                    "http://localhost:5000/compare-analysis",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                comparison = result['comparison']
                
                print(f"📊 Local Analysis:")
                print(f"   Prediction: {comparison['local_analysis']['prediction']}")
                print(f"   Confidence: {comparison['local_analysis']['confidence']:.3f}")
                print(f"   Method: {comparison['local_analysis'].get('method', 'unknown')}")
                
                if 'external_analysis' in comparison and comparison['external_analysis']:
                    if 'error' in comparison['external_analysis']:
                        print(f"🤖 External AI: Error - {comparison['external_analysis']['error']}")
                    else:
                        print(f"🤖 External AI:")
                        print(f"   Prediction: {comparison['external_analysis']['prediction']}")
                        print(f"   Confidence: {comparison['external_analysis']['confidence']:.3f}")
                
                print(f"🔄 Comparison:")
                print(f"   Match: {comparison['comparison']['prediction_match']}")
                print(f"   Local Method: {comparison['comparison']['analysis_method']}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing image: {e}")
    
    print(f"\n📋 Analysis Summary")
    print("=" * 30)
    print("🔧 To get accurate ChatGPT comparison:")
    print("1. Get your OpenAI API key from: https://platform.openai.com/api-keys")
    print("2. Replace 'test_key_123' with your real API key")
    print("3. Run this test again")
    print("\n💡 The comparison will show:")
    print("   • Local model prediction vs ChatGPT prediction")
    print("   • Confidence levels for both")
    print("   • Whether predictions match")
    print("   • Analysis method used (model vs fallback)")

def test_local_analysis_only():
    """Test only local analysis without external AI."""
    
    print("\n🔍 Local Analysis Only Test")
    print("=" * 40)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not running")
            return
        print("✅ Backend is running")
    except requests.exceptions.RequestException:
        print("❌ Backend is not running")
        return
    
    # Find test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("❌ No test_images directory found")
        return
    
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_files:
        print("❌ No images found in test_images directory")
        return
    
    print(f"📸 Testing {len(image_files)} images with local analysis only")
    
    for i, image_path in enumerate(image_files[:3]):
        print(f"\n🔍 Image {i+1}: {image_path.name}")
        print("-" * 25)
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    "http://localhost:5000/analyze-intelligent",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['analysis']
                
                print(f"📊 Prediction: {analysis['prediction']}")
                print(f"🎯 Confidence: {analysis['confidence']:.3f}")
                print(f"🔧 Method: {analysis.get('method', 'unknown')}")
                
                if 'analysis_details' in analysis:
                    details = analysis['analysis_details']
                    print(f"📈 Analysis Details:")
                    if 'color_analysis' in details:
                        hsv = details['color_analysis']['hsv_mean']
                        print(f"   Color (HSV): H={hsv[0]:.1f}, S={hsv[1]:.1f}, V={hsv[2]:.1f}")
                    if 'texture_analysis' in details:
                        print(f"   Edge Density: {details['texture_analysis']['edge_density']:.3f}")
                    if 'prediction_scores' in details:
                        scores = details['prediction_scores']
                        print(f"   All Scores: {scores}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Wound Analysis Comparison Tests")
    print("=" * 60)
    
    # Test local analysis first
    test_local_analysis_only()
    
    # Test comparison (will fail without real API key)
    test_analysis_comparison()
    
    print(f"\n✅ Tests completed!")
    print("=" * 30)
    print("💡 Next steps:")
    print("1. Add wound images to 'test_images' directory")
    print("2. Get OpenAI API key for ChatGPT comparison")
    print("3. Run this test to see local vs ChatGPT differences")
    print("4. Use the comparison to improve local model accuracy")

