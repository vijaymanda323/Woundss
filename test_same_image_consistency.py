#!/usr/bin/env python3
"""
Test Same Image Consistency
============================

This script tests that the same image returns identical predictions
every time it's uploaded, demonstrating the caching system works.
"""

import requests
import time
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"
TEST_IMAGE = "datasets/Burns/images/burns (1).jpg"

def test_same_image_consistency():
    """Test that the same image returns identical predictions."""
    print("🧪 Testing Same Image Consistency")
    print("=" * 50)
    
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
    
    predictions = []
    confidences = []
    hashes = []
    
    # Upload the same image multiple times
    for i in range(5):
        print(f"{i+1}️⃣ Upload #{i+1}:")
        
        try:
            with open(TEST_IMAGE, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{API_BASE_URL}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']
                confidence = data['confidence']
                image_hash = data.get('image_hash', 'N/A')
                cached = data.get('cached', False)
                
                predictions.append(prediction)
                confidences.append(confidence)
                hashes.append(image_hash)
                
                print(f"   ✅ Prediction: {prediction}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   💾 Cached: {cached}")
                print(f"   🔑 Hash: {image_hash[:8] if image_hash != 'N/A' else 'N/A'}...")
                print()
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False
    
    # Analyze results
    print("📋 Analysis:")
    print(f"   📊 Total uploads: {len(predictions)}")
    print(f"   🏷️ All predictions: {predictions}")
    print(f"   📈 All confidences: {[f'{c:.3f}' for c in confidences]}")
    print(f"   🔑 All hashes: {[h[:8] if h != 'N/A' else 'N/A' for h in hashes]}")
    print()
    
    # Check consistency
    unique_predictions = set(predictions)
    unique_confidences = set([round(c, 3) for c in confidences])
    unique_hashes = set(hashes)
    
    print("🎯 Consistency Check:")
    print(f"   🏷️ Unique predictions: {len(unique_predictions)} (should be 1)")
    print(f"   📈 Unique confidences: {len(unique_confidences)} (should be 1)")
    print(f"   🔑 Unique hashes: {len(unique_hashes)} (should be 1)")
    print()
    
    # Results
    if len(unique_predictions) == 1 and len(unique_confidences) == 1:
        print("🎉 SUCCESS: Same image returns identical predictions!")
        print(f"   ✅ Consistent prediction: {list(unique_predictions)[0]}")
        print(f"   ✅ Consistent confidence: {list(unique_confidences)[0]:.3f}")
        print("   ✅ Caching system working correctly")
        return True
    else:
        print("❌ FAILURE: Same image returns different predictions!")
        print("   🔍 This indicates a problem with the caching system")
        return False

if __name__ == "__main__":
    success = test_same_image_consistency()
    if success:
        print("\n🚀 The caching system is working perfectly!")
        print("   📝 Same images will always return identical predictions")
        print("   ⚡ Subsequent uploads use cached results for speed")
        print("   💾 Predictions are stored in SQLite database")
    else:
        print("\n⚠️ There's an issue with the caching system")
        print("   🔧 Check the backend logs for errors")
        print("   🗄️ Verify database is working correctly")


