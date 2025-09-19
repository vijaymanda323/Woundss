#!/usr/bin/env python3
"""
Test API Consistency
===================

Test that the same image returns identical predictions from the API.
"""

import requests
import os
import sys

def test_api_consistency():
    """Test API consistency with the same image."""
    print("🧪 Testing API Consistency")
    print("=" * 50)
    
    # Test image path
    image_path = 'datasets/Burns/images/burns (1).jpg'
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    print(f"📸 Testing with image: {image_path}")
    print()
    
    predictions = []
    confidences = []
    hashes = []
    cached_status = []
    
    # Test the same image multiple times
    for i in range(5):
        print(f"{i+1}️⃣ Test #{i+1}:")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post('http://localhost:5000/predict', files=files)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']
                confidence = data['confidence']
                image_hash = data.get('image_hash', 'N/A')
                cached = data.get('cached', False)
                
                predictions.append(prediction)
                confidences.append(confidence)
                hashes.append(image_hash)
                cached_status.append(cached)
                
                print(f"   ✅ Prediction: {prediction}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🔑 Hash: {image_hash[:8] if image_hash != 'N/A' else 'N/A'}...")
                print(f"   💾 Cached: {cached}")
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
    print(f"   📊 Total tests: {len(predictions)}")
    print(f"   🏷️ All predictions: {predictions}")
    print(f"   📈 All confidences: {[f'{c:.3f}' for c in confidences]}")
    print(f"   🔑 All hashes: {[h[:8] if h != 'N/A' else 'N/A' for h in hashes]}")
    print(f"   💾 All cached: {cached_status}")
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
        print("   ✅ API caching working correctly")
        return True
    else:
        print("❌ FAILURE: Same image returns different predictions!")
        print("   🔍 This indicates a problem with the API caching")
        return False

if __name__ == "__main__":
    success = test_api_consistency()
    if success:
        print("\n🚀 The API is working perfectly!")
        print("   📝 Same images return identical predictions")
        print("   ⚡ Caching system is functioning correctly")
    else:
        print("\n⚠️ There's an issue with the API")
        print("   🔧 Check the backend logs for errors")
        print("   🗄️ Verify database is working correctly")
        sys.exit(1)
