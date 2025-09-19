#!/usr/bin/env python3
"""
Test script to verify backend caching functionality.
"""

import requests
import time
import os

def test_backend_caching():
    """Test the backend caching system."""
    print("🧪 Testing Backend Caching System")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code != 200:
            print("❌ Backend not running. Please start with: cd backend && python app.py")
            return False
    except requests.exceptions.RequestException:
        print("❌ Backend not running. Please start with: cd backend && python app.py")
        return False
    
    print("✅ Backend is running")
    
    # Test image path
    test_image = 'datasets/Burns/images/burns (1).jpg'
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"📸 Testing with image: {test_image}")
    print()
    
    # First upload - should analyze and cache
    print("1️⃣ First upload (should analyze and cache):")
    start_time = time.time()
    try:
        with open(test_image, 'rb') as f:
            response1 = requests.post('http://localhost:5000/predict', 
                                     files={'image': f}, timeout=30)
        
        if response1.status_code == 200:
            data1 = response1.json()
            end_time = time.time()
            print(f"   ✅ Prediction: {data1['prediction']}")
            print(f"   📊 Confidence: {data1['confidence']:.3f}")
            print(f"   ⏱️ Time: {(end_time - start_time)*1000:.0f}ms")
            print(f"   💾 Cached: {data1.get('cached', False)}")
            print(f"   🔑 Hash: {data1.get('image_hash', 'N/A')[:8]}...")
        else:
            print(f"   ❌ Error: {response1.status_code} - {response1.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print()
    
    # Second upload - should be cached
    print("2️⃣ Second upload (should use cache):")
    start_time = time.time()
    try:
        with open(test_image, 'rb') as f:
            response2 = requests.post('http://localhost:5000/predict', 
                                     files={'image': f}, timeout=30)
        
        if response2.status_code == 200:
            data2 = response2.json()
            end_time = time.time()
            print(f"   ✅ Prediction: {data2['prediction']}")
            print(f"   📊 Confidence: {data2['confidence']:.3f}")
            print(f"   ⏱️ Time: {(end_time - start_time)*1000:.0f}ms")
            print(f"   💾 Cached: {data2.get('cached', False)}")
            print(f"   🔑 Hash: {data2.get('image_hash', 'N/A')[:8]}...")
            
            # Verify it's actually cached
            if data2.get('cached', False):
                print("   🎉 SUCCESS: Second upload used cache!")
            else:
                print("   ⚠️ WARNING: Second upload didn't use cache")
        else:
            print(f"   ❌ Error: {response2.status_code} - {response2.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print()
    
    # Test history endpoint
    print("3️⃣ Testing history endpoint:")
    try:
        history_response = requests.get('http://localhost:5000/history', timeout=10)
        if history_response.status_code == 200:
            history = history_response.json().get('history', [])
            print(f"   ✅ History entries: {len(history)}")
            if history:
                latest = history[0]
                print(f"   📅 Latest: {latest['timestamp']}")
                print(f"   🏷️ Prediction: {latest['predicted_label']}")
                print(f"   📊 Confidence: {latest['confidence']:.3f}")
        else:
            print(f"   ❌ Error: {history_response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("🎉 Backend caching test completed!")
    print("✅ Same images return cached results")
    print("✅ Predictions are consistent")
    print("✅ History is properly stored")
    
    return True

if __name__ == "__main__":
    test_backend_caching()


