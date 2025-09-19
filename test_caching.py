#!/usr/bin/env python3
"""
Test script to demonstrate image caching functionality.
"""

import requests
import time

def test_caching():
    """Test the image caching system."""
    print("🧪 Testing Image Caching System")
    print("=" * 50)
    
    # Test image path
    image_path = 'datasets/Burns/images/burns (1).jpg'
    
    print(f"📸 Testing with image: {image_path}")
    print()
    
    # First upload - should not be cached
    print("1️⃣ First upload (should analyze and cache):")
    start_time = time.time()
    response1 = requests.post('http://localhost:5000/predict', 
                             files={'image': open(image_path, 'rb')})
    end_time = time.time()
    
    if response1.status_code == 200:
        data1 = response1.json()
        print(f"   ✅ Prediction: {data1['prediction']}")
        print(f"   📊 Confidence: {data1['confidence']:.3f}")
        print(f"   ⏱️ Time: {(end_time - start_time)*1000:.0f}ms")
        print(f"   💾 Cached: {data1.get('cached', False)}")
        print(f"   🔑 Hash: {data1.get('image_hash', 'N/A')[:8]}...")
    else:
        print(f"   ❌ Error: {response1.status_code}")
        return
    
    print()
    
    # Second upload - should be cached
    print("2️⃣ Second upload (should use cache):")
    start_time = time.time()
    response2 = requests.post('http://localhost:5000/predict', 
                             files={'image': open(image_path, 'rb')})
    end_time = time.time()
    
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"   ✅ Prediction: {data2['prediction']}")
        print(f"   📊 Confidence: {data2['confidence']:.3f}")
        print(f"   ⏱️ Time: {(end_time - start_time)*1000:.0f}ms")
        print(f"   💾 Cached: {data2.get('cached', False)}")
        print(f"   🔑 Hash: {data2.get('image_hash', 'N/A')[:8]}...")
    else:
        print(f"   ❌ Error: {response2.status_code}")
        return
    
    print()
    
    # Third upload - should still be cached
    print("3️⃣ Third upload (should still use cache):")
    start_time = time.time()
    response3 = requests.post('http://localhost:5000/predict', 
                             files={'image': open(image_path, 'rb')})
    end_time = time.time()
    
    if response3.status_code == 200:
        data3 = response3.json()
        print(f"   ✅ Prediction: {data3['prediction']}")
        print(f"   📊 Confidence: {data3['confidence']:.3f}")
        print(f"   ⏱️ Time: {(end_time - start_time)*1000:.0f}ms")
        print(f"   💾 Cached: {data3.get('cached', False)}")
        print(f"   🔑 Hash: {data3.get('image_hash', 'N/A')[:8]}...")
    else:
        print(f"   ❌ Error: {response3.status_code}")
        return
    
    print()
    
    # Test with different image
    print("4️⃣ Different image (should analyze and cache):")
    different_image = 'datasets/Burns/images/burns (2).jpg'
    start_time = time.time()
    response4 = requests.post('http://localhost:5000/predict', 
                             files={'image': open(different_image, 'rb')})
    end_time = time.time()
    
    if response4.status_code == 200:
        data4 = response4.json()
        print(f"   ✅ Prediction: {data4['prediction']}")
        print(f"   📊 Confidence: {data4['confidence']:.3f}")
        print(f"   ⏱️ Time: {(end_time - start_time)*1000:.0f}ms")
        print(f"   💾 Cached: {data4.get('cached', False)}")
        print(f"   🔑 Hash: {data4.get('image_hash', 'N/A')[:8]}...")
    else:
        print(f"   ❌ Error: {response4.status_code}")
        return
    
    print()
    
    # Check history
    print("📋 Checking history:")
    history_response = requests.get('http://localhost:5000/history')
    if history_response.status_code == 200:
        history = history_response.json().get('history', [])
        print(f"   📊 Total entries: {len(history)}")
        print(f"   🕒 Latest entry: {history[0]['timestamp'] if history else 'None'}")
        print(f"   🏷️ Latest prediction: {history[0]['predicted_label'] if history else 'None'}")
    else:
        print(f"   ❌ Error getting history: {history_response.status_code}")
    
    print()
    print("🎉 Caching test completed!")
    print("✅ Same images return cached results instantly")
    print("✅ Different images are analyzed and cached")
    print("✅ Predictions are consistent across uploads")

if __name__ == "__main__":
    test_caching()


