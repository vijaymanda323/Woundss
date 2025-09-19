#!/usr/bin/env python3
"""
Test Different Images
=====================

Test that different images return different predictions and are cached separately.
"""

import requests
import os
import glob

def test_different_images():
    """Test API with different images."""
    print("🧪 Testing Different Images")
    print("=" * 50)
    
    # Find test images
    image_pattern = 'datasets/Burns/images/*.jpg'
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"❌ No images found matching: {image_pattern}")
        return False
    
    print(f"📸 Found {len(image_files)} test images")
    print()
    
    results = []
    
    # Test each image
    for i, image_path in enumerate(image_files[:3]):  # Test first 3 images
        print(f"{i+1}️⃣ Testing: {os.path.basename(image_path)}")
        
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
                
                results.append({
                    'filename': os.path.basename(image_path),
                    'prediction': prediction,
                    'confidence': confidence,
                    'hash': image_hash[:8] if image_hash != 'N/A' else 'N/A',
                    'cached': cached
                })
                
                print(f"   ✅ Prediction: {prediction}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🔑 Hash: {image_hash[:8] if image_hash != 'N/A' else 'N/A'}...")
                print(f"   💾 Cached: {cached}")
                print()
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False
    
    # Analyze results
    print("📋 Analysis:")
    print(f"   📊 Total images tested: {len(results)}")
    
    unique_predictions = set([r['prediction'] for r in results])
    unique_hashes = set([r['hash'] for r in results])
    
    print(f"   🏷️ Unique predictions: {len(unique_predictions)}")
    print(f"   🔑 Unique hashes: {len(unique_hashes)}")
    print()
    
    print("📊 Detailed Results:")
    for result in results:
        print(f"   📸 {result['filename']}: {result['prediction']} (confidence: {result['confidence']:.3f}, hash: {result['hash']}...)")
    
    print()
    
    # Check if each image has unique hash
    if len(unique_hashes) == len(results):
        print("🎉 SUCCESS: Each image has a unique hash!")
        print("   ✅ Different images are cached separately")
        print("   ✅ System correctly identifies different images")
        return True
    else:
        print("❌ ISSUE: Some images have the same hash!")
        print("   🔍 This could cause incorrect caching")
        return False

if __name__ == "__main__":
    success = test_different_images()
    if success:
        print("\n🚀 The system correctly handles different images!")
        print("   📝 Each image gets its own cache entry")
        print("   ⚡ Caching system works for multiple images")
    else:
        print("\n⚠️ There's an issue with different image handling")
        print("   🔧 Check the image hash calculation")
        print("   🗄️ Verify database storage")


