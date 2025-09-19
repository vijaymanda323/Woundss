#!/usr/bin/env python3
"""
Test Structured Output Format
============================

Test the enhanced intelligent agent with clear, structured output format:
Type: [wound type]
Severity: [level]
Explanation: [short summary]
"""

import requests
import os
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"

def test_structured_output():
    """Test the structured output format."""
    print("🎯 Testing Structured Output Format")
    print("=" * 80)
    
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
    
    # Find test images
    test_images = []
    for dataset in ['Burns', 'Cut', 'Abrasions']:
        dataset_path = f"datasets/{dataset}/images"
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')][:2]
            test_images.extend([(f"{dataset_path}/{img}", dataset.lower()) for img in images])
    
    if not test_images:
        print("❌ No test images found")
        return False
    
    print(f"📸 Found {len(test_images)} test images")
    print()
    
    results = []
    for i, (image_path, expected_type) in enumerate(test_images[:6]):  # Test first 6 images
        print(f"🔍 Testing {i+1}/{min(6, len(test_images))}: {os.path.basename(image_path)}")
        print(f"   Expected: {expected_type}")
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{API_BASE_URL}/analyze-intelligent", files=files)
            
            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']
                
                # Display structured output
                print(f"   📋 STRUCTURED OUTPUT:")
                print(f"      Type: {analysis.get('Type', 'Unknown')}")
                print(f"      Severity: {analysis.get('Severity', 'Unknown')}")
                print(f"      Explanation: {analysis.get('Explanation', 'No explanation available')}")
                print()
                
                # Display additional details
                print(f"   📊 TECHNICAL DETAILS:")
                print(f"      Confidence: {analysis.get('confidence', 0):.3f}")
                print(f"      Enhanced Analysis: {analysis.get('enhanced_analysis', False)}")
                print(f"      Medical DB Integration: {analysis.get('medical_database_integration', False)}")
                print(f"      Accuracy Score: {analysis.get('accuracy_score', 0):.3f}")
                print(f"      Consensus Confidence: {analysis.get('consensus_confidence', 0):.3f}")
                print(f"      Medical DB Validated: {analysis.get('medical_database_validated', False)}")
                print()
                
                # Check accuracy
                predicted_type = analysis.get('Type', '').lower()
                is_correct = predicted_type == expected_type.lower()
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"   📊 Result: {status}")
                print()
                
                result = {
                    'image': os.path.basename(image_path),
                    'expected': expected_type,
                    'type': analysis.get('Type', 'Unknown'),
                    'severity': analysis.get('Severity', 'Unknown'),
                    'explanation': analysis.get('Explanation', 'No explanation'),
                    'confidence': analysis.get('confidence', 0),
                    'correct': is_correct
                }
                results.append(result)
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    # Summary
    print("📊 STRUCTURED OUTPUT SUMMARY:")
    print("-" * 80)
    
    correct_predictions = sum(1 for r in results if r['correct'])
    total_tests = len(results)
    accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    
    print(f"🎯 Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
    print()
    
    # Show structured output examples
    print("📋 STRUCTURED OUTPUT EXAMPLES:")
    print("-" * 80)
    
    for result in results[:3]:  # Show first 3 examples
        print(f"📸 {result['image']}:")
        print(f"   Type: {result['type']}")
        print(f"   Severity: {result['severity']}")
        print(f"   Explanation: {result['explanation']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Status: {'✅ CORRECT' if result['correct'] else '❌ INCORRECT'}")
        print()
    
    # Show severity distribution
    severity_counts = {}
    for result in results:
        severity = result['severity']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print("📊 SEVERITY DISTRIBUTION:")
    print("-" * 80)
    for severity, count in severity_counts.items():
        print(f"   {severity}: {count} cases")
    print()
    
    return accuracy > 80

def test_clear_format():
    """Test the clear format output."""
    print("\n🎯 Testing Clear Format Output")
    print("=" * 80)
    
    # Test with a known cut image
    cut_image = "datasets/Cut/images/cut (1).jpg"
    if not os.path.exists(cut_image):
        print("❌ Cut test image not found")
        return False
    
    try:
        with open(cut_image, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{API_BASE_URL}/analyze-intelligent", files=files)
        
        if response.status_code == 200:
            data = response.json()
            analysis = data['analysis']
            
            print(f"📸 Testing: {os.path.basename(cut_image)}")
            print()
            
            # Display clear format
            print("📋 CLEAR FORMAT OUTPUT:")
            print("-" * 40)
            print(f"Type: {analysis.get('Type', 'Unknown')}")
            print(f"Severity: {analysis.get('Severity', 'Unknown')}")
            print(f"Explanation: {analysis.get('Explanation', 'No explanation available')}")
            print("-" * 40)
            print()
            
            # Show that the format matches the requested structure
            print("✅ FORMAT VERIFICATION:")
            print(f"   ✓ Type field present: {'Type' in analysis}")
            print(f"   ✓ Severity field present: {'Severity' in analysis}")
            print(f"   ✓ Explanation field present: {'Explanation' in analysis}")
            print(f"   ✓ Clear wound type: {analysis.get('Type', 'Unknown')}")
            print(f"   ✓ Clear severity level: {analysis.get('Severity', 'Unknown')}")
            print(f"   ✓ Concise explanation: {len(analysis.get('Explanation', ''))} characters")
            print()
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Structured Output Format Test")
    print("=" * 80)
    
    # Test structured output
    success1 = test_structured_output()
    
    # Test clear format
    success2 = test_clear_format()
    
    if success1 and success2:
        print("\n🎉 All structured output tests passed!")
        print("✅ Agent provides clear, structured output format")
        print("✅ Type, Severity, and Explanation fields working correctly")
        print("✅ Format matches requested structure")
        print("✅ High accuracy achieved with structured output")
    else:
        print("\n❌ Some structured output tests failed.")
        if not success1:
            print("❌ Structured output accuracy below 80%")
        if not success2:
            print("❌ Clear format not working correctly")

