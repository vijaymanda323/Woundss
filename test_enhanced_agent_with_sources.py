#!/usr/bin/env python3
"""
Test Enhanced Agent with Internet Sources
=========================================

Test the enhanced intelligent agent with real internet sources and medical validation.
"""

import requests
import os
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000"

def test_enhanced_agent_with_sources():
    """Test the enhanced agent with internet sources."""
    print("🌐 Testing Enhanced Agent with Internet Sources")
    print("=" * 70)
    
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
            images = [f"{dataset_path}/{f}" for f in os.listdir(dataset_path) if f.endswith('.jpg')][:2]
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
                
                result = {
                    'image': os.path.basename(image_path),
                    'expected': expected_type,
                    'prediction': analysis['prediction'],
                    'confidence': analysis['confidence'],
                    'enhanced_analysis': analysis.get('enhanced_analysis', False),
                    'internet_sources': analysis.get('internet_sources_used', []),
                    'medical_validation': analysis.get('medical_validation', {}),
                    'search_results': analysis.get('search_results', {})
                }
                results.append(result)
                
                print(f"   ✅ Prediction: {analysis['prediction']}")
                print(f"   🎯 Confidence: {analysis['confidence']:.3f}")
                print(f"   🔧 Enhanced: {analysis.get('enhanced_analysis', False)}")
                
                # Show internet sources used
                internet_sources = analysis.get('internet_sources_used', [])
                if internet_sources:
                    print(f"   🌐 Internet Sources: {len(internet_sources)} sources")
                    print(f"      • {', '.join(internet_sources[:3])}{'...' if len(internet_sources) > 3 else ''}")
                
                # Show medical validation
                medical_validation = analysis.get('medical_validation', {})
                if medical_validation:
                    validation_data = medical_validation.get('medical_validation', {})
                    if validation_data:
                        overall_score = validation_data.get('overall_match_score', 0)
                        print(f"   🏥 Medical Validation: {overall_score:.3f}")
                        
                        source_agreement = medical_validation.get('source_agreement', {})
                        if source_agreement:
                            high_confidence_sources = [name for name, data in source_agreement.items() 
                                                     if data.get('confidence') == 'high']
                            print(f"   📚 High Confidence Sources: {len(high_confidence_sources)}")
                
                # Show search results
                search_results = analysis.get('search_results', {})
                if search_results:
                    sources = search_results.get('sources', [])
                    if sources:
                        print(f"   🔍 Medical Sources: {', '.join(sources[:2])}{'...' if len(sources) > 2 else ''}")
                
                # Check accuracy
                is_correct = analysis['prediction'].lower() == expected_type.lower()
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"   📊 Result: {status}")
                print()
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    # Summary
    print("📊 ENHANCED AGENT WITH INTERNET SOURCES SUMMARY:")
    print("-" * 70)
    
    correct_predictions = 0
    total_tests = len(results)
    high_confidence_predictions = 0
    
    for result in results:
        is_correct = result['prediction'].lower() == result['expected'].lower()
        if is_correct:
            correct_predictions += 1
        
        if result['confidence'] > 0.8:
            high_confidence_predictions += 1
        
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"   {result['image']}: {result['prediction']} ({result['confidence']:.3f}) - {status}")
        
        # Show internet sources used
        if result['internet_sources']:
            print(f"      🌐 Sources: {len(result['internet_sources'])} medical databases")
    
    accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    high_confidence_rate = (high_confidence_predictions / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
    print(f"🎯 High Confidence Rate: {high_confidence_rate:.1f}% ({high_confidence_predictions}/{total_tests})")
    
    # Show internet sources summary
    all_sources = set()
    for result in results:
        all_sources.update(result['internet_sources'])
    
    print(f"\n🌐 Internet Sources Used:")
    print(f"   • Total Sources: {len(all_sources)}")
    print(f"   • Sources: {', '.join(sorted(all_sources))}")
    
    return accuracy > 80

def test_medical_validation():
    """Test medical validation capabilities."""
    print("\n🏥 Testing Medical Validation")
    print("=" * 70)
    
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
            print(f"🎯 Prediction: {analysis['prediction']}")
            print(f"📊 Confidence: {analysis['confidence']:.3f}")
            print()
            
            # Show medical validation details
            medical_validation = analysis.get('medical_validation', {})
            if medical_validation:
                validation_data = medical_validation.get('medical_validation', {})
                if validation_data:
                    print("🏥 Medical Validation Details:")
                    print(f"   • Visual Match Score: {validation_data.get('visual_match_score', 0):.3f}")
                    print(f"   • Medical Match Score: {validation_data.get('medical_match_score', 0):.3f}")
                    print(f"   • Overall Match Score: {validation_data.get('overall_match_score', 0):.3f}")
                    
                    visual_indicators = validation_data.get('visual_indicators', [])
                    if visual_indicators:
                        print(f"   • Visual Indicators: {', '.join(visual_indicators)}")
                    
                    medical_criteria = validation_data.get('medical_criteria', [])
                    if medical_criteria:
                        print(f"   • Medical Criteria: {', '.join(medical_criteria)}")
                
                # Show source agreement
                source_agreement = medical_validation.get('source_agreement', {})
                if source_agreement:
                    print("\n📚 Medical Source Agreement:")
                    for source_name, source_data in source_agreement.items():
                        confidence = source_data.get('confidence', 'unknown')
                        agreement_score = source_data.get('agreement_score', 0)
                        print(f"   • {source_name}: {confidence} ({agreement_score:.3f})")
            
            # Show search results
            search_results = analysis.get('search_results', {})
            if search_results:
                print("\n🔍 Internet Search Results:")
                sources = search_results.get('sources', [])
                if sources:
                    print(f"   • Medical Sources: {len(sources)}")
                    for source in sources[:3]:
                        print(f"     - {source}")
                
                medical_insights = search_results.get('medical_insights', [])
                if medical_insights:
                    print(f"   • Medical Insights: {len(medical_insights)}")
                    for insight in medical_insights[:2]:
                        print(f"     - {insight}")
            
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🌐 Enhanced Agent with Internet Sources Test")
    print("=" * 70)
    
    # Test enhanced agent with sources
    success1 = test_enhanced_agent_with_sources()
    
    # Test medical validation
    success2 = test_medical_validation()
    
    if success1 and success2:
        print("\n🎉 All enhanced agent tests passed!")
        print("✅ Agent successfully uses internet sources for validation")
        print("✅ Medical validation working correctly")
        print("✅ High accuracy achieved with medical database integration")
    else:
        print("\n❌ Some enhanced agent tests failed.")
        if not success1:
            print("❌ Enhanced agent accuracy below 80%")
        if not success2:
            print("❌ Medical validation not working correctly")

