#!/usr/bin/env python3
"""
Simple Test for Enhanced Agent with Internet Sources
"""

import requests
import os

def test_enhanced_agent():
    """Test the enhanced agent with internet sources."""
    print("🌐 Testing Enhanced Agent with Internet Sources")
    print("=" * 60)
    
    # Test with a cut image
    cut_image = "datasets/Cut/images/cut (1).jpg"
    if not os.path.exists(cut_image):
        print("❌ Cut image not found")
        return False
    
    try:
        print(f"📸 Testing: {os.path.basename(cut_image)}")
        
        with open(cut_image, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5000/analyze-intelligent', files=files)
        
        if response.status_code == 200:
            data = response.json()
            analysis = data['analysis']
            
            print(f"✅ Prediction: {analysis['prediction']}")
            print(f"🎯 Confidence: {analysis['confidence']:.3f}")
            print(f"🔧 Enhanced Analysis: {analysis.get('enhanced_analysis', False)}")
            print()
            
            # Show internet sources
            sources = analysis.get('internet_sources_used', [])
            print(f"🌐 Internet Sources Used: {len(sources)}")
            if sources:
                print(f"   • {', '.join(sources[:5])}")
            print()
            
            # Show medical validation
            medical_validation = analysis.get('medical_validation', {})
            if medical_validation:
                validation_data = medical_validation.get('medical_validation', {})
                if validation_data:
                    overall_score = validation_data.get('overall_match_score', 0)
                    visual_score = validation_data.get('visual_match_score', 0)
                    medical_score = validation_data.get('medical_match_score', 0)
                    
                    print("🏥 Medical Validation:")
                    print(f"   • Visual Match Score: {visual_score:.3f}")
                    print(f"   • Medical Match Score: {medical_score:.3f}")
                    print(f"   • Overall Match Score: {overall_score:.3f}")
                    
                    visual_indicators = validation_data.get('visual_indicators', [])
                    if visual_indicators:
                        print(f"   • Visual Indicators: {', '.join(visual_indicators)}")
                print()
                
                # Show source agreement
                source_agreement = medical_validation.get('source_agreement', {})
                if source_agreement:
                    print("📚 Medical Source Agreement:")
                    high_confidence_count = 0
                    for source_name, source_data in source_agreement.items():
                        confidence = source_data.get('confidence', 'unknown')
                        agreement_score = source_data.get('agreement_score', 0)
                        if confidence == 'high':
                            high_confidence_count += 1
                        print(f"   • {source_name}: {confidence} ({agreement_score:.3f})")
                    
                    print(f"   • High Confidence Sources: {high_confidence_count}/{len(source_agreement)}")
                print()
            
            # Show search results
            search_results = analysis.get('search_results', {})
            if search_results:
                sources_list = search_results.get('sources', [])
                print(f"🔍 Medical Database Sources: {len(sources_list)}")
                if sources_list:
                    for source in sources_list[:3]:
                        print(f"   • {source}")
                
                medical_insights = search_results.get('medical_insights', [])
                if medical_insights:
                    print(f"   • Medical Insights: {len(medical_insights)}")
                    for insight in medical_insights[:2]:
                        print(f"     - {insight}")
                print()
            
            # Show cut analysis
            cut_analysis = analysis.get('cut_analysis', {})
            if cut_analysis:
                cut_prob = cut_analysis.get('cut_probability', 0)
                linear_structure = cut_analysis.get('linear_structure', False)
                color_match = cut_analysis.get('color_match', False)
                
                print("🔪 Cut Analysis:")
                print(f"   • Cut Probability: {cut_prob:.3f}")
                print(f"   • Linear Structure: {linear_structure}")
                print(f"   • Color Match: {color_match}")
                print()
            
            print("🎉 Enhanced agent with internet sources working successfully!")
            return True
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_agent()

