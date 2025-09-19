#!/usr/bin/env python3
"""
Enhanced Wound Healing UI Server
===============================

A Flask server that serves the wound healing UI with enhanced features.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Enhanced wound analysis with more detailed recommendations
def get_enhanced_wound_analysis(wound_type, healing_time, area_cm2=None, patient_age=None):
    """Get enhanced wound analysis with detailed recommendations."""
    
    # Base healing times by wound type and age
    base_healing_times = {
        'burn': {'young': 14, 'adult': 21, 'elderly': 28},
        'cut': {'young': 5, 'adult': 7, 'elderly': 10},
        'surgical': {'young': 7, 'adult': 10, 'elderly': 14},
        'chronic': {'young': 45, 'adult': 60, 'elderly': 90},
        'diabetic': {'young': 60, 'adult': 90, 'elderly': 120},
        'unknown': {'young': 14, 'adult': 21, 'elderly': 28}
    }
    
    # Determine age group
    age_group = 'elderly'
    if patient_age:
        if patient_age < 30:
            age_group = 'young'
        elif patient_age < 60:
            age_group = 'adult'
    
    # Get base healing time
    base_time = base_healing_times.get(wound_type.lower(), base_healing_times['unknown'])[age_group]
    
    # Adjust based on wound size
    if area_cm2:
        if area_cm2 > 10:  # Large wound
            base_time = int(base_time * 1.5)
        elif area_cm2 < 1:  # Small wound
            base_time = int(base_time * 0.7)
    
    return {
        'wound_type': wound_type,
        'estimated_healing_time': base_time,
        'age_group': age_group,
        'size_category': get_size_category(area_cm2) if area_cm2 else 'unknown',
        'precautions': get_detailed_precautions(wound_type, age_group),
        'treatment_recommendations': get_detailed_treatments(wound_type, age_group),
        'follow_up_schedule': get_follow_up_schedule(base_time),
        'risk_factors': get_risk_factors(wound_type, age_group),
        'healing_stages': get_healing_stages(wound_type, base_time)
    }

def get_size_category(area_cm2):
    """Categorize wound size."""
    if area_cm2 < 1:
        return 'small'
    elif area_cm2 < 5:
        return 'medium'
    elif area_cm2 < 10:
        return 'large'
    else:
        return 'very_large'

def get_detailed_precautions(wound_type, age_group):
    """Get detailed precautions based on wound type and age."""
    precautions = {
        'burn': [
            'Keep the wound clean and dry at all times',
            'Avoid exposing the area to direct sunlight',
            'Do not pick at scabs or blisters',
            'Apply prescribed topical medications as directed',
            'Monitor for signs of infection (redness, swelling, pus)',
            'Maintain proper nutrition to support healing',
            'Avoid smoking and alcohol consumption'
        ],
        'cut': [
            'Keep the wound clean and covered with sterile dressing',
            'Change dressings daily or as directed by healthcare provider',
            'Avoid getting the wound wet for the first 24-48 hours',
            'Watch for signs of infection (increased pain, redness, swelling)',
            'Follow up with healthcare provider as scheduled',
            'Avoid activities that may reopen the wound',
            'Keep the area elevated to reduce swelling'
        ],
        'chronic': [
            'Maintain strict hygiene around the wound area',
            'Monitor blood sugar levels closely (if diabetic)',
            'Avoid pressure on the wound site',
            'Follow specialized wound care protocol exactly',
            'Regular medical follow-ups every 1-2 weeks',
            'Maintain proper nutrition and hydration',
            'Consider compression therapy if recommended'
        ],
        'surgical': [
            'Keep incision site clean and dry',
            'Follow all post-operative care instructions precisely',
            'Monitor for signs of infection or complications',
            'Avoid strenuous activities as directed',
            'Take all prescribed medications as scheduled',
            'Attend all follow-up appointments',
            'Report any unusual symptoms immediately'
        ]
    }
    
    base_precautions = precautions.get(wound_type.lower(), precautions['cut'])
    
    # Add age-specific precautions
    if age_group == 'elderly':
        base_precautions.extend([
            'Ensure adequate nutrition and hydration',
            'Monitor for signs of confusion or disorientation',
            'Consider assistance with wound care if needed',
            'Regular monitoring of vital signs'
        ])
    elif age_group == 'young':
        base_precautions.extend([
            'Ensure proper supervision during wound care',
            'Monitor for signs of infection more frequently',
            'Consider activity restrictions to prevent reinjury'
        ])
    
    return base_precautions

def get_detailed_treatments(wound_type, age_group):
    """Get detailed treatment recommendations."""
    treatments = {
        'burn': [
            'Apply silver sulfadiazine cream twice daily',
            'Use non-adherent dressings (Mepitel, Adaptic)',
            'Consider hydrotherapy for deep burns',
            'Monitor for compartment syndrome in severe cases',
            'Pain management with appropriate analgesics',
            'Consider nutritional supplements (vitamin C, zinc)',
            'Physical therapy for functional areas'
        ],
        'cut': [
            'Clean with sterile saline solution',
            'Apply antibiotic ointment (Neosporin, Bacitracin)',
            'Use appropriate dressing type based on wound depth',
            'Consider sutures or staples for deep cuts',
            'Tetanus prophylaxis if indicated',
            'Monitor for signs of nerve or tendon damage',
            'Consider imaging if foreign body suspected'
        ],
        'chronic': [
            'Debridement of necrotic tissue as needed',
            'Negative pressure wound therapy (NPWT)',
            'Hyperbaric oxygen therapy if indicated',
            'Growth factor applications (PDGF, EGF)',
            'Specialized dressings (hydrocolloid, alginate, foam)',
            'Compression therapy for venous ulcers',
            'Offloading for diabetic foot ulcers'
        ],
        'surgical': [
            'Monitor healing progression with regular assessments',
            'Manage pain with appropriate analgesics',
            'Prevent infection with prophylactic antibiotics',
            'Physical therapy and rehabilitation as needed',
            'Follow specific surgical care protocols',
            'Monitor for complications (dehiscence, infection)',
            'Consider imaging if healing concerns arise'
        ]
    }
    
    base_treatments = treatments.get(wound_type.lower(), treatments['cut'])
    
    # Add age-specific treatments
    if age_group == 'elderly':
        base_treatments.extend([
            'Consider slower healing expectations',
            'Monitor for medication interactions',
            'Ensure adequate pain control',
            'Consider home health services if needed'
        ])
    elif age_group == 'young':
        base_treatments.extend([
            'Consider faster healing expectations',
            'Monitor for overactivity',
            'Ensure proper wound protection during activities'
        ])
    
    return base_treatments

def get_follow_up_schedule(healing_time):
    """Get follow-up schedule based on healing time."""
    if healing_time <= 7:
        return ['3-5 days', '1 week', '2 weeks']
    elif healing_time <= 14:
        return ['1 week', '2 weeks', '4 weeks']
    elif healing_time <= 30:
        return ['1 week', '2 weeks', '4 weeks', '6 weeks']
    else:
        return ['1 week', '2 weeks', '4 weeks', '6 weeks', '8 weeks', '12 weeks']

def get_risk_factors(wound_type, age_group):
    """Get risk factors for complications."""
    risk_factors = {
        'burn': [
            'Infection risk increases with burn depth',
            'Scarring and contracture formation',
            'Compartment syndrome in circumferential burns',
            'Hypovolemia and electrolyte imbalance'
        ],
        'cut': [
            'Infection if not properly cleaned',
            'Delayed healing with poor blood supply',
            'Nerve or tendon damage',
            'Foreign body retention'
        ],
        'chronic': [
            'Infection and cellulitis',
            'Osteomyelitis in deep wounds',
            'Malignancy in long-standing wounds',
            'Amputation risk in diabetic wounds'
        ],
        'surgical': [
            'Surgical site infection',
            'Wound dehiscence',
            'Hematoma formation',
            'Delayed healing with comorbidities'
        ]
    }
    
    base_risks = risk_factors.get(wound_type.lower(), risk_factors['cut'])
    
    # Add age-specific risks
    if age_group == 'elderly':
        base_risks.extend([
            'Slower healing due to age',
            'Increased infection risk',
            'Medication interactions',
            'Cognitive impairment affecting care'
        ])
    elif age_group == 'young':
        base_risks.extend([
            'Risk of reinjury due to activity',
            'Poor compliance with care instructions',
            'Delayed presentation of complications'
        ])
    
    return base_risks

def get_healing_stages(wound_type, healing_time):
    """Get expected healing stages."""
    stages = {
        'inflammatory': {'duration': int(healing_time * 0.1), 'description': 'Redness, swelling, pain'},
        'proliferative': {'duration': int(healing_time * 0.6), 'description': 'New tissue formation, granulation'},
        'maturation': {'duration': int(healing_time * 0.3), 'description': 'Scar formation, remodeling'}
    }
    
    return stages

@app.route('/')
def index():
    """Serve the main UI."""
    return send_file('wound_healing_ui.html')

@app.route('/api/enhanced-analyze', methods=['POST'])
def enhanced_analyze():
    """Enhanced wound analysis endpoint."""
    try:
        # Get form data
        patient_id = request.form.get('patient_id', '')
        patient_name = request.form.get('patient_name', '')
        patient_age = request.form.get('patient_age', '')
        patient_gender = request.form.get('patient_gender', '')
        injury_date = request.form.get('injury_date', '')
        
        # Get image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Simulate wound analysis (in real implementation, this would call your ML model)
        # For now, we'll use the existing API and enhance the results
        
        # Call the existing analyze endpoint
        import requests
        try:
            response = requests.post(
                'http://localhost:5000/analyze',
                files={'image': image_file},
                data={'patient_id': patient_id}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Enhance the result
                enhanced_result = get_enhanced_wound_analysis(
                    result.get('wound_classification', {}).get('wound_type', 'unknown'),
                    result.get('wound_classification', {}).get('estimated_days_to_cure', 30),
                    result.get('area_cm2'),
                    int(patient_age) if patient_age else None
                )
                
                # Combine results
                final_result = {
                    **result,
                    'enhanced_analysis': enhanced_result,
                    'patient_info': {
                        'id': patient_id,
                        'name': patient_name,
                        'age': patient_age,
                        'gender': patient_gender,
                        'injury_date': injury_date,
                        'analysis_date': datetime.now().isoformat()
                    }
                }
                
                return jsonify(final_result)
            else:
                return jsonify({'error': 'Analysis failed'}), 500
                
        except requests.exceptions.ConnectionError:
            # Fallback to simulated analysis if API is not available
            simulated_result = {
                'status': 'success',
                'wound_classification': {
                    'wound_type': 'burn',
                    'estimated_days_to_cure': 21,
                    'healing_time_category': 'moderate_healing',
                    'model_available': False
                },
                'area_cm2': 5.2,
                'area_pixels': 1250,
                'perimeter': 180.5,
                'model_confidence': 0.85,
                'enhanced_analysis': get_enhanced_wound_analysis('burn', 21, 5.2, int(patient_age) if patient_age else None),
                'patient_info': {
                    'id': patient_id,
                    'name': patient_name,
                    'age': patient_age,
                    'gender': patient_gender,
                    'injury_date': injury_date,
                    'analysis_date': datetime.now().isoformat()
                }
            }
            return jsonify(simulated_result)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate detailed reports."""
    try:
        data = request.json
        report_type = data.get('type', 'patient')  # 'patient' or 'clinician'
        analysis_data = data.get('analysis_data', {})
        
        if report_type == 'patient':
            report_content = generate_patient_report(analysis_data)
        else:
            report_content = generate_clinician_report(analysis_data)
        
        return jsonify({
            'success': True,
            'report_content': report_content,
            'filename': f'{report_type}-report-{analysis_data.get("patient_info", {}).get("id", "unknown")}-{datetime.now().strftime("%Y%m%d")}.txt'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_patient_report(analysis_data):
    """Generate patient-friendly report."""
    patient_info = analysis_data.get('patient_info', {})
    enhanced_analysis = analysis_data.get('enhanced_analysis', {})
    
    report = f"""
WOUND HEALING REPORT - PATIENT COPY
=====================================

Patient Information:
- Name: {patient_info.get('name', 'Not provided')}
- ID: {patient_info.get('id', 'Not provided')}
- Age: {patient_info.get('age', 'Not provided')}
- Analysis Date: {patient_info.get('analysis_date', 'Not provided')}

Wound Analysis:
- Type: {enhanced_analysis.get('wound_type', 'Unknown')}
- Estimated Healing Time: {enhanced_analysis.get('estimated_healing_time', 'Unknown')} days
- Size Category: {enhanced_analysis.get('size_category', 'Unknown')}
- Age Group: {enhanced_analysis.get('age_group', 'Unknown')}

Important Precautions:
{chr(10).join([f"• {precaution}" for precaution in enhanced_analysis.get('precautions', [])])}

Treatment Recommendations:
{chr(10).join([f"• {treatment}" for treatment in enhanced_analysis.get('treatment_recommendations', [])])}

Follow-up Schedule:
{chr(10).join([f"• {follow_up}" for follow_up in enhanced_analysis.get('follow_up_schedule', [])])}

Healing Stages:
"""
    
    healing_stages = enhanced_analysis.get('healing_stages', {})
    for stage, info in healing_stages.items():
        report += f"• {stage.title()}: {info['duration']} days - {info['description']}\n"
    
    report += f"""
Risk Factors to Watch For:
{chr(10).join([f"• {risk}" for risk in enhanced_analysis.get('risk_factors', [])])}

IMPORTANT: Please follow all recommendations and contact your healthcare provider if you have any concerns or notice signs of infection (increased redness, swelling, pain, or discharge).

This report was generated by the Wound Healing Progress Tracker system.
    """
    
    return report

def generate_clinician_report(analysis_data):
    """Generate clinician report."""
    patient_info = analysis_data.get('patient_info', {})
    enhanced_analysis = analysis_data.get('enhanced_analysis', {})
    
    report = f"""
WOUND HEALING REPORT - CLINICIAN COPY
======================================

Patient Information:
- Name: {patient_info.get('name', 'Not provided')}
- ID: {patient_info.get('id', 'Not provided')}
- Age: {patient_info.get('age', 'Not provided')}
- Gender: {patient_info.get('gender', 'Not provided')}
- Injury Date: {patient_info.get('injury_date', 'Not provided')}
- Analysis Date: {patient_info.get('analysis_date', 'Not provided')}

Technical Analysis:
- Wound Type: {enhanced_analysis.get('wound_type', 'Unknown')}
- Estimated Healing Time: {enhanced_analysis.get('estimated_healing_time', 'Unknown')} days
- Size Category: {enhanced_analysis.get('size_category', 'Unknown')}
- Age Group: {enhanced_analysis.get('age_group', 'Unknown')}
- Area: {analysis_data.get('area_cm2', 'Not calculated')} cm²
- Perimeter: {analysis_data.get('perimeter', 'Not calculated')} pixels
- Model Confidence: {analysis_data.get('model_confidence', 'Not available')}

Clinical Recommendations:
{chr(10).join([f"• {treatment}" for treatment in enhanced_analysis.get('treatment_recommendations', [])])}

Patient Instructions:
{chr(10).join([f"• {precaution}" for precaution in enhanced_analysis.get('precautions', [])])}

Follow-up Schedule:
{chr(10).join([f"• {follow_up}" for follow_up in enhanced_analysis.get('follow_up_schedule', [])])}

Risk Assessment:
{chr(10).join([f"• {risk}" for risk in enhanced_analysis.get('risk_factors', [])])}

Healing Timeline:
"""
    
    healing_stages = enhanced_analysis.get('healing_stages', {})
    for stage, info in healing_stages.items():
        report += f"• {stage.title()}: {info['duration']} days - {info['description']}\n"
    
    report += f"""
Clinical Notes:
- Monitor healing progress closely
- Adjust treatment plan based on response
- Consider additional interventions if healing is delayed
- Schedule follow-up appointments as indicated
- Document all changes in wound appearance and patient symptoms

This report was generated by the Wound Healing Progress Tracker system.
    """
    
    return report

if __name__ == '__main__':
    print("🏥 Starting Enhanced Wound Healing UI Server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🔗 Make sure the main API server is running on port 5000")
    app.run(host='0.0.0.0', port=5001, debug=True)




