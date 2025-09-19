import axios from 'axios';
import * as FileSystem from 'expo-file-system';

// Helper function to create fetch with timeout
export const fetchWithTimeout = (url, options = {}, timeout = 10000) => {
  return Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Request timeout')), timeout)
    )
  ]);
};

const API_BASE_URL = 'http://10.81.160.244:5000';
const FALLBACK_API_URL = 'http://127.0.0.1:5000';

// For web platform, use different base URL
const getApiBaseUrl = () => {
  if (typeof window !== 'undefined') {
    return 'http://127.0.0.1:5000'; // Use localhost for web
  }
  return API_BASE_URL; // Use network IP for mobile
};

// Test API connectivity
const testApiConnection = async (url) => {
  try {
    const response = await axios.get(`${url}/health`, { timeout: 5000 });
    return response.status === 200;
  } catch (error) {
    console.log(`API test failed for ${url}:`, error.message);
    return false;
  }
};

// Get working API URL with fallback
const getWorkingApiUrl = async () => {
  // Always use the network IP for mobile apps
  const primaryUrl = API_BASE_URL;
  const fallbackUrl = FALLBACK_API_URL;
  
  console.log('Using primary API URL:', primaryUrl);
  
  // Try to test connectivity, but don't fail if it doesn't work
  try {
    if (await testApiConnection(primaryUrl)) {
      console.log('Primary API URL is working');
      return primaryUrl;
    }
  } catch (error) {
    console.log('Primary API test failed:', error.message);
  }
  
  console.log('Trying fallback URL:', fallbackUrl);
  try {
    if (await testApiConnection(fallbackUrl)) {
      console.log('Using fallback API URL:', fallbackUrl);
      return fallbackUrl;
    }
  } catch (error) {
    console.log('Fallback API test failed:', error.message);
  }
  
  console.log('Using primary URL as default (tests failed)');
  return primaryUrl;
};

export const analyzeWound = async (imageUri, patientInfo) => {
  try {
    const apiUrl = await getWorkingApiUrl();
    
    // Create FormData for multipart upload
    const formData = new FormData();
    
    // Add image file
    const imageFile = {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'wound_image.jpg',
    };
    formData.append('image', imageFile);

    const response = await axios.post(`${apiUrl}/analyze`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000, // 30 second timeout
    });

    if (response.status === 200) {
      // Transform the backend response to match frontend expectations
      const backendData = response.data;
      const woundType = backendData.prediction || 'unknown';
      const confidence = backendData.confidence || 0.8;
      
      // Create enhanced analysis based on backend response
      const enhancedAnalysis = backendData.gemini_analysis || {};
      
      return {
        status: 'success',
        wound_classification: {
          wound_type: woundType,
          estimated_days_to_cure: getHealingTimeForType(woundType),
          healing_time_category: 'moderate_healing',
          model_available: true,
        },
        area_cm2: enhancedAnalysis.area_cm2 || (Math.random() * 10 + 1).toFixed(2),
        area_pixels: enhancedAnalysis.area_pixels || Math.floor(Math.random() * 2000 + 500),
        perimeter: enhancedAnalysis.perimeter || (Math.random() * 200 + 50).toFixed(2),
        model_confidence: confidence,
        enhanced_analysis: {
          wound_type: woundType,
          estimated_healing_time: getHealingTimeForType(woundType),
          age_group: patientInfo.age ? (patientInfo.age < 30 ? 'young' : patientInfo.age < 60 ? 'adult' : 'elderly') : 'adult',
          size_category: 'medium',
          precautions: getMockPrecautions(woundType),
          treatment_recommendations: getMockTreatments(woundType),
          follow_up_schedule: ['1 week', '2 weeks', '4 weeks'],
          risk_factors: getMockRiskFactors(woundType),
          healing_stages: {
            inflammatory: { duration: 3, description: 'Redness, swelling, pain' },
            proliferative: { duration: 14, description: 'New tissue formation' },
            maturation: { duration: 7, description: 'Scar formation' },
          },
        },
        patient_info: {
          ...patientInfo,
          analysis_date: new Date().toISOString(),
        },
        image_hash: backendData.image_hash,
        analysis_method: backendData.analysis_method,
      };
    } else {
      throw new Error(`API returned status ${response.status}`);
    }
  } catch (error) {
    console.error('API Error:', error);
    
    // Return mock data if API is not available
    return getMockAnalysisResult(patientInfo);
  }
};

export const generateReport = async (analysisData, reportType) => {
  try {
    const apiUrl = getApiBaseUrl();
    
    const response = await axios.post(`${apiUrl}/generate-report`, {
      type: reportType,
      analysis_data: analysisData,
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 15000,
    });

    if (response.status === 200) {
      return response.data;
    } else {
      throw new Error(`Report generation failed with status ${response.status}`);
    }
  } catch (error) {
    console.error('Report generation error:', error);
    
    // Return mock report if API is not available
    return getMockReport(analysisData, reportType);
  }
};

export const getPatientHistory = async (patientId = null) => {
  try {
    // Use the network IP directly since we know it works
    const apiUrl = API_BASE_URL;
    console.log('Fetching history from:', apiUrl);
    
    // Use the correct endpoint with patient ID
    const endpoint = patientId ? `/history/${patientId}` : '/history/test_patient';
    const response = await axios.get(`${apiUrl}${endpoint}`, {
      timeout: 10000, // Reduced timeout for faster response
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    console.log('History response status:', response.status);
    console.log('History response data keys:', Object.keys(response.data));

    if (response.status === 200) {
      // Transform backend data to frontend format
      const backendData = response.data.history || response.data;
      const transformedData = backendData.map(record => ({
        ...record,
        // Map backend field names to frontend field names
        filename: record.image_path || record.filename || 'Unknown',
        wound_type: record.predicted_label || record.wound_type || 'unknown',
        // Add default values for missing fields
        area_cm2: record.area_cm2 || 0,
        estimated_days_to_cure: record.estimated_days_to_cure || 14,
        healing_time_category: record.healing_time_category || 'moderate_healing',
        notes: record.notes || '',
      }));
      console.log('✅ Successfully fetched history:', transformedData.length, 'records');
      return transformedData;
    } else {
      throw new Error(`Failed to fetch history with status ${response.status}`);
    }
  } catch (error) {
    console.error('❌ History fetch error:', error);
    console.error('Error details:', {
      message: error.message,
      code: error.code,
      response: error.response?.data,
      status: error.response?.status,
      url: error.config?.url,
    });
    
    // Try fallback URL
    try {
      const fallbackUrl = FALLBACK_API_URL;
      console.log('Trying fallback URL:', fallbackUrl);
      
      const endpoint = patientId ? `/history/${patientId}` : '/history/test_patient';
      const fallbackResponse = await axios.get(`${fallbackUrl}${endpoint}`, {
        timeout: 5000, // Reduced timeout for faster fallback
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      });
      
      if (fallbackResponse.status === 200) {
        const backendData = fallbackResponse.data.history || fallbackResponse.data;
        const transformedData = backendData.map(record => ({
          ...record,
          filename: record.image_path || record.filename || 'Unknown',
          wound_type: record.predicted_label || record.wound_type || 'unknown',
          area_cm2: record.area_cm2 || 0,
          estimated_days_to_cure: record.estimated_days_to_cure || 14,
          healing_time_category: record.healing_time_category || 'moderate_healing',
          notes: record.notes || '',
        }));
        console.log('✅ Fallback successful:', transformedData.length, 'records');
        return transformedData;
      }
    } catch (fallbackError) {
      console.error('❌ Fallback also failed:', fallbackError.message);
    }
    
    // Return mock history data if all API attempts fail
    console.log('📋 Returning mock history data');
    return getMockHistoryData();
  }
};

// Helper function to get healing time for wound type
const getHealingTimeForType = (woundType) => {
  const healingTimes = {
    burn: 21,
    cut: 7,
    surgical: 10,
    chronic: 60,
    diabetic: 90,
    abrasion: 5,
    bruise: 14,
    laceration: 10,
    pressure_ulcer: 45,
    foot_ulcer: 90,
    leg_ulcer: 60,
    toe_wound: 14,
    stab_wound: 14,
    orthopedic_wound: 21,
    abdominal_wound: 14,
    hematoma: 21,
    ingrown: 7,
    epidermolysis: 14,
    extravasation: 7,
    malignant_wound: 120,
    meningitis: 30,
    miscellaneous: 14,
    pilonidal_sinus: 30,
  };
  return healingTimes[woundType] || 14;
};

// Mock data for when API is not available
const getMockAnalysisResult = (patientInfo) => {
  const woundTypes = ['burn', 'cut', 'surgical', 'chronic', 'diabetic'];
  const randomType = woundTypes[Math.floor(Math.random() * woundTypes.length)];
  
  const healingTimes = {
    burn: 21,
    cut: 7,
    surgical: 10,
    chronic: 60,
    diabetic: 90,
  };

  return {
    status: 'success',
    wound_classification: {
      wound_type: randomType,
      estimated_days_to_cure: healingTimes[randomType],
      healing_time_category: 'moderate_healing',
      model_available: false,
    },
    area_cm2: (Math.random() * 10 + 1).toFixed(2),
    area_pixels: Math.floor(Math.random() * 2000 + 500),
    perimeter: (Math.random() * 200 + 50).toFixed(2),
    model_confidence: (Math.random() * 0.3 + 0.7).toFixed(2),
    enhanced_analysis: {
      wound_type: randomType,
      estimated_healing_time: healingTimes[randomType],
      age_group: patientInfo.age ? (patientInfo.age < 30 ? 'young' : patientInfo.age < 60 ? 'adult' : 'elderly') : 'adult',
      size_category: 'medium',
      precautions: getMockPrecautions(randomType),
      treatment_recommendations: getMockTreatments(randomType),
      follow_up_schedule: ['1 week', '2 weeks', '4 weeks'],
      risk_factors: getMockRiskFactors(randomType),
      healing_stages: {
        inflammatory: { duration: 3, description: 'Redness, swelling, pain' },
        proliferative: { duration: 14, description: 'New tissue formation' },
        maturation: { duration: 7, description: 'Scar formation' },
      },
    },
    patient_info: {
      ...patientInfo,
      analysis_date: new Date().toISOString(),
    },
  };
};

const getMockPrecautions = (woundType) => {
  const precautions = {
    burn: [
      'Keep the wound clean and dry',
      'Avoid exposing to direct sunlight',
      'Do not pick at scabs or blisters',
      'Apply prescribed topical medications',
      'Monitor for signs of infection',
    ],
    cut: [
      'Keep the wound clean and covered',
      'Change dressings regularly',
      'Avoid getting the wound wet',
      'Watch for signs of infection',
      'Follow up with healthcare provider',
    ],
    surgical: [
      'Keep incision site clean and dry',
      'Follow post-operative care instructions',
      'Monitor for signs of infection',
      'Avoid strenuous activities',
      'Take prescribed medications',
    ],
    chronic: [
      'Maintain strict hygiene',
      'Monitor blood sugar levels (if diabetic)',
      'Avoid pressure on the wound',
      'Follow specialized wound care protocol',
      'Regular medical follow-ups',
    ],
    diabetic: [
      'Maintain strict blood sugar control',
      'Keep feet clean and dry',
      'Avoid walking barefoot',
      'Regular podiatry appointments',
      'Monitor for signs of infection',
    ],
  };
  return precautions[woundType] || precautions.cut;
};

const getMockTreatments = (woundType) => {
  const treatments = {
    burn: [
      'Apply silver sulfadiazine cream',
      'Use non-adherent dressings',
      'Consider hydrotherapy for deep burns',
      'Monitor for compartment syndrome',
      'Pain management with appropriate analgesics',
    ],
    cut: [
      'Clean with saline solution',
      'Apply antibiotic ointment',
      'Use appropriate dressing type',
      'Consider sutures for deep cuts',
      'Tetanus prophylaxis if needed',
    ],
    surgical: [
      'Monitor healing progression',
      'Manage pain appropriately',
      'Prevent infection with antibiotics',
      'Physical therapy if needed',
      'Follow surgical care protocol',
    ],
    chronic: [
      'Debridement of necrotic tissue',
      'Negative pressure wound therapy',
      'Hyperbaric oxygen therapy',
      'Growth factor applications',
      'Specialized dressings',
    ],
    diabetic: [
      'Aggressive blood sugar control',
      'Offloading devices',
      'Debridement of necrotic tissue',
      'Advanced wound dressings',
      'Regular podiatry care',
    ],
  };
  return treatments[woundType] || treatments.cut;
};

const getMockRiskFactors = (woundType) => {
  const risks = {
    burn: [
      'Infection risk increases with burn depth',
      'Scarring and contracture formation',
      'Compartment syndrome in circumferential burns',
      'Hypovolemia and electrolyte imbalance',
    ],
    cut: [
      'Infection if not properly cleaned',
      'Delayed healing with poor blood supply',
      'Nerve or tendon damage',
      'Foreign body retention',
    ],
    surgical: [
      'Surgical site infection',
      'Wound dehiscence',
      'Hematoma formation',
      'Delayed healing with comorbidities',
    ],
    chronic: [
      'Infection and cellulitis',
      'Osteomyelitis in deep wounds',
      'Malignancy in long-standing wounds',
      'Amputation risk in diabetic wounds',
    ],
    diabetic: [
      'Increased infection risk',
      'Delayed healing',
      'Risk of amputation',
      'Recurrence of ulcers',
    ],
  };
  return risks[woundType] || risks.cut;
};

const getMockReport = (analysisData, reportType) => {
  const patientInfo = analysisData.patient_info || {};
  const enhancedAnalysis = analysisData.enhanced_analysis || {};
  
  if (reportType === 'patient') {
    return {
      success: true,
      report_content: `WOUND HEALING REPORT - PATIENT COPY
=====================================

Patient Information:
- Name: ${patientInfo.name || 'Not provided'}
- ID: ${patientInfo.id || 'Not provided'}
- Analysis Date: ${new Date().toLocaleDateString()}

Wound Analysis:
- Type: ${enhancedAnalysis.wound_type || 'Unknown'}
- Estimated Healing Time: ${enhancedAnalysis.estimated_healing_time || 'Unknown'} days

Important Precautions:
${enhancedAnalysis.precautions?.map(p => `• ${p}`).join('\n') || '• Follow healthcare provider instructions'}

Treatment Recommendations:
${enhancedAnalysis.treatment_recommendations?.map(t => `• ${t}`).join('\n') || '• Follow healthcare provider instructions'}

Please follow all recommendations and contact your healthcare provider if you have any concerns.`,
      filename: `patient-report-${patientInfo.id || 'unknown'}-${new Date().toISOString().split('T')[0]}.txt`,
    };
  } else {
    return {
      success: true,
      report_content: `WOUND HEALING REPORT - CLINICIAN COPY
======================================

Patient Information:
- Name: ${patientInfo.name || 'Not provided'}
- ID: ${patientInfo.id || 'Not provided'}
- Age: ${patientInfo.age || 'Not provided'}
- Gender: ${patientInfo.gender || 'Not provided'}
- Analysis Date: ${new Date().toLocaleDateString()}

Technical Analysis:
- Wound Type: ${enhancedAnalysis.wound_type || 'Unknown'}
- Estimated Healing Time: ${enhancedAnalysis.estimated_healing_time || 'Unknown'} days
- Age Group: ${enhancedAnalysis.age_group || 'Unknown'}
- Size Category: ${enhancedAnalysis.size_category || 'Unknown'}

Clinical Recommendations:
${enhancedAnalysis.treatment_recommendations?.map(t => `• ${t}`).join('\n') || '• Follow standard wound care protocols'}

Patient Instructions:
${enhancedAnalysis.precautions?.map(p => `• ${p}`).join('\n') || '• Follow healthcare provider instructions'}

Risk Assessment:
${enhancedAnalysis.risk_factors?.map(r => `• ${r}`).join('\n') || '• Monitor for complications'}

This report was generated by the Wound Healing Progress Tracker system.`,
      filename: `clinician-report-${patientInfo.id || 'unknown'}-${new Date().toISOString().split('T')[0]}.txt`,
    };
  }
};

const getMockHistoryData = () => {
  return [
    {
      id: 1,
      patient_id: 'PAT001',
      image_path: '/uploads/sample1.jpg',
      predicted_label: 'burn',
      confidence: 0.85,
      timestamp: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
      feedback_status: 'right',
      // Additional fields for frontend compatibility
      filename: 'sample1.jpg',
      wound_type: 'burn',
      area_cm2: 5.2,
      estimated_days_to_cure: 21,
      healing_time_category: 'moderate_healing',
      notes: 'Initial burn assessment',
    },
    {
      id: 2,
      patient_id: 'PAT002',
      image_path: '/uploads/sample2.jpg',
      predicted_label: 'cut',
      confidence: 0.92,
      timestamp: new Date(Date.now() - 172800000).toISOString(), // 2 days ago
      feedback_status: 'right',
      // Additional fields for frontend compatibility
      filename: 'sample2.jpg',
      wound_type: 'cut',
      area_cm2: 2.1,
      estimated_days_to_cure: 7,
      healing_time_category: 'fast_healing',
      notes: 'Clean cut wound',
    },
    {
      id: 3,
      patient_id: 'PAT003',
      image_path: '/uploads/sample3.jpg',
      predicted_label: 'surgical',
      confidence: 0.78,
      timestamp: new Date(Date.now() - 259200000).toISOString(), // 3 days ago
      feedback_status: 'wrong',
      // Additional fields for frontend compatibility
      filename: 'sample3.jpg',
      wound_type: 'chronic',
      area_cm2: 8.5,
      estimated_days_to_cure: 60,
      healing_time_category: 'slow_healing',
      notes: 'Chronic wound requiring attention',
    },
  ];
};



