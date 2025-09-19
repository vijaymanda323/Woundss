import { Platform } from 'react-native';

// Generate unique patient ID
export const generatePatientId = () => {
  const timestamp = Date.now().toString(36);
  const randomStr = Math.random().toString(36).substring(2, 8);
  return `P${timestamp}${randomStr}`.toUpperCase();
};

// Generate unique analysis ID
export const generateAnalysisId = () => {
  const timestamp = Date.now().toString(36);
  const randomStr = Math.random().toString(36).substring(2, 6);
  return `A${timestamp}${randomStr}`.toUpperCase();
};

// Generate unique report ID
export const generateReportId = () => {
  const timestamp = Date.now().toString(36);
  const randomStr = Math.random().toString(36).substring(2, 6);
  return `R${timestamp}${randomStr}`.toUpperCase();
};

// Format date for display
export const formatDate = (date) => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

// Calculate healing progress percentage
export const calculateHealingProgress = (currentArea, previousArea) => {
  if (!previousArea || previousArea === 0) return 0;
  const progress = ((previousArea - currentArea) / previousArea) * 100;
  return Math.max(0, Math.min(100, progress));
};

// Get wound severity level
export const getWoundSeverity = (area, woundType) => {
  const severityThresholds = {
    burn: { mild: 5, moderate: 15, severe: 30 },
    cut: { mild: 2, moderate: 8, severe: 15 },
    surgical: { mild: 3, moderate: 10, severe: 20 },
    chronic: { mild: 10, moderate: 25, severe: 50 },
    diabetic: { mild: 5, moderate: 15, severe: 30 }
  };

  const thresholds = severityThresholds[woundType] || severityThresholds.cut;
  
  if (area <= thresholds.mild) return 'Mild';
  if (area <= thresholds.moderate) return 'Moderate';
  return 'Severe';
};

// Generate treatment recommendations based on wound characteristics
export const generateTreatmentPlan = (analysisData) => {
  const { woundType, area, severity, patientAge } = analysisData;
  
  const baseRecommendations = {
    burn: [
      'Apply silver sulfadiazine cream twice daily',
      'Use non-adherent dressings',
      'Monitor for signs of infection',
      'Consider hydrotherapy for deep burns',
      'Pain management with appropriate analgesics'
    ],
    cut: [
      'Clean with saline solution',
      'Apply antibiotic ointment',
      'Use appropriate dressing type',
      'Consider sutures for deep cuts',
      'Tetanus prophylaxis if needed'
    ],
    surgical: [
      'Monitor healing progression',
      'Manage pain appropriately',
      'Prevent infection with antibiotics',
      'Physical therapy if needed',
      'Follow surgical care protocol'
    ],
    chronic: [
      'Debridement of necrotic tissue',
      'Negative pressure wound therapy',
      'Hyperbaric oxygen therapy',
      'Growth factor applications',
      'Specialized dressings'
    ],
    diabetic: [
      'Aggressive blood sugar control',
      'Offloading devices',
      'Debridement of necrotic tissue',
      'Advanced wound dressings',
      'Regular podiatry care'
    ]
  };

  const recommendations = baseRecommendations[woundType] || baseRecommendations.cut;
  
  // Add age-specific recommendations
  if (patientAge > 65) {
    recommendations.push('Monitor for delayed healing due to age');
    recommendations.push('Consider nutritional supplements');
  }
  
  // Add severity-specific recommendations
  if (severity === 'Severe') {
    recommendations.push('Consider specialist consultation');
    recommendations.push('Monitor for complications closely');
  }

  return recommendations;
};

// Generate follow-up schedule
export const generateFollowUpSchedule = (woundType, severity) => {
  const baseSchedule = {
    burn: { mild: [7, 14], moderate: [3, 7, 14, 21], severe: [1, 3, 7, 14, 21, 30] },
    cut: { mild: [7, 14], moderate: [3, 7, 14], severe: [1, 3, 7, 14, 21] },
    surgical: { mild: [7, 14], moderate: [3, 7, 14], severe: [1, 3, 7, 14, 21] },
    chronic: { mild: [14, 30], moderate: [7, 14, 30], severe: [3, 7, 14, 30, 60] },
    diabetic: { mild: [7, 14], moderate: [3, 7, 14, 30], severe: [1, 3, 7, 14, 30, 60] }
  };

  const schedule = baseSchedule[woundType] || baseSchedule.cut;
  return schedule[severity] || schedule.mild;
};

// Store patient data locally (for demo purposes)
export const storePatientData = (patientData) => {
  try {
    const existingData = getStoredPatients();
    
    // Check if patient already exists and update or add
    const existingIndex = existingData.findIndex(p => p.id === patientData.id);
    let updatedData;
    
    if (existingIndex >= 0) {
      // Update existing patient
      updatedData = [...existingData];
      updatedData[existingIndex] = { ...updatedData[existingIndex], ...patientData };
    } else {
      // Add new patient
      updatedData = [...existingData, patientData];
    }
    
    if (Platform.OS === 'web') {
      localStorage.setItem('woundPatients', JSON.stringify(updatedData));
    } else {
      // For mobile, you would use AsyncStorage or similar
      console.log('Storing patient data:', patientData);
    }
    
    console.log('Patient data stored successfully:', patientData.id);
  } catch (error) {
    console.error('Error storing patient data:', error);
  }
};

// Retrieve stored patient data
export const getStoredPatients = () => {
  try {
    if (Platform.OS === 'web') {
      const data = localStorage.getItem('woundPatients');
      return data ? JSON.parse(data) : [];
    } else {
      // For mobile, you would use AsyncStorage or similar
      return [];
    }
  } catch (error) {
    console.error('Error retrieving patient data:', error);
    return [];
  }
};

// Get patient history
export const getPatientHistory = (patientId) => {
  const allPatients = getStoredPatients();
  return allPatients.filter(patient => patient.id === patientId);
};

// Generate PDF content for reports
export const generatePDFContent = (reportData, reportType) => {
  const { patient, analysis, treatmentPlan } = reportData;
  
  if (reportType === 'patient') {
    return `
WOUND HEALING REPORT - PATIENT COPY
=====================================

Patient Information:
- Name: ${patient.name || 'Not provided'}
- Patient ID: ${patient.id}
- Date of Birth: ${patient.dateOfBirth || 'Not provided'}
- Gender: ${patient.gender || 'Not provided'}
- Analysis Date: ${formatDate(analysis.timestamp)}

Wound Analysis:
- Type: ${analysis.woundType}
- Severity: ${analysis.severity}
- Area: ${analysis.area} cm²
- Estimated Healing Time: ${analysis.healingTime} days
- Confidence Level: ${(analysis.confidence * 100).toFixed(1)}%

Treatment Plan:
${treatmentPlan.recommendations.map(rec => `• ${rec}`).join('\n')}

Follow-up Schedule:
${treatmentPlan.followUpSchedule.map(day => `• Day ${day}`).join('\n')}

Important Precautions:
${treatmentPlan.precautions.map(prec => `• ${prec}`).join('\n')}

Please follow all recommendations and contact your healthcare provider if you have any concerns.

Report Generated: ${formatDate(new Date())}
Report ID: ${generateReportId()}
    `;
  } else {
    return `
WOUND HEALING REPORT - CLINICIAN COPY
======================================

Patient Information:
- Name: ${patient.name || 'Not provided'}
- Patient ID: ${patient.id}
- Date of Birth: ${patient.dateOfBirth || 'Not provided'}
- Gender: ${patient.gender || 'Not provided'}
- Contact: ${patient.contact || 'Not provided'}
- Analysis Date: ${formatDate(analysis.timestamp)}

Technical Analysis:
- Wound Type: ${analysis.woundType}
- Severity: ${analysis.severity}
- Area: ${analysis.area} cm²
- Perimeter: ${analysis.perimeter || 'N/A'} cm
- Estimated Healing Time: ${analysis.healingTime} days
- Confidence Level: ${(analysis.confidence * 100).toFixed(1)}%
- Model Version: ${analysis.modelVersion || '1.0'}

Clinical Recommendations:
${treatmentPlan.recommendations.map(rec => `• ${rec}`).join('\n')}

Patient Instructions:
${treatmentPlan.precautions.map(prec => `• ${prec}`).join('\n')}

Follow-up Schedule:
${treatmentPlan.followUpSchedule.map(day => `• Day ${day}`).join('\n')}

Risk Assessment:
${treatmentPlan.riskFactors.map(risk => `• ${risk}`).join('\n')}

Previous History:
${patient.history ? patient.history.map(h => `• ${formatDate(h.date)}: ${h.description}`).join('\n') : 'No previous history'}

This report was generated by the Wound Healing Progress Tracker system.

Report Generated: ${formatDate(new Date())}
Report ID: ${generateReportId()}
Clinician: ${patient.clinician || 'System Generated'}
    `;
  }
};

// Generate HTML content for PDF conversion
export const generateHTMLContent = (reportData, reportType) => {
  const { patient, analysis, treatmentPlan } = reportData;
  const reportId = generateReportId();
  
  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>Wound Healing Report</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 20px;
          line-height: 1.6;
          color: #333;
        }
        .header {
          text-align: center;
          border-bottom: 2px solid #667eea;
          padding-bottom: 20px;
          margin-bottom: 30px;
        }
        .title {
          font-size: 24px;
          font-weight: bold;
          color: #667eea;
          margin-bottom: 10px;
        }
        .subtitle {
          font-size: 18px;
          color: #666;
        }
        .section {
          margin-bottom: 25px;
        }
        .section-title {
          font-size: 18px;
          font-weight: bold;
          color: #667eea;
          margin-bottom: 15px;
          border-bottom: 1px solid #ddd;
          padding-bottom: 5px;
        }
        .info-item {
          margin-bottom: 8px;
        }
        .info-label {
          font-weight: bold;
          display: inline-block;
          width: 150px;
        }
        .recommendations, .precautions, .risks {
          margin-left: 20px;
        }
        .recommendations li, .precautions li, .risks li {
          margin-bottom: 5px;
        }
        .footer {
          margin-top: 40px;
          padding-top: 20px;
          border-top: 1px solid #ddd;
          font-size: 12px;
          color: #666;
        }
        .report-id {
          font-weight: bold;
          color: #667eea;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <div class="title">WOUND HEALING REPORT</div>
        <div class="subtitle">${reportType === 'patient' ? 'PATIENT COPY' : 'CLINICIAN COPY'}</div>
      </div>
      
      <div class="section">
        <div class="section-title">Patient Information</div>
        <div class="info-item">
          <span class="info-label">Name:</span>
          ${patient.name || 'Not provided'}
        </div>
        <div class="info-item">
          <span class="info-label">Patient ID:</span>
          ${patient.id}
        </div>
        <div class="info-item">
          <span class="info-label">Date of Birth:</span>
          ${patient.dateOfBirth || 'Not provided'}
        </div>
        <div class="info-item">
          <span class="info-label">Gender:</span>
          ${patient.gender || 'Not provided'}
        </div>
        <div class="info-item">
          <span class="info-label">Analysis Date:</span>
          ${formatDate(analysis.timestamp)}
        </div>
        ${reportType === 'clinician' ? `
        <div class="info-item">
          <span class="info-label">Contact:</span>
          ${patient.contact || 'Not provided'}
        </div>
        ` : ''}
      </div>
      
      <div class="section">
        <div class="section-title">Wound Analysis</div>
        <div class="info-item">
          <span class="info-label">Type:</span>
          ${analysis.woundType}
        </div>
        <div class="info-item">
          <span class="info-label">Severity:</span>
          ${analysis.severity}
        </div>
        <div class="info-item">
          <span class="info-label">Area:</span>
          ${analysis.area} cm²
        </div>
        <div class="info-item">
          <span class="info-label">Healing Time:</span>
          ${analysis.healingTime} days
        </div>
        <div class="info-item">
          <span class="info-label">Confidence:</span>
          ${(analysis.confidence * 100).toFixed(1)}%
        </div>
        ${reportType === 'clinician' ? `
        <div class="info-item">
          <span class="info-label">Perimeter:</span>
          ${analysis.perimeter || 'N/A'} cm
        </div>
        <div class="info-item">
          <span class="info-label">Model Version:</span>
          ${analysis.modelVersion || '1.0'}
        </div>
        ` : ''}
      </div>
      
      <div class="section">
        <div class="section-title">Treatment Plan</div>
        <ul class="recommendations">
          ${treatmentPlan.recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Follow-up Schedule</div>
        <ul class="recommendations">
          ${treatmentPlan.followUpSchedule.map(day => `<li>Day ${day}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Important Precautions</div>
        <ul class="precautions">
          ${treatmentPlan.precautions.map(prec => `<li>${prec}</li>`).join('')}
        </ul>
      </div>
      
      ${reportType === 'clinician' ? `
      <div class="section">
        <div class="section-title">Risk Assessment</div>
        <ul class="risks">
          ${treatmentPlan.riskFactors.map(risk => `<li>${risk}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Previous History</div>
        ${patient.history && patient.history.length > 0 ? 
          `<ul class="recommendations">
            ${patient.history.map(h => `<li>${formatDate(h.date)}: ${h.description}</li>`).join('')}
          </ul>` : 
          '<p>No previous history</p>'
        }
      </div>
      ` : ''}
      
      <div class="footer">
        <p>This report was generated by the Wound Healing Progress Tracker system.</p>
        <p>Report Generated: ${formatDate(new Date())}</p>
        <p>Report ID: <span class="report-id">${reportId}</span></p>
        ${reportType === 'clinician' ? `<p>Clinician: ${patient.clinician || 'System Generated'}</p>` : ''}
      </div>
    </body>
    </html>
  `;
  
  return htmlContent;
};

// Generate PDF file using Expo compatible method
export const generatePDFFile = async (reportData, reportType) => {
  try {
    if (Platform.OS === 'web') {
      // For web, create a downloadable HTML file
      const htmlContent = generateHTMLContent(reportData, reportType);
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${reportType}-report-${generateReportId()}.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      return `${reportType}-report-${generateReportId()}.html`;
    } else {
      // For mobile, create a text file that can be shared
      const textContent = generatePDFContent(reportData, reportType);
      const reportId = generateReportId();
      const fileName = `${reportType}-report-${reportId}.txt`;
      
      // Return the text content for sharing
      return {
        content: textContent,
        fileName: fileName,
        mimeType: 'text/plain'
      };
    }
  } catch (error) {
    console.error('PDF generation error:', error);
    throw error;
  }
};
