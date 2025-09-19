import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  Platform,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  TextInput,
  List,
  Divider,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { generatePDFContent, generatePDFFile, generateReportId, formatDate, storePatientData, generatePatientId } from '../utils/patientUtils';
import { fetchWithTimeout } from '../services/apiService';

export default function ReportsScreen({ navigation, route }) {
  const { analysisData, treatmentPlan, imageUri, patientHistory } = route.params || {};
  
  // Provide default values to prevent undefined errors
  const safeAnalysisData = analysisData || {};
  const safeTreatmentPlan = treatmentPlan || {};
  const safePatientHistory = patientHistory || [];
  
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    dateOfBirth: '',
    gender: '',
    contact: '',
    address: '',
    clinician: '',
    notes: '',
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportType, setReportType] = useState('patient');
  const [patientIdInput, setPatientIdInput] = useState('');
  const [isLoadingPatientData, setIsLoadingPatientData] = useState(false);
  const [patientReportData, setPatientReportData] = useState(null);

  // If no data is provided, redirect to PhotoUpload
  React.useEffect(() => {
    if (!analysisData && !treatmentPlan) {
      navigation.replace('PhotoUpload');
    }
  }, [analysisData, treatmentPlan, navigation]);

  const handleInputChange = (field, value) => {
    setPatientInfo(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const fetchPatientData = async (patientId) => {
    if (!patientId.trim()) {
      Alert.alert('Error', 'Please enter a patient ID.');
      return;
    }

    setIsLoadingPatientData(true);
    try {
        const response = await fetchWithTimeout(`http://10.81.160.244:5000/generate-report/${patientId}`, {
          method: 'GET',
        }, 10000);
      const data = await response.json();
      
      if (data.status === 'success') {
        setPatientReportData(data.report_data);
        Alert.alert('Success', `Found ${data.report_data.total_records} records for patient ${patientId}`);
      } else {
        Alert.alert('Error', data.error || 'Failed to fetch patient data');
        setPatientReportData(null);
      }
    } catch (error) {
      console.error('Error fetching patient data:', error);
      Alert.alert('Error', 'Failed to connect to server. Please check your connection.');
      setPatientReportData(null);
    } finally {
      setIsLoadingPatientData(false);
    }
  };

  const generateComprehensiveReport = async (reportData) => {
    setIsGenerating(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      // Create comprehensive report data
      const comprehensiveReportData = {
        patient: {
          id: reportData.patient_id,
          name: patientInfo.name || `Patient ${reportData.patient_id}`,
          dateOfBirth: patientInfo.dateOfBirth || 'Not provided',
          gender: patientInfo.gender || 'Not provided',
          contact: patientInfo.contact || 'Not provided',
          address: patientInfo.address || 'Not provided',
          clinician: patientInfo.clinician || 'Not provided',
          notes: patientInfo.notes || 'Not provided',
          history: reportData.all_records,
        },
        analysis: {
          woundType: reportData.wound_classification?.wound_type || 'unknown',
          severity: reportData.wound_classification?.healing_time_category || 'moderate',
          area: reportData.latest_analysis?.area_cm2 || 0,
          healingTime: reportData.wound_classification?.estimated_days_to_cure || 30,
          confidence: reportData.latest_analysis?.model_confidence || 0.85,
          timestamp: reportData.latest_analysis?.timestamp || new Date().toISOString(),
          modelVersion: reportData.latest_analysis?.model_version || '1.0',
        },
        treatmentPlan: {
          recommendations: generateTreatmentRecommendations(reportData),
          followUpSchedule: generateFollowUpSchedule(reportData),
          precautions: generatePrecautions(reportData),
          riskFactors: generateRiskFactors(reportData),
        },
        comprehensiveData: {
          totalRecords: reportData.total_records,
          timeSpanDays: reportData.time_span_days,
          overallHealingPercentage: reportData.overall_healing_percentage,
          healingProgress: reportData.healing_progress,
          statistics: reportData.statistics,
          allRecords: reportData.all_records,
        }
      };

      // Generate PDF content
      const pdfContent = generateComprehensivePDFContent(comprehensiveReportData);
      const reportId = generateReportId();

      // Show success message
      Alert.alert(
        'Comprehensive Report Generated Successfully!',
        `Comprehensive report has been generated with ${reportData.total_records} records.\n\nReport ID: ${reportId}`,
        [
          {
            text: 'View Report',
            onPress: () => {
              Alert.alert(
                'Comprehensive Report Content',
                pdfContent,
                [
                  { text: 'OK' },
                  {
                    text: 'Download PDF',
                    onPress: () => downloadComprehensivePDF(comprehensiveReportData, reportId),
                  },
                ]
              );
            },
          },
          { text: 'OK' },
        ]
      );

    } catch (error) {
      console.error('Comprehensive report generation error:', error);
      Alert.alert('Error', 'Failed to generate comprehensive report. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const generateReport = async (type) => {
    if (!patientInfo.name.trim()) {
      Alert.alert('Required Field', 'Please enter the patient name.');
      return;
    }

    setIsGenerating(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      // Create report data
      const reportData = {
        patient: {
          ...patientInfo,
          id: safeAnalysisData.patientId || generatePatientId(),
          history: safePatientHistory,
        },
        analysis: {
          woundType: safeAnalysisData.woundType || 'unknown',
          severity: safeAnalysisData.severity || 'moderate',
          area: safeAnalysisData.area || 5.0,
          healingTime: safeAnalysisData.healingTime || 21,
          confidence: safeAnalysisData.confidence || 0.85,
          timestamp: safeAnalysisData.timestamp || new Date().toISOString(),
          modelVersion: '1.0',
        },
        treatmentPlan: {
          recommendations: safeTreatmentPlan?.recommendations || ['Follow healthcare provider instructions'],
          followUpSchedule: safeTreatmentPlan?.followUpSchedule || ['Day 7', 'Day 14'],
          precautions: safeTreatmentPlan?.precautions || ['Keep wound clean and dry'],
          riskFactors: safeTreatmentPlan?.riskFactors || ['Monitor for infection'],
        },
      };

      // Generate PDF content
      const pdfContent = generatePDFContent(reportData, type);
      const reportId = generateReportId();

      // Store patient data
      const patientData = {
        id: analysisData.patientId,
        ...patientInfo,
        analysisData,
        treatmentPlan,
        reportId,
        timestamp: new Date().toISOString(),
      };
      storePatientData(patientData);

      // Save patient details to backend
      try {
        await fetchWithTimeout(`http://10.81.160.244:5000/patient/${analysisData.patientId}/save`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(patientInfo),
        }, 8000);
      } catch (error) {
        console.log('Failed to save patient details to backend:', error);
      }

      // Save analysis data to backend if available
      if (safeAnalysisData.patientId && safeAnalysisData.areaPixels) {
        try {
          const analysisPayload = {
            timestamp: safeAnalysisData.timestamp || new Date().toISOString(),
            area_pixels: safeAnalysisData.areaPixels,
            area_cm2: safeAnalysisData.area || 0,
            model_confidence: safeAnalysisData.confidence || 0.85,
            model_version: 'report_generated',
            notes: `Report generated - ${safeAnalysisData.woundType || 'unknown'} wound`,
            healing_pct: 0, // Will be calculated by backend
            days_to_heal: safeAnalysisData.healingTime || 21
          };

          const response = await fetchWithTimeout(`http://10.81.160.244:5000/patient/${safeAnalysisData.patientId}/analysis`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(analysisPayload),
          }, 8000);

          if (response.ok) {
            const result = await response.json();
            console.log('Analysis data saved to backend successfully:', result);
          } else {
            console.log('Failed to save analysis data to backend:', response.status);
          }
        } catch (error) {
          console.log('Failed to save analysis data to backend:', error);
        }
      }

      // Simulate PDF generation
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Show success message
      Alert.alert(
        'Report Generated Successfully!',
        `${type.charAt(0).toUpperCase() + type.slice(1)} report has been generated.\n\nReport ID: ${reportId}`,
        [
          {
            text: 'View Report',
            onPress: () => {
              Alert.alert(
                'Report Content',
                pdfContent,
                [
                  { text: 'OK' },
                  {
                    text: 'Download PDF',
                    onPress: () => downloadPDF(reportData, type),
                  },
                ]
              );
            },
          },
          { text: 'OK' },
        ]
      );

    } catch (error) {
      console.error('Report generation error:', error);
      Alert.alert('Error', 'Failed to generate report. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadPDF = async (reportData, type) => {
    try {
      if (Platform.OS === 'web') {
        // For web, generate and download HTML file
        const fileName = await generatePDFFile(reportData, type);
        Alert.alert('Success', `Report "${fileName}" downloaded successfully!`);
      } else {
        // For mobile, try to create and share a file
        try {
          const textContent = generatePDFContent(reportData, type);
          const reportId = generateReportId();
          const fileName = `${type}-report-${reportId}.txt`;
          const fileUri = `${FileSystem.documentDirectory}${fileName}`;
          
          // Write content to file without encoding parameter
          await FileSystem.writeAsStringAsync(fileUri, textContent);
          
          // Check if sharing is available
          const isAvailable = await Sharing.isAvailableAsync();
          if (isAvailable) {
            await Sharing.shareAsync(fileUri, {
              mimeType: 'text/plain',
              dialogTitle: `${type.charAt(0).toUpperCase() + type.slice(1)} Report`,
            });
          } else {
            throw new Error('Sharing not available');
          }
        } catch (fileError) {
          // Fallback: show report content in alert
          console.log('File sharing failed, showing in alert:', fileError);
          const textContent = generatePDFContent(reportData, type);
          
          Alert.alert(
            `${type.charAt(0).toUpperCase() + type.slice(1)} Report`,
            textContent.substring(0, 500) + (textContent.length > 500 ? '...' : ''),
            [
              { text: 'OK' },
              {
                text: 'View Full Report',
                onPress: () => {
                  Alert.alert(
                    'Full Report',
                    textContent,
                    [{ text: 'OK' }],
                    { cancelable: true }
                  );
                }
              }
            ]
          );
        }
      }
    } catch (error) {
      console.error('Download error:', error);
      Alert.alert('Error', 'Failed to generate report.');
    }
  };

  const getWoundTypeColor = (type) => {
    const colors = {
      burn: '#e74c3c',
      cut: '#3498db',
      surgical: '#9b59b6',
      chronic: '#f39c12',
      diabetic: '#e67e22',
    };
    return colors[type] || '#95a5a6';
  };

  const generateTreatmentRecommendations = (reportData) => {
    const woundType = reportData.wound_classification?.wound_type || 'unknown';
    const healingPct = reportData.overall_healing_percentage;
    
    const baseRecommendations = {
      burn: ['Apply silver sulfadiazine cream twice daily', 'Use non-adherent dressings', 'Monitor for signs of infection'],
      cut: ['Clean with saline solution', 'Apply antibiotic ointment', 'Use appropriate dressing type'],
      surgical: ['Monitor healing progression', 'Manage pain appropriately', 'Prevent infection with antibiotics'],
      chronic: ['Debridement of necrotic tissue', 'Negative pressure wound therapy', 'Hyperbaric oxygen therapy'],
      diabetic: ['Aggressive blood sugar control', 'Offloading devices', 'Debridement of necrotic tissue']
    };
    
    let recommendations = baseRecommendations[woundType] || baseRecommendations.cut;
    
    if (healingPct < 20) {
      recommendations.push('Consider specialist consultation');
      recommendations.push('Monitor for complications closely');
    } else if (healingPct > 50) {
      recommendations.push('Continue current treatment protocol');
      recommendations.push('Consider reducing follow-up frequency');
    }
    
    return recommendations;
  };

  const generateFollowUpSchedule = (reportData) => {
    const healingPct = reportData.overall_healing_percentage;
    const timeSpan = reportData.time_span_days;
    
    if (healingPct < 20) {
      return ['Day 3', 'Day 7', 'Day 14', 'Day 21', 'Day 30'];
    } else if (healingPct < 50) {
      return ['Day 7', 'Day 14', 'Day 30'];
    } else {
      return ['Day 14', 'Day 30'];
    }
  };

  const generatePrecautions = (reportData) => {
    const healingPct = reportData.overall_healing_percentage;
    const woundType = reportData.wound_classification?.wound_type || 'unknown';
    
    let precautions = ['Keep wound clean and dry', 'Monitor for signs of infection'];
    
    if (woundType === 'diabetic') {
      precautions.push('Maintain strict blood sugar control');
      precautions.push('Avoid pressure on wound area');
    } else if (woundType === 'burn') {
      precautions.push('Avoid sun exposure');
      precautions.push('Use gentle, fragrance-free products');
    }
    
    if (healingPct < 30) {
      precautions.push('Seek immediate medical attention if condition worsens');
    }
    
    return precautions;
  };

  const generateRiskFactors = (reportData) => {
    const healingPct = reportData.overall_healing_percentage;
    const timeSpan = reportData.time_span_days;
    
    let riskFactors = ['Monitor for infection', 'Watch for delayed healing'];
    
    if (healingPct < 20) {
      riskFactors.push('High risk of complications');
      riskFactors.push('May require surgical intervention');
    }
    
    if (timeSpan > 30 && healingPct < 50) {
      riskFactors.push('Chronic wound development risk');
    }
    
    return riskFactors;
  };

  const generateComprehensivePDFContent = (reportData) => {
    const { patient, analysis, treatmentPlan, comprehensiveData } = reportData;
    
    return `
COMPREHENSIVE WOUND HEALING REPORT
==================================

Patient Information:
- Name: ${patient.name}
- Patient ID: ${patient.id}
- Date of Birth: ${patient.dateOfBirth}
- Gender: ${patient.gender}
- Contact: ${patient.contact}
- Analysis Date: ${formatDate(analysis.timestamp)}

Comprehensive Analysis Summary:
- Total Records: ${comprehensiveData.totalRecords}
- Time Span: ${comprehensiveData.timeSpanDays} days
- Overall Healing Progress: ${comprehensiveData.overallHealingPercentage.toFixed(1)}%
- Initial Area: ${comprehensiveData.allRecords[comprehensiveData.allRecords.length - 1]?.area_pixels || 0} pixels
- Current Area: ${comprehensiveData.allRecords[0]?.area_pixels || 0} pixels
- Average Area: ${comprehensiveData.statistics.average_area.toFixed(1)} pixels
- Healing Rate: ${comprehensiveData.statistics.healing_rate_per_day.toFixed(2)}% per day

Latest Analysis:
- Wound Type: ${analysis.woundType}
- Severity: ${analysis.severity}
- Area: ${analysis.area} cm²
- Estimated Healing Time: ${analysis.healingTime} days
- Confidence Level: ${(analysis.confidence * 100).toFixed(1)}%
- Model Version: ${analysis.modelVersion}

Treatment Recommendations:
${treatmentPlan.recommendations.map(rec => `• ${rec}`).join('\n')}

Follow-up Schedule:
${treatmentPlan.followUpSchedule.map(day => `• ${day}`).join('\n')}

Important Precautions:
${treatmentPlan.precautions.map(prec => `• ${prec}`).join('\n')}

Risk Assessment:
${treatmentPlan.riskFactors.map(risk => `• ${risk}`).join('\n')}

Healing Progress Timeline:
${comprehensiveData.healingProgress.map(progress => 
  `• ${formatDate(progress.date)}: ${progress.progress_percentage.toFixed(1)}% healing (${progress.days_between} days)`
).join('\n')}

All Records Summary:
${comprehensiveData.allRecords.map((record, index) => 
  `Record ${index + 1}: ${formatDate(record.timestamp)} - Area: ${record.area_pixels} pixels${record.notes ? ` - Notes: ${record.notes}` : ''}`
).join('\n')}

This comprehensive report was generated by the Wound Healing Progress Tracker system.

Report Generated: ${formatDate(new Date())}
Report ID: ${generateReportId()}
Clinician: ${patient.clinician || 'System Generated'}
    `;
  };

  const downloadComprehensivePDF = async (reportData, reportId) => {
    try {
      if (Platform.OS === 'web') {
        const fileName = await generateComprehensivePDFFile(reportData, reportId);
        Alert.alert('Success', `Comprehensive report "${fileName}" downloaded successfully!`);
      } else {
        try {
          const textContent = generateComprehensivePDFContent(reportData);
          const fileName = `comprehensive-report-${reportId}.txt`;
          const fileUri = `${FileSystem.documentDirectory}${fileName}`;
          
          await FileSystem.writeAsStringAsync(fileUri, textContent);
          
          const isAvailable = await Sharing.isAvailableAsync();
          if (isAvailable) {
            await Sharing.shareAsync(fileUri, {
              mimeType: 'text/plain',
              dialogTitle: 'Comprehensive Report',
            });
          } else {
            throw new Error('Sharing not available');
          }
        } catch (fileError) {
          console.log('File sharing failed, showing in alert:', fileError);
          const textContent = generateComprehensivePDFContent(reportData);
          
          Alert.alert(
            'Comprehensive Report',
            textContent.substring(0, 500) + (textContent.length > 500 ? '...' : ''),
            [
              { text: 'OK' },
              {
                text: 'View Full Report',
                onPress: () => {
                  Alert.alert(
                    'Full Comprehensive Report',
                    textContent,
                    [{ text: 'OK' }],
                    { cancelable: true }
                  );
                }
              }
            ]
          );
        }
      }
    } catch (error) {
      console.error('Download error:', error);
      Alert.alert('Error', 'Failed to generate comprehensive report.');
    }
  };

  const generateComprehensivePDFFile = async (reportData, reportId) => {
    try {
      if (Platform.OS === 'web') {
        const htmlContent = generateComprehensiveHTMLContent(reportData, reportId);
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `comprehensive-report-${reportId}.html`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        return `comprehensive-report-${reportId}.html`;
      } else {
        const textContent = generateComprehensivePDFContent(reportData);
        const fileName = `comprehensive-report-${reportId}.txt`;
        
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

  const generateComprehensiveHTMLContent = (reportData, reportId) => {
    const { patient, analysis, treatmentPlan, comprehensiveData } = reportData;
    
    return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>Comprehensive Wound Healing Report</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }
        .header { text-align: center; border-bottom: 2px solid #667eea; padding-bottom: 20px; margin-bottom: 30px; }
        .title { font-size: 24px; font-weight: bold; color: #667eea; margin-bottom: 10px; }
        .subtitle { font-size: 18px; color: #666; }
        .section { margin-bottom: 25px; }
        .section-title { font-size: 18px; font-weight: bold; color: #667eea; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .info-item { margin-bottom: 8px; }
        .info-label { font-weight: bold; display: inline-block; width: 150px; }
        .recommendations, .precautions, .risks { margin-left: 20px; }
        .recommendations li, .precautions li, .risks li { margin-bottom: 5px; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }
        .report-id { font-weight: bold; color: #667eea; }
        .progress-bar { background-color: #f0f0f0; border-radius: 10px; padding: 3px; margin: 5px 0; }
        .progress-fill { background-color: #27ae60; height: 20px; border-radius: 8px; text-align: center; color: white; line-height: 20px; }
      </style>
    </head>
    <body>
      <div class="header">
        <div class="title">COMPREHENSIVE WOUND HEALING REPORT</div>
        <div class="subtitle">PATIENT: ${patient.id}</div>
      </div>
      
      <div class="section">
        <div class="section-title">Patient Information</div>
        <div class="info-item"><span class="info-label">Name:</span> ${patient.name}</div>
        <div class="info-item"><span class="info-label">Patient ID:</span> ${patient.id}</div>
        <div class="info-item"><span class="info-label">Date of Birth:</span> ${patient.dateOfBirth}</div>
        <div class="info-item"><span class="info-label">Gender:</span> ${patient.gender}</div>
        <div class="info-item"><span class="info-label">Contact:</span> ${patient.contact}</div>
        <div class="info-item"><span class="info-label">Analysis Date:</span> ${formatDate(analysis.timestamp)}</div>
      </div>
      
      <div class="section">
        <div class="section-title">Comprehensive Analysis Summary</div>
        <div class="info-item"><span class="info-label">Total Records:</span> ${comprehensiveData.totalRecords}</div>
        <div class="info-item"><span class="info-label">Time Span:</span> ${comprehensiveData.timeSpanDays} days</div>
        <div class="info-item"><span class="info-label">Overall Healing:</span> ${comprehensiveData.overallHealingPercentage.toFixed(1)}%</div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${Math.min(100, comprehensiveData.overallHealingPercentage)}%">
            ${comprehensiveData.overallHealingPercentage.toFixed(1)}%
          </div>
        </div>
        <div class="info-item"><span class="info-label">Healing Rate:</span> ${comprehensiveData.statistics.healing_rate_per_day.toFixed(2)}% per day</div>
      </div>
      
      <div class="section">
        <div class="section-title">Latest Analysis</div>
        <div class="info-item"><span class="info-label">Wound Type:</span> ${analysis.woundType}</div>
        <div class="info-item"><span class="info-label">Severity:</span> ${analysis.severity}</div>
        <div class="info-item"><span class="info-label">Area:</span> ${analysis.area} cm²</div>
        <div class="info-item"><span class="info-label">Healing Time:</span> ${analysis.healingTime} days</div>
        <div class="info-item"><span class="info-label">Confidence:</span> ${(analysis.confidence * 100).toFixed(1)}%</div>
      </div>
      
      <div class="section">
        <div class="section-title">Treatment Recommendations</div>
        <ul class="recommendations">
          ${treatmentPlan.recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Follow-up Schedule</div>
        <ul class="recommendations">
          ${treatmentPlan.followUpSchedule.map(day => `<li>${day}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Important Precautions</div>
        <ul class="precautions">
          ${treatmentPlan.precautions.map(prec => `<li>${prec}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Risk Assessment</div>
        <ul class="risks">
          ${treatmentPlan.riskFactors.map(risk => `<li>${risk}</li>`).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">Healing Progress Timeline</div>
        <ul class="recommendations">
          ${comprehensiveData.healingProgress.map(progress => 
            `<li>${formatDate(progress.date)}: ${progress.progress_percentage.toFixed(1)}% healing (${progress.days_between} days)</li>`
          ).join('')}
        </ul>
      </div>
      
      <div class="section">
        <div class="section-title">All Records Summary</div>
        <ul class="recommendations">
          ${comprehensiveData.allRecords.map((record, index) => 
            `<li>Record ${index + 1}: ${formatDate(record.timestamp)} - Area: ${record.area_pixels} pixels${record.notes ? ` - Notes: ${record.notes}` : ''}</li>`
          ).join('')}
        </ul>
      </div>
      
      <div class="footer">
        <p>This comprehensive report was generated by the Wound Healing Progress Tracker system.</p>
        <p>Report Generated: ${formatDate(new Date())}</p>
        <p>Report ID: <span class="report-id">${reportId}</span></p>
        <p>Clinician: ${patient.clinician || 'System Generated'}</p>
      </div>
    </body>
    </html>
    `;
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content>
          <Title>📄 Generate Reports</Title>
          <Paragraph>
            Complete patient information and generate comprehensive reports.
          </Paragraph>
          <View style={styles.reportInfo}>
            <Text style={styles.patientId}>Patient ID: {safeAnalysisData.patientId || generatePatientId()}</Text>
            <Text style={styles.analysisDate}>{formatDate(safeAnalysisData.timestamp)}</Text>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.patientInfoCard}>
        <Card.Content>
          <Title>👤 Patient Information</Title>
          <TextInput
            label="Patient Name *"
            value={patientInfo.name}
            onChangeText={(text) => handleInputChange('name', text)}
            style={styles.input}
            mode="outlined"
            autoCapitalize="words"
          />

          <TextInput
            label="Date of Birth"
            value={patientInfo.dateOfBirth}
            onChangeText={(text) => handleInputChange('dateOfBirth', text)}
            style={styles.input}
            mode="outlined"
            placeholder="YYYY-MM-DD"
          />

          <TextInput
            label="Gender"
            value={patientInfo.gender}
            onChangeText={(text) => handleInputChange('gender', text)}
            style={styles.input}
            mode="outlined"
            placeholder="Male/Female/Other"
          />

          <TextInput
            label="Contact Information"
            value={patientInfo.contact}
            onChangeText={(text) => handleInputChange('contact', text)}
            style={styles.input}
            mode="outlined"
            placeholder="Phone/Email"
          />

          <TextInput
            label="Address"
            value={patientInfo.address}
            onChangeText={(text) => handleInputChange('address', text)}
            style={styles.input}
            mode="outlined"
            multiline
            numberOfLines={2}
          />

          <TextInput
            label="Clinician Name"
            value={patientInfo.clinician}
            onChangeText={(text) => handleInputChange('clinician', text)}
            style={styles.input}
            mode="outlined"
            autoCapitalize="words"
          />

          <TextInput
            label="Additional Notes"
            value={patientInfo.notes}
            onChangeText={(text) => handleInputChange('notes', text)}
            style={styles.input}
            mode="outlined"
            multiline
            numberOfLines={3}
            placeholder="Any additional information..."
          />
        </Card.Content>
      </Card>

      <Card style={styles.analysisSummaryCard}>
        <Card.Content>
          <Title>🔬 Analysis Summary</Title>
          <View style={styles.summaryGrid}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Wound Type</Text>
              <Text style={[styles.summaryValue, { color: getWoundTypeColor(safeAnalysisData.woundType) }]}>
                {(safeAnalysisData.woundType || 'unknown').toUpperCase()}
              </Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Severity</Text>
              <Text style={styles.summaryValue}>{safeAnalysisData.severity || 'moderate'}</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Area</Text>
              <Text style={styles.summaryValue}>{safeAnalysisData.area || 5.0} cm²</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryLabel}>Healing Time</Text>
              <Text style={styles.summaryValue}>{safeAnalysisData.healingTime || 21} days</Text>
            </View>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.reportTypeCard}>
        <Card.Content>
          <Title>📋 Report Type</Title>
          <View style={styles.reportTypeButtons}>
            <Button
              mode={reportType === 'patient' ? 'contained' : 'outlined'}
              onPress={() => setReportType('patient')}
              style={styles.reportTypeButton}
              icon="account"
            >
              Patient Report
            </Button>
            <Button
              mode={reportType === 'clinician' ? 'contained' : 'outlined'}
              onPress={() => setReportType('clinician')}
              style={styles.reportTypeButton}
              icon="doctor"
            >
              Clinician Report
            </Button>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.generateCard}>
        <Card.Content>
          <Title>📄 Generate Report</Title>
          <Paragraph>
            {reportType === 'patient' 
              ? 'Generate a patient-friendly report with clear instructions and care recommendations.'
              : 'Generate a detailed clinical report with technical analysis and treatment protocols.'
            }
          </Paragraph>
          
          <View style={styles.generateButtons}>
            <Button
              mode="contained"
              onPress={() => generateReport(reportType)}
              style={styles.generateButton}
              loading={isGenerating}
              disabled={isGenerating}
              icon="file-document"
            >
              Generate {reportType.charAt(0).toUpperCase() + reportType.slice(1)} Report
            </Button>
            
            <Button
              mode="contained"
              onPress={() => navigation.navigate('History', { 
                patientId: safeAnalysisData.patientId || generatePatientId()
              })}
              style={[styles.actionButton, { backgroundColor: '#e74c3c' }]}
              icon="history"
            >
              View History
            </Button>
            
            <Button
              mode="outlined"
              onPress={() => navigation.navigate('DoctorAppointment', { 
                patientInfo,
                analysisData: safeAnalysisData,
                treatmentPlan: safeTreatmentPlan 
              })}
              style={styles.appointmentButton}
              icon="calendar-plus"
            >
              Book Appointment to Doctor
            </Button>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.patientIdCard}>
        <Card.Content>
          <Title>🔍 Generate Report by Patient ID</Title>
          <Paragraph>
            Enter a patient ID to generate a comprehensive report with all stored data for that patient.
          </Paragraph>
          
          <TextInput
            label="Patient ID"
            value={patientIdInput}
            onChangeText={setPatientIdInput}
            style={styles.input}
            mode="outlined"
            placeholder="Enter patient ID (e.g., P123456)"
            autoCapitalize="characters"
          />
          
          <View style={styles.fetchButtons}>
            <Button
              mode="contained"
              onPress={() => fetchPatientData(patientIdInput)}
              style={styles.fetchButton}
              loading={isLoadingPatientData}
              disabled={isLoadingPatientData || !patientIdInput.trim()}
              icon="magnify"
            >
              Fetch Patient Data
            </Button>
            
            <Button
              mode="outlined"
              onPress={() => navigation.navigate('History', { 
                patientId: patientIdInput.trim()
              })}
              style={styles.historyButton}
              disabled={!patientIdInput.trim()}
              icon="history"
            >
              View History
            </Button>
          </View>
          
          {patientReportData && (
            <View style={styles.patientDataSummary}>
              <Title style={styles.summaryTitle}>📊 Patient Data Summary</Title>
              <View style={styles.summaryGrid}>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Patient ID</Text>
                  <Text style={styles.summaryValue}>{patientReportData.patient_id}</Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Total Records</Text>
                  <Text style={styles.summaryValue}>{patientReportData.total_records}</Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Time Span</Text>
                  <Text style={styles.summaryValue}>{patientReportData.time_span_days} days</Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Healing Progress</Text>
                  <Text style={[styles.summaryValue, { color: '#27ae60' }]}>
                    {patientReportData.overall_healing_percentage.toFixed(1)}%
                  </Text>
                </View>
              </View>
              
              <Button
                mode="contained"
                onPress={() => generateComprehensiveReport(patientReportData)}
                style={styles.comprehensiveReportButton}
                icon="file-document-multiple"
              >
                Generate Comprehensive Report
              </Button>
            </View>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.featuresCard}>
        <Card.Content>
          <Title>✨ Report Features</Title>
          <List.Section>
            <List.Item
              title="Patient Information"
              description="Complete patient details and demographics"
              left={() => <List.Icon icon="account" color="#3498db" />}
            />
            <List.Item
              title="Wound Analysis"
              description="Detailed wound classification and characteristics"
              left={() => <List.Icon icon="microscope" color="#9b59b6" />}
            />
            <List.Item
              title="Treatment Recommendations"
              description="Evidence-based treatment protocols"
              left={() => <List.Icon icon="medical" color="#27ae60" />}
            />
            <List.Item
              title="Follow-up Schedule"
              description="Automated appointment scheduling"
              left={() => <List.Icon icon="calendar" color="#f39c12" />}
            />
            <List.Item
              title="Risk Assessment"
              description="Complication risk factors and prevention"
              left={() => <List.Icon icon="alert" color="#e74c3c" />}
            />
            <List.Item
              title="Patient History"
              description="Previous treatments and outcomes"
              left={() => <List.Icon icon="history" color="#34495e" />}
            />
          </List.Section>
        </Card.Content>
      </Card>

      <View style={styles.actionButtons}>
        <Button
          mode="outlined"
          onPress={() => navigation.goBack()}
          style={styles.actionButton}
          icon="arrow-left"
        >
          Back to Treatment Plan
        </Button>
        
        <Button
          mode="contained"
          onPress={() => navigation.navigate('Home')}
          style={[styles.actionButton, { backgroundColor: '#667eea' }]}
          icon="home"
        >
          New Analysis
        </Button>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  headerCard: {
    margin: 15,
    elevation: 4,
  },
  reportInfo: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#e3f2fd',
    borderRadius: 10,
  },
  patientId: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  analysisDate: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  patientInfoCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  input: {
    marginBottom: 15,
  },
  analysisSummaryCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  summaryItem: {
    width: '48%',
    marginBottom: 15,
    alignItems: 'center',
  },
  summaryLabel: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 5,
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  reportTypeCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  reportTypeButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  reportTypeButton: {
    flex: 0.48,
  },
  generateCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  generateButtons: {
    marginTop: 15,
  },
  generateButton: {
    marginBottom: 10,
    backgroundColor: '#667eea',
  },
  appointmentButton: {
    borderColor: '#3498db',
    borderWidth: 1,
  },
  featuresCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 15,
    paddingTop: 0,
  },
  actionButton: {
    flex: 0.48,
  },
  patientIdCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  fetchButtons: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 15,
  },
  fetchButton: {
    flex: 1,
    backgroundColor: '#3498db',
  },
  historyButton: {
    flex: 1,
    borderColor: '#e74c3c',
  },
  patientDataSummary: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#e8f5e8',
    borderRadius: 10,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#27ae60',
    marginBottom: 15,
  },
  comprehensiveReportButton: {
    marginTop: 15,
    backgroundColor: '#27ae60',
  },
});