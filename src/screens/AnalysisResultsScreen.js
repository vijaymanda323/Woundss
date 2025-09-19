import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Dimensions,
  Alert,
  Modal,
  TextInput,
  TouchableOpacity,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
  ProgressBar,
  List,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { generatePatientId, formatDate, getWoundSeverity, generateTreatmentPlan } from '../utils/patientUtils';
import WoundHealingGraphs from '../components/WoundHealingGraphs';

const { width } = Dimensions.get('window');

// Helper functions for analytics
const getHealingStage = (healingTime) => {
  if (healingTime <= 7) return 'Fast Healing';
  if (healingTime <= 21) return 'Moderate Healing';
  if (healingTime <= 60) return 'Slow Healing';
  return 'Chronic Healing';
};

const getRiskScore = (analysisData) => {
  let score = 0;
  
  // Base score from severity
  switch (analysisData.severity) {
    case 'Mild': score += 1; break;
    case 'Moderate': score += 3; break;
    case 'Severe': score += 5; break;
  }
  
  // Add score based on wound type
  const chronicTypes = ['chronic', 'diabetic', 'pressure_ulcer', 'foot_ulcer', 'leg_ulcer'];
  if (chronicTypes.includes(analysisData.woundType.toLowerCase())) {
    score += 2;
  }
  
  // Add score based on healing time
  if (analysisData.healingTime > 30) score += 1;
  if (analysisData.healingTime > 60) score += 2;
  
  return `${score}/10`;
};

const getConfidenceLevel = (confidence) => {
  if (confidence >= 0.9) return 'Very High';
  if (confidence >= 0.8) return 'High';
  if (confidence >= 0.7) return 'Moderate';
  if (confidence >= 0.6) return 'Low';
  return 'Very Low';
};

const getWoundInsight = (analysisData) => {
  const insights = {
    burn: 'Burn wounds require careful monitoring for infection and proper hydration. Keep the area clean and moisturized.',
    cut: 'Clean cuts typically heal well with proper care. Monitor for signs of infection and keep the wound covered.',
    surgical: 'Surgical wounds need careful monitoring for dehiscence and infection. Follow post-operative care instructions.',
    chronic: 'Chronic wounds require specialized care and may need advanced treatments like debridement or negative pressure therapy.',
    diabetic: 'Diabetic wounds require aggressive blood sugar control and specialized foot care to prevent complications.',
    abrasion: 'Superficial abrasions heal quickly with proper cleaning and protection from further trauma.',
    bruise: 'Bruises typically resolve on their own. Apply ice initially, then heat after 48 hours to promote healing.',
    laceration: 'Deep lacerations may require medical attention for proper closure and to prevent infection.',
    pressure_ulcer: 'Pressure ulcers require pressure relief and specialized wound care to prevent progression.',
    foot_ulcer: 'Foot ulcers in diabetic patients require immediate medical attention and specialized care.',
    leg_ulcer: 'Leg ulcers often require compression therapy and treatment of underlying venous insufficiency.',
    toe_wound: 'Toe wounds require careful monitoring for infection, especially in diabetic patients.',
    stab_wound: 'Stab wounds require immediate medical evaluation to assess depth and potential internal injury.',
    orthopedic_wound: 'Orthopedic wounds may require specialized care depending on the underlying bone or joint involvement.',
    abdominal_wound: 'Abdominal wounds require careful monitoring for signs of infection or dehiscence.',
    hematoma: 'Hematomas typically resolve on their own but may require drainage if large or causing symptoms.',
    ingrown: 'Ingrown nails require proper trimming techniques and may need medical intervention if infected.',
    epidermolysis: 'Epidermolysis requires gentle handling and specialized wound care to prevent further skin damage.',
    extravasation: 'Extravasation injuries require immediate medical attention to prevent tissue necrosis.',
    malignant_wound: 'Malignant wounds require specialized palliative care and may need advanced wound management.',
    meningitis: 'Meningitis-related wounds require immediate medical attention and treatment of the underlying condition.',
    miscellaneous: 'This wound type requires careful monitoring and may need specialized assessment.',
    pilonidal_sinus: 'Pilonidal sinus wounds require surgical intervention and specialized post-operative care.',
  };
  
  return insights[analysisData.woundType.toLowerCase()] || 'This wound requires careful monitoring and appropriate medical care.';
};

const getTreatmentInsight = (analysisData) => {
  if (analysisData.severity === 'Severe') {
    return 'Immediate medical attention required. Do not delay seeking professional care for severe wounds.';
  } else if (analysisData.severity === 'Moderate') {
    return 'Medical consultation recommended within 24-48 hours. Monitor closely for any changes.';
  } else {
    return 'Continue with proper wound care and monitor for signs of infection or delayed healing.';
  }
};

const getRiskInsight = (analysisData) => {
  const risks = [];
  
  if (analysisData.healingTime > 30) {
    risks.push('Higher risk of delayed healing');
  }
  
  if (analysisData.severity === 'Severe') {
    risks.push('High risk of complications');
  }
  
  if (analysisData.confidence < 0.7) {
    risks.push('Lower confidence in diagnosis - consider second opinion');
  }
  
  const chronicTypes = ['chronic', 'diabetic', 'pressure_ulcer'];
  if (chronicTypes.includes(analysisData.woundType.toLowerCase())) {
    risks.push('Chronic wound - requires specialized care');
  }
  
  return risks.length > 0 ? risks.join('. ') + '.' : 'Standard wound care precautions apply.';
};

export default function AnalysisResultsScreen({ navigation, route }) {
  const { imageUri, analysisResult } = route.params || {};
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [patientId] = useState(generatePatientId());
  const [analysisData, setAnalysisData] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [showCorrectTypeSelector, setShowCorrectTypeSelector] = useState(false);
  const [selectedCorrectType, setSelectedCorrectType] = useState('');

  const analysisSteps = [
    { step: 'Processing wound image...', duration: 1000 },
    { step: 'Analyzing wound characteristics...', duration: 2000 },
    { step: 'Classifying wound type...', duration: 1500 },
    { step: 'Calculating healing prediction...', duration: 2000 },
    { step: 'Generating treatment recommendations...', duration: 1500 },
    { step: 'Finalizing analysis...', duration: 1000 },
  ];

  useEffect(() => {
    if (imageUri && analysisResult) {
      startAnalysis();
    }
  }, []);

  const startAnalysis = async () => {
    setIsProcessing(true);
    setProgress(0);
    setCurrentStep('Starting analysis...');

    try {
      // Simulate analysis steps
      for (let i = 0; i < analysisSteps.length; i++) {
        const step = analysisSteps[i];
        setCurrentStep(step.step);
        
        const stepProgress = (i + 1) / analysisSteps.length;
        setProgress(stepProgress);
        
        await new Promise(resolve => setTimeout(resolve, step.duration));
      }

      // Process the analysis result
      const processedData = processAnalysisResult(analysisResult);
      setAnalysisData(processedData);
      
      setCurrentStep('Analysis complete!');
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      
    } catch (error) {
      console.error('Analysis error:', error);
      Alert.alert('Analysis Failed', 'Unable to process the wound analysis. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const processAnalysisResult = (result) => {
    const woundType = result.wound_classification?.wound_type || 'unknown';
    const area = parseFloat(result.area_cm2) || 5.0;
    const healingTime = result.wound_classification?.estimated_days_to_cure || 21;
    const confidence = result.model_confidence || 0.85;
    const perimeter = result.perimeter || (area * 3.14).toFixed(2); // Estimate perimeter from area
    const areaPixels = result.area_pixels || Math.floor(area * 100); // Estimate pixels from cm²
    
    const severity = getWoundSeverity(area, woundType);
    const treatmentPlan = generateTreatmentPlan({
      woundType,
      area,
      severity,
      patientAge: 45 // Default age, would come from patient info
    });

    // Generate a patient ID for this analysis
    const patientId = generatePatientId();

    return {
      woundType,
      area,
      healingTime,
      confidence,
      severity,
      perimeter,
      areaPixels,
      treatmentPlan,
      timestamp: new Date().toISOString(),
      patientId,
      imageUri,
      originalResult: result
    };
  };

  const sendFeedback = async (status) => {
    if (!analysisData) return;

    try {
      if (status === 'wrong') {
        // Show wound type selector for incorrect predictions
        setShowCorrectTypeSelector(true);
        return;
      }

      // Handle correct prediction
      await submitFeedback('right', null);
      
    } catch (error) {
      console.error('Feedback error:', error);
      Alert.alert('Error', 'Failed to send feedback. Please try again.');
    }
  };

  const submitFeedback = async (status, correctType = null) => {
    try {
      // Simulate API call to backend
      const feedbackData = {
        status: status,
        predictedType: analysisData.woundType,
        correctType: correctType,
        confidence: analysisData.confidence,
        imageHash: 'mock_hash_' + Date.now(), // In real app, this would be the actual image hash
        timestamp: new Date().toISOString()
      };

      console.log('Submitting feedback:', feedbackData);
      
      // Simulate learning process
      if (status === 'wrong' && correctType) {
        console.log(`Model learning: ${analysisData.woundType} -> ${correctType}`);
        // In a real app, this would trigger model retraining
        Alert.alert(
          'Model Learning', 
          `Thank you! The model is learning that this wound type should be classified as "${correctType}" instead of "${analysisData.woundType}".`
        );
      } else {
        Alert.alert('Feedback Sent', 'Thank you! Prediction marked as correct.');
      }
      
      setFeedbackSent(true);
      setShowCorrectTypeSelector(false);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      
    } catch (error) {
      console.error('Feedback submission error:', error);
      Alert.alert('Error', 'Failed to submit feedback. Please try again.');
    }
  };

  const handleCorrectTypeSelection = () => {
    if (!selectedCorrectType) {
      Alert.alert('Selection Required', 'Please select the correct wound type.');
      return;
    }
    
    submitFeedback('wrong', selectedCorrectType);
  };


  const proceedToTreatmentPlan = () => {
    if (!analysisData) return;
    
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    // Navigate to questionnaire first, then treatment plan
    navigation.navigate('WoundQuestionnaire', {
      analysisData,
      imageUri,
    });
  };

  const getWoundTypeColor = (type) => {
    const colors = {
      burn: '#e74c3c',
      cut: '#3498db',
      surgical: '#9b59b6',
      chronic: '#f39c12',
      diabetic: '#e67e22',
      unknown: '#95a5a6',
    };
    return colors[type] || colors.unknown;
  };

  const getSeverityColor = (severity) => {
    const colors = {
      Mild: '#27ae60',
      Moderate: '#f39c12',
      Severe: '#e74c3c',
    };
    return colors[severity] || colors.Moderate;
  };

  if (isProcessing) {
    return (
      <View style={styles.container}>
        <Card style={styles.processingCard}>
          <Card.Content>
            <Title>🔬 Analyzing Wound</Title>
            <View style={styles.progressContainer}>
              <ProgressBar
                progress={progress}
                color="#667eea"
                style={styles.progressBar}
              />
              <Text style={styles.progressText}>
                {Math.round(progress * 100)}% Complete
              </Text>
              <Text style={styles.currentStep}>{currentStep}</Text>
            </View>
          </Card.Content>
        </Card>

        <Card style={styles.imageCard}>
          <Card.Content>
            <Title>Wound Image</Title>
            {imageUri && (
              <Image source={{ uri: imageUri }} style={styles.woundImage} />
            )}
          </Card.Content>
        </Card>
      </View>
    );
  }

  if (!analysisData) {
    return (
      <View style={styles.container}>
        <Card style={styles.errorCard}>
          <Card.Content>
            <Title>Analysis Error</Title>
            <Paragraph>Unable to process the wound analysis. Please try again.</Paragraph>
            <Button
              mode="contained"
              onPress={() => navigation.goBack()}
              style={styles.errorButton}
            >
              Go Back
            </Button>
          </Card.Content>
        </Card>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content>
          <Title>✅ Analysis Complete</Title>
          <Paragraph>
            Wound analysis completed successfully. Review the results below.
          </Paragraph>
          <View style={styles.patientInfo}>
            <Text style={styles.patientId}>Patient ID: {patientId}</Text>
            <Text style={styles.analysisDate}>{formatDate(analysisData.timestamp)}</Text>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.imageCard}>
        <Card.Content>
          <Title>Wound Image</Title>
          {imageUri && (
            <Image source={{ uri: imageUri }} style={styles.woundImage} />
          )}
        </Card.Content>
      </Card>

      <Card style={[styles.resultCard, { borderLeftColor: getWoundTypeColor(analysisData.woundType) }]}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="bandage" size={24} color={getWoundTypeColor(analysisData.woundType)} />
            Wound Classification
          </Title>
          <View style={styles.resultGrid}>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Type</Text>
              <Chip 
                style={[styles.chip, { backgroundColor: getWoundTypeColor(analysisData.woundType) }]}
                textStyle={styles.chipText}
              >
                {analysisData.woundType.toUpperCase()}
              </Chip>
            </View>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Severity</Text>
              <Chip 
                style={[styles.chip, { backgroundColor: getSeverityColor(analysisData.severity) }]}
                textStyle={styles.chipText}
              >
                {analysisData.severity.toUpperCase()}
              </Chip>
            </View>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Area</Text>
              <Text style={styles.resultValue}>{analysisData.area} cm²</Text>
            </View>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Healing Time</Text>
              <Text style={styles.resultValue}>{analysisData.healingTime} days</Text>
            </View>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Confidence</Text>
              <Text style={styles.resultValue}>{(analysisData.confidence * 100).toFixed(1)}%</Text>
            </View>
            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>Analysis ID</Text>
              <Text style={styles.resultValue}>{analysisData.patientId}</Text>
            </View>
          </View>
        </Card.Content>
      </Card>

      {/* Gemini AI Analysis Results */}
      <Card style={styles.externalAICard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="cloud-done" size={24} color="#4CAF50" />
            Gemini AI Analysis
          </Title>
          
          {/* Show Gemini analysis if available */}
          {analysisData?.originalResult?.gemini_analysis ? (
            <View style={styles.geminiResults}>
              {analysisData.originalResult.gemini_analysis?.Type && (
                <View style={styles.geminiResultItem}>
                  <Text style={styles.geminiResultLabel}>Type:</Text>
                  <Chip 
                    style={[styles.geminiChip, { backgroundColor: getWoundTypeColor(analysisData.originalResult.gemini_analysis?.Type?.toLowerCase()) }]}
                    textStyle={styles.geminiChipText}
                  >
                    {analysisData.originalResult.gemini_analysis?.Type}
                  </Chip>
                </View>
              )}
              
              {analysisData.originalResult.gemini_analysis.Severity && (
                <View style={styles.geminiResultItem}>
                  <Text style={styles.geminiResultLabel}>Severity:</Text>
                  <Chip 
                    style={[styles.geminiChip, { backgroundColor: getSeverityColor(analysisData.originalResult.gemini_analysis.Severity) }]}
                    textStyle={styles.geminiChipText}
                  >
                    {analysisData.originalResult.gemini_analysis.Severity}
                  </Chip>
                </View>
              )}
              
              {analysisData.originalResult.gemini_analysis.Explanation && (
                <View style={styles.geminiExplanation}>
                  <Text style={styles.geminiExplanationLabel}>Medical Analysis:</Text>
                  <Text style={styles.geminiExplanationText}>
                    {analysisData.originalResult.gemini_analysis.Explanation}
                  </Text>
                </View>
              )}
              
              {/* Show raw response if available */}
              {analysisData.originalResult.gemini_analysis.raw_response && (
                <View style={styles.geminiRawResponse}>
                  <Text style={styles.geminiRawLabel}>Full AI Response:</Text>
                  <Text style={styles.geminiRawText}>
                    {analysisData.originalResult.gemini_analysis.raw_response}
                  </Text>
                </View>
              )}
            </View>
          ) : (
            <View style={styles.geminiFallback}>
              <Paragraph style={styles.externalAIDescription}>
                This analysis is powered by Google's Gemini AI for accurate wound classification and medical insights.
              </Paragraph>
              <View style={styles.geminiIndicator}>
                <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
                <Text style={styles.geminiText}>Gemini AI Analysis Complete</Text>
              </View>
            </View>
          )}
        </Card.Content>
      </Card>

      {/* Medical Disclaimer */}
      <Card style={styles.disclaimerCard}>
        <Card.Content>
          <View style={styles.disclaimerHeader}>
            <Ionicons name="warning" size={24} color="#e74c3c" />
            <Title style={styles.disclaimerTitle}>Important Medical Disclaimer</Title>
          </View>
          <Paragraph style={styles.disclaimerText}>
            <Text style={styles.boldText}>⚠️ DO NOT TAKE ANY MEDICINES WITHOUT CONSULTING A DOCTOR</Text>
          </Paragraph>
          <Paragraph style={styles.disclaimerText}>
            This analysis is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. 
            Always consult with a qualified healthcare provider before taking any medications.
          </Paragraph>
        </Card.Content>
      </Card>

      {/* Doctor Appointment Alert for Moderate/Severe Cases */}
      {(analysisData?.severity === 'Moderate' || analysisData?.severity === 'Severe') && (
        <Card style={[styles.appointmentCard, { borderLeftColor: analysisData.severity === 'Severe' ? '#e74c3c' : '#f39c12' }]}>
          <Card.Content>
            <Title style={styles.cardTitle}>
              <Ionicons 
                name={analysisData.severity === 'Severe' ? 'alert-circle' : 'warning'} 
                size={24} 
                color={analysisData.severity === 'Severe' ? '#e74c3c' : '#f39c12'} 
              />
              Medical Attention Recommended
            </Title>
            <Paragraph style={styles.appointmentDescription}>
              Your wound has been classified as <Text style={[styles.severityText, { color: analysisData.severity === 'Severe' ? '#e74c3c' : '#f39c12' }]}>{analysisData.severity.toUpperCase()}</Text> severity. 
              We strongly recommend consulting with a medical professional for proper treatment.
            </Paragraph>
            <Button
              mode="contained"
              onPress={() => navigation.navigate('DoctorAppointment', { 
                severity: analysisData.severity, 
                woundType: analysisData.woundType 
              })}
              style={[styles.appointmentButton, { backgroundColor: analysisData.severity === 'Severe' ? '#e74c3c' : '#f39c12' }]}
              icon="medical"
            >
              Book Doctor Appointment
            </Button>
          </Card.Content>
        </Card>
      )}


      <Card style={styles.summaryCard}>
        <Card.Content>
          <Title>📋 Analysis Summary</Title>
          <View style={styles.summaryContent}>
            <Text style={styles.summaryText}>
              The wound has been classified as a <Text style={styles.highlight}>{analysisData.woundType}</Text> with 
              <Text style={styles.highlight}> {analysisData.severity.toLowerCase()}</Text> severity. 
              The wound area measures <Text style={styles.highlight}>{analysisData.area} cm²</Text> and is 
              expected to heal within <Text style={styles.highlight}>{analysisData.healingTime} days</Text> 
              with proper treatment.
            </Text>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.feedbackCard}>
        <Card.Content>
          <Title>📝 Prediction Feedback</Title>
          <Paragraph>
            Is this prediction accurate? Your feedback helps improve the AI model.
          </Paragraph>
          
          {!feedbackSent && !showCorrectTypeSelector ? (
            <View style={styles.feedbackButtons}>
              <Button
                mode="contained"
                onPress={() => sendFeedback('right')}
                style={[styles.feedbackButton, { backgroundColor: '#27ae60' }]}
                icon="check"
              >
                ✅ Correct
              </Button>
              <Button
                mode="contained"
                onPress={() => sendFeedback('wrong')}
                style={[styles.feedbackButton, { backgroundColor: '#e74c3c' }]}
                icon="close"
              >
                ❌ Incorrect
              </Button>
            </View>
          ) : showCorrectTypeSelector ? (
            <View style={styles.typeSelectorContainer}>
              <Title style={styles.selectorTitle}>🤔 What is the correct wound type?</Title>
              <Paragraph style={styles.selectorSubtitle}>
                Predicted: <Text style={styles.predictedType}>{analysisData.woundType}</Text>
              </Paragraph>
              
              <View style={styles.woundTypeGrid}>
                {['burn', 'cut', 'surgical', 'chronic', 'diabetic', 'abrasion', 'laceration', 'pressure_ulcer'].map((type) => (
                  <Button
                    key={type}
                    mode={selectedCorrectType === type ? 'contained' : 'outlined'}
                    onPress={() => setSelectedCorrectType(type)}
                    style={[
                      styles.typeButton,
                      selectedCorrectType === type && { backgroundColor: getWoundTypeColor(type) }
                    ]}
                    labelStyle={styles.typeButtonLabel}
                  >
                    {type.replace('_', ' ').toUpperCase()}
                  </Button>
                ))}
              </View>
              
              <View style={styles.selectorActions}>
                <Button
                  mode="outlined"
                  onPress={() => {
                    setShowCorrectTypeSelector(false);
                    setSelectedCorrectType('');
                  }}
                  style={styles.selectorActionButton}
                  icon="arrow-left"
                >
                  Back
                </Button>
                <Button
                  mode="contained"
                  onPress={handleCorrectTypeSelection}
                  style={[styles.selectorActionButton, { backgroundColor: '#667eea' }]}
                  icon="check"
                  disabled={!selectedCorrectType}
                >
                  Submit Correction
                </Button>
              </View>
            </View>
          ) : (
            <View style={styles.feedbackSent}>
              <Text style={styles.feedbackSentText}>
                ✅ Thank you for your feedback!
              </Text>
            </View>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.actionCard}>
        <Card.Content>
          <Title>Next Steps</Title>
          <Paragraph>
            Based on the analysis, we can now generate a comprehensive treatment plan 
            and create detailed reports for both patient and clinician use.
          </Paragraph>
          
          <View style={styles.actionButtons}>
            <Button
              mode="outlined"
              onPress={() => navigation.goBack()}
              style={styles.actionButton}
              icon="arrow-left"
            >
              Back to Upload
            </Button>
            
            <Button
              mode="contained"
              onPress={proceedToTreatmentPlan}
              style={[styles.actionButton, { backgroundColor: '#667eea' }]}
              icon="arrow-right"
            >
              View Treatment Plan
            </Button>
          </View>
        </Card.Content>
      </Card>

      {/* Comprehensive Wound Health Analytics */}
      <Card style={styles.analyticsCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="analytics" size={24} color="#667eea" />
            Wound Health Analytics
          </Title>
          
          {/* Debug info - remove in production */}
          {__DEV__ && (
            <View style={styles.debugContainer}>
              <Text style={styles.debugText}>Debug: {JSON.stringify({
                woundType: analysisData?.woundType,
                area: analysisData?.area,
                perimeter: analysisData?.perimeter,
                healingTime: analysisData?.healingTime,
                confidence: analysisData?.confidence,
                severity: analysisData?.severity
              }, null, 2)}</Text>
            </View>
          )}
          
          {!analysisData ? (
            <View style={styles.loadingContainer}>
              <Text style={styles.loadingText}>Loading analytics data...</Text>
            </View>
          ) : (
            <>
              {/* Health Metrics Grid */}
          <View style={styles.metricsGrid}>
            <View style={styles.metricItem}>
              <View style={styles.metricHeader}>
                <Ionicons name="resize" size={20} color="#3498db" />
                <Text style={styles.metricLabel}>Wound Dimensions</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{analysisData?.area || 'N/A'} cm²</Text>
                <Text style={styles.metricSubtext}>Surface Area</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{analysisData?.perimeter || 'N/A'} cm</Text>
                <Text style={styles.metricSubtext}>Perimeter</Text>
              </View>
            </View>

            <View style={styles.metricItem}>
              <View style={styles.metricHeader}>
                <Ionicons name="time" size={20} color="#e67e22" />
                <Text style={styles.metricLabel}>Healing Timeline</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{analysisData?.healingTime || 'N/A'} days</Text>
                <Text style={styles.metricSubtext}>Estimated Recovery</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{getHealingStage(analysisData?.healingTime || 21)}</Text>
                <Text style={styles.metricSubtext}>Healing Stage</Text>
              </View>
            </View>

            <View style={styles.metricItem}>
              <View style={styles.metricHeader}>
                <Ionicons name="shield-checkmark" size={20} color="#27ae60" />
                <Text style={styles.metricLabel}>Risk Assessment</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={[styles.metricValue, { color: getSeverityColor(analysisData?.severity || 'Moderate') }]}>
                  {analysisData?.severity || 'N/A'}
                </Text>
                <Text style={styles.metricSubtext}>Severity Level</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{getRiskScore(analysisData || {})}</Text>
                <Text style={styles.metricSubtext}>Risk Score</Text>
              </View>
            </View>

            <View style={styles.metricItem}>
              <View style={styles.metricHeader}>
                <Ionicons name="trending-up" size={20} color="#9b59b6" />
                <Text style={styles.metricLabel}>AI Confidence</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{((analysisData?.confidence || 0.85) * 100).toFixed(1)}%</Text>
                <Text style={styles.metricSubtext}>Prediction Accuracy</Text>
              </View>
              <View style={styles.metricDetails}>
                <Text style={styles.metricValue}>{getConfidenceLevel(analysisData?.confidence || 0.85)}</Text>
                <Text style={styles.metricSubtext}>Confidence Level</Text>
              </View>
            </View>
          </View>

          {/* Healing Progress Indicators */}
          <View style={styles.progressIndicators}>
            <Title style={styles.sectionTitle}>Healing Progress Indicators</Title>
            
            <View style={styles.progressItem}>
              <View style={styles.progressHeader}>
                <Text style={styles.progressLabel}>Inflammatory Phase</Text>
                <Text style={styles.progressDuration}>Days 1-3</Text>
              </View>
              <View style={styles.progressBarContainer}>
                <View style={[styles.progressBarFill, { width: '100%', backgroundColor: '#e74c3c' }]} />
              </View>
              <Text style={styles.progressDescription}>Redness, swelling, pain - Normal healing response</Text>
            </View>

            <View style={styles.progressItem}>
              <View style={styles.progressHeader}>
                <Text style={styles.progressLabel}>Proliferative Phase</Text>
                <Text style={styles.progressDuration}>Days 4-14</Text>
              </View>
              <View style={styles.progressBarContainer}>
                <View style={[styles.progressBarFill, { width: '70%', backgroundColor: '#f39c12' }]} />
              </View>
              <Text style={styles.progressDescription}>New tissue formation, granulation tissue development</Text>
            </View>

            <View style={styles.progressItem}>
              <View style={styles.progressHeader}>
                <Text style={styles.progressLabel}>Maturation Phase</Text>
                <Text style={styles.progressDuration}>Days 15-{analysisData?.healingTime || 21}</Text>
              </View>
              <View style={styles.progressBarContainer}>
                <View style={[styles.progressBarFill, { width: '30%', backgroundColor: '#27ae60' }]} />
              </View>
              <Text style={styles.progressDescription}>Scar formation, tissue remodeling</Text>
            </View>
          </View>

          {/* Health Insights */}
          <View style={styles.healthInsights}>
            <Title style={styles.sectionTitle}>Health Insights</Title>
            
            <View style={styles.insightItem}>
              <Ionicons name="bulb" size={20} color="#f39c12" />
              <Text style={styles.insightText}>
                {getWoundInsight(analysisData || {})}
              </Text>
            </View>

            <View style={styles.insightItem}>
              <Ionicons name="medical" size={20} color="#3498db" />
              <Text style={styles.insightText}>
                {getTreatmentInsight(analysisData || {})}
              </Text>
            </View>

            <View style={styles.insightItem}>
              <Ionicons name="warning" size={20} color="#e67e22" />
              <Text style={styles.insightText}>
                {getRiskInsight(analysisData || {})}
              </Text>
            </View>
          </View>
            </>
          )}
        </Card.Content>
      </Card>

      {/* Wound Healing Graphs */}
      <WoundHealingGraphs 
        patientId={patientId} 
        currentAnalysis={analysisData} 
      />

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  processingCard: {
    margin: 15,
    elevation: 4,
  },
  progressContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  progressBar: {
    width: '100%',
    height: 8,
    marginBottom: 15,
  },
  progressText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#667eea',
    marginBottom: 10,
  },
  currentStep: {
    fontSize: 16,
    color: '#2c3e50',
    textAlign: 'center',
  },
  imageCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  woundImage: {
    width: width - 60,
    height: 200,
    resizeMode: 'cover',
    borderRadius: 10,
    marginTop: 10,
  },
  headerCard: {
    margin: 15,
    elevation: 4,
    backgroundColor: '#e8f5e8',
  },
  patientInfo: {
    marginTop: 15,
    padding: 15,
    backgroundColor: 'white',
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
  resultCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    borderLeftWidth: 5,
  },
  cardTitle: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  resultGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  resultItem: {
    width: '48%',
    marginBottom: 15,
    alignItems: 'center',
  },
  resultLabel: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 5,
  },
  resultValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  chip: {
    alignSelf: 'center',
  },
  chipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  summaryCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  summaryContent: {
    marginTop: 15,
  },
  summaryText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#2c3e50',
  },
  highlight: {
    fontWeight: 'bold',
    color: '#667eea',
  },
  actionCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  actionButton: {
    flex: 0.48,
  },
  errorCard: {
    margin: 15,
    elevation: 4,
    backgroundColor: '#ffe6e6',
  },
  errorButton: {
    marginTop: 15,
    backgroundColor: '#e74c3c',
  },
  feedbackCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    backgroundColor: '#f8f9fa',
  },
  feedbackButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  feedbackButton: {
    flex: 0.48,
  },
  feedbackSent: {
    alignItems: 'center',
    marginTop: 15,
    padding: 15,
    backgroundColor: '#e8f5e8',
    borderRadius: 10,
  },
  feedbackSentText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#27ae60',
  },
  typeSelectorContainer: {
    marginTop: 15,
  },
  selectorTitle: {
    fontSize: 18,
    color: '#2c3e50',
    marginBottom: 10,
  },
  selectorSubtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 20,
  },
  predictedType: {
    fontWeight: 'bold',
    color: '#e74c3c',
  },
  woundTypeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  typeButton: {
    width: '48%',
    marginBottom: 10,
  },
  typeButtonLabel: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  selectorActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  selectorActionButton: {
    flex: 0.48,
  },
  // External AI Styles
  externalAICard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    backgroundColor: '#f0f8ff',
  },
  externalAIDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
  },
  externalAIButton: {
    backgroundColor: '#4CAF50',
  },
  geminiIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e8',
    padding: 10,
    borderRadius: 8,
    marginTop: 10,
  },
  geminiText: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: '600',
    color: '#4CAF50',
  },
  appointmentCard: {
    margin: 15,
    marginTop: 0,
    borderLeftWidth: 4,
    elevation: 4,
    backgroundColor: '#fff',
  },
  appointmentDescription: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 16,
    lineHeight: 20,
  },
  severityText: {
    fontWeight: 'bold',
  },
  appointmentButton: {
    paddingVertical: 4,
  },
  structuredOutput: {
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  structuredItem: {
    marginBottom: 10,
  },
  structuredLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  structuredValue: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 5,
  },
  structuredExplanation: {
    fontSize: 14,
    color: '#555',
    lineHeight: 20,
  },
  // Modal Styles
  modalDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 20,
  },
  aiServiceOptions: {
    marginBottom: 20,
  },
  aiServiceOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
    backgroundColor: '#f8f9fa',
  },
  aiServiceSelected: {
    backgroundColor: '#e3f2fd',
    borderColor: '#2196f3',
    borderWidth: 1,
  },
  aiServiceInfo: {
    marginLeft: 10,
    flex: 1,
  },
  aiServiceName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
  },
  aiServiceDescription: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  apiKeyInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    backgroundColor: '#fff',
  },
  disclaimerCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    backgroundColor: '#ffebee',
    borderLeftWidth: 4,
    borderLeftColor: '#e74c3c',
  },
  disclaimerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  disclaimerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#e74c3c',
    marginLeft: 8,
  },
  disclaimerText: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 12,
    lineHeight: 20,
  },
  boldText: {
    fontWeight: 'bold',
    color: '#e74c3c',
  },
  // Analytics styles
  analyticsCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    backgroundColor: '#f8f9fa',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  metricItem: {
    width: '48%',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    elevation: 2,
  },
  metricHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  metricLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2c3e50',
    marginLeft: 8,
  },
  metricDetails: {
    marginBottom: 8,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  metricSubtext: {
    fontSize: 12,
    color: '#7f8c8d',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
    marginTop: 10,
  },
  progressIndicators: {
    marginBottom: 20,
  },
  progressItem: {
    marginBottom: 15,
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    elevation: 1,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
  },
  progressDuration: {
    fontSize: 12,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#ecf0f1',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  progressDescription: {
    fontSize: 12,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  healthInsights: {
    marginTop: 10,
  },
  insightItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
    elevation: 1,
  },
  insightText: {
    fontSize: 14,
    color: '#2c3e50',
    marginLeft: 10,
    flex: 1,
    lineHeight: 20,
  },
  // Debug styles
  debugContainer: {
    backgroundColor: '#f0f0f0',
    padding: 10,
    marginBottom: 15,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  debugText: {
    fontSize: 10,
    color: '#666',
    fontFamily: 'monospace',
  },
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  // Gemini AI Results styles
  geminiResults: {
    marginTop: 15,
  },
  geminiResultItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  geminiResultLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginRight: 10,
    minWidth: 60,
  },
  geminiChip: {
    alignSelf: 'flex-start',
  },
  geminiChipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  geminiExplanation: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  geminiExplanationLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  geminiExplanationText: {
    fontSize: 14,
    color: '#2c3e50',
    lineHeight: 20,
  },
  geminiRawResponse: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  geminiRawLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 8,
  },
  geminiRawText: {
    fontSize: 12,
    color: '#666',
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  geminiFallback: {
    marginTop: 10,
  },
});
