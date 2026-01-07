import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  Image,
  Dimensions,
} from 'react-native';
import { Card, Title, Paragraph, ProgressBar, Button } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { analyzeWound } from '../services/apiService';

const { width } = Dimensions.get('window');

export default function AnalysisScreen({ navigation, route }) {
  const { images, patientInfo } = route.params || {};
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);

  const analysisSteps = [
    { step: 'Uploading image...', duration: 1000 },
    { step: 'Processing image...', duration: 2000 },
    { step: 'Analyzing wound characteristics...', duration: 3000 },
    { step: 'Classifying wound type...', duration: 2000 },
    { step: 'Calculating healing prediction...', duration: 2000 },
    { step: 'Generating recommendations...', duration: 1500 },
    { step: 'Finalizing results...', duration: 1000 },
  ];

  useEffect(() => {
    if (images && images.length && patientInfo) {
      startAnalysis();
    }
  }, []);

  const startAnalysis = async () => {
    if (!images || !images.length || !patientInfo) {
      Alert.alert('Missing data', 'Please select one or more images and enter patient info.');
      navigation.goBack();
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);
    setCurrentStep('Starting analysis...');

    try {
      // Simulate analysis steps with progress
      for (let i = 0; i < analysisSteps.length; i++) {
        const step = analysisSteps[i];
        setCurrentStep(step.step);
        
        // Simulate step progress
        const stepProgress = (i + 1) / analysisSteps.length;
        setProgress(stepProgress);
        
        // Wait for step duration
        await new Promise(resolve => setTimeout(resolve, step.duration));
      }

      // Perform actual analysis
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      const result = await analyzeWound(images, patientInfo);
      
      setAnalysisResult(result);
      setCurrentStep('Analysis complete!');
      
      // Navigate to results after a short delay
      setTimeout(() => {
        const firstImageUri = images[0]?.uri;
        navigation.navigate('AnalysisResults', {
          imageUri: firstImageUri,
          patientInfo,
          analysisResult: result,
        });
      }, 1000);

    } catch (error) {
      console.error('Analysis error:', error);

      // Fallback to a local mock analysis so the user still sees results
      const woundTypes = ['burn', 'cut', 'surgical', 'chronic', 'diabetic'];
      const randomType = woundTypes[Math.floor(Math.random() * woundTypes.length)];
      const randomArea = parseFloat((Math.random() * 10 + 1).toFixed(2));
      const randomHealingTime = Math.floor(Math.random() * 60) + 7;

      const mockAnalysisResult = {
        status: 'offline_demo',
        wound_classification: {
          wound_type: randomType,
          estimated_days_to_cure: randomHealingTime,
          healing_time_category: 'moderate_healing',
          model_available: false,
        },
        area_cm2: randomArea,
        area_pixels: Math.floor(Math.random() * 2000 + 500),
        perimeter: (Math.random() * 200 + 50).toFixed(2),
        model_confidence: parseFloat((Math.random() * 0.3 + 0.7).toFixed(2)),
      };

      Alert.alert(
        'Using Offline Demo Results',
        'The live server could not be reached (network / cold start). Showing demo analysis so you can continue.',
        [
          {
            text: 'OK',
            onPress: () => {
              const firstImageUri = images[0]?.uri;
              navigation.navigate('AnalysisResults', {
                imageUri: firstImageUri,
                patientInfo,
                analysisResult: mockAnalysisResult,
              });
            },
          },
        ]
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const cancelAnalysis = () => {
    Alert.alert(
      'Cancel Analysis',
      'Are you sure you want to cancel the analysis?',
      [
        {
          text: 'Continue Analysis',
          style: 'cancel',
        },
        {
          text: 'Cancel',
          style: 'destructive',
          onPress: () => navigation.goBack(),
        },
      ]
    );
  };

  return (
    <View style={styles.container}>
      <Card style={styles.imageCard}>
        <Card.Content>
          <Title>Wound Image</Title>
          {images && images.length > 0 && images[0]?.uri && (
            <Image source={{ uri: images[0].uri }} style={styles.woundImage} />
          )}
        </Card.Content>
      </Card>

      <Card style={styles.analysisCard}>
        <Card.Content>
          <Title>Analysis Progress</Title>
          
          {isAnalyzing ? (
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
          ) : (
            <View style={styles.completeContainer}>
              <Ionicons name="checkmark-circle" size={60} color="#27ae60" />
              <Text style={styles.completeText}>Analysis Complete!</Text>
              <Text style={styles.completeSubtext}>
                Results are ready for review
              </Text>
            </View>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.infoCard}>
        <Card.Content>
          <Title>Analysis Details</Title>
          <Paragraph>
            Our AI system is analyzing your wound image to determine:
          </Paragraph>
          {isAnalyzing && (
            <Paragraph style={styles.waitTimeNote}>
              ⏱️ Note: Analysis may take up to 5 minutes on free tier servers. Please be patient.
            </Paragraph>
          )}
          <View style={styles.featuresList}>
            <View style={styles.featureItem}>
              <Ionicons name="checkmark" size={20} color="#27ae60" />
              <Text style={styles.featureText}>Wound type and classification</Text>
            </View>
            <View style={styles.featureItem}>
              <Ionicons name="checkmark" size={20} color="#27ae60" />
              <Text style={styles.featureText}>Healing time prediction</Text>
            </View>
            <View style={styles.featureItem}>
              <Ionicons name="checkmark" size={20} color="#27ae60" />
              <Text style={styles.featureText}>Treatment recommendations</Text>
            </View>
            <View style={styles.featureItem}>
              <Ionicons name="checkmark" size={20} color="#27ae60" />
              <Text style={styles.featureText}>Precautions and care instructions</Text>
            </View>
          </View>
        </Card.Content>
      </Card>

      {isAnalyzing && (
        <View style={styles.cancelContainer}>
          <Button
            mode="outlined"
            onPress={cancelAnalysis}
            style={styles.cancelButton}
            icon="close"
          >
            Cancel Analysis
          </Button>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 15,
  },
  imageCard: {
    marginBottom: 15,
    elevation: 4,
  },
  woundImage: {
    width: width - 60,
    height: 200,
    resizeMode: 'cover',
    borderRadius: 10,
    marginTop: 10,
  },
  analysisCard: {
    marginBottom: 15,
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
  completeContainer: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  completeText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#27ae60',
    marginTop: 10,
  },
  completeSubtext: {
    fontSize: 16,
    color: '#7f8c8d',
    marginTop: 5,
  },
  infoCard: {
    marginBottom: 15,
    elevation: 4,
  },
  featuresList: {
    marginTop: 15,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  featureText: {
    marginLeft: 10,
    fontSize: 16,
    color: '#2c3e50',
  },
  waitTimeNote: {
    marginTop: 15,
    fontSize: 14,
    color: '#f39c12',
    fontStyle: 'italic',
    textAlign: 'center',
    backgroundColor: '#fff3cd',
    padding: 10,
    borderRadius: 5,
  },
  cancelContainer: {
    paddingTop: 20,
  },
  cancelButton: {
    borderColor: '#e74c3c',
  },
});
