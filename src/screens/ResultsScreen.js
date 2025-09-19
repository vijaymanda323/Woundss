import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Dimensions,
  Alert,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
  Divider,
  List,
  IconButton,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import * as Sharing from 'expo-sharing';
import { generateReport } from '../services/apiService';

const { width } = Dimensions.get('window');

export default function ResultsScreen({ navigation, route }) {
  const { imageUri, patientInfo, analysisResult } = route.params || {};
  
  const [activeTab, setActiveTab] = useState('analysis');
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // Provide default values to prevent undefined errors
  const safeAnalysisResult = analysisResult || {};
  const enhancedAnalysis = safeAnalysisResult.enhanced_analysis || {};
  const woundClassification = safeAnalysisResult.wound_classification || {};

  // If no data is provided, redirect to PhotoUpload
  React.useEffect(() => {
    if (!imageUri && !analysisResult) {
      navigation.replace('PhotoUpload');
    }
  }, [imageUri, analysisResult, navigation]);

  const getWoundTypeColor = (type) => {
    const colors = {
      burn: '#e74c3c',
      cut: '#3498db',
      surgical: '#9b59b6',
      chronic: '#f39c12',
      diabetic: '#e67e22',
      unknown: '#95a5a6',
    };
    return colors[type?.toLowerCase()] || colors.unknown;
  };

  const getHealingTimeColor = (days) => {
    if (days <= 7) return '#27ae60';
    if (days <= 21) return '#f39c12';
    if (days <= 60) return '#e67e22';
    return '#e74c3c';
  };

  const handleGenerateReport = async (reportType) => {
    setIsGeneratingReport(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      const reportData = await generateReport(safeAnalysisResult, reportType);
      
      if (reportData.success) {
        // For mobile, we'll show the report content in an alert
        // In a real app, you might want to save it to files or share it
        Alert.alert(
          `${reportType.charAt(0).toUpperCase() + reportType.slice(1)} Report Generated`,
          'Report has been generated successfully. In a full implementation, this would be saved or shared.',
          [
            {
              text: 'OK',
              onPress: () => {
                // Here you could implement file saving or sharing
                console.log('Report content:', reportData.report_content);
              },
            },
          ]
        );
      }
    } catch (error) {
      console.error('Report generation error:', error);
      Alert.alert('Error', 'Failed to generate report. Please try again.');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const renderAnalysisTab = () => (
    <View>
      <Card style={[styles.resultCard, { borderLeftColor: getWoundTypeColor(enhancedAnalysis.wound_type) }]}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="bandage" size={24} color={getWoundTypeColor(enhancedAnalysis.wound_type)} />
            Wound Type
          </Title>
          <Chip 
            style={[styles.chip, { backgroundColor: getWoundTypeColor(enhancedAnalysis.wound_type) }]}
            textStyle={styles.chipText}
          >
            {enhancedAnalysis.wound_type?.toUpperCase() || 'UNKNOWN'}
          </Chip>
        </Card.Content>
      </Card>

      <Card style={[styles.resultCard, { borderLeftColor: getHealingTimeColor(enhancedAnalysis.estimated_healing_time) }]}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="time" size={24} color={getHealingTimeColor(enhancedAnalysis.estimated_healing_time)} />
            Healing Time
          </Title>
          <Chip 
            style={[styles.chip, { backgroundColor: getHealingTimeColor(enhancedAnalysis.estimated_healing_time) }]}
            textStyle={styles.chipText}
          >
            {enhancedAnalysis.estimated_healing_time || 'Unknown'} DAYS
          </Chip>
        </Card.Content>
      </Card>

      <Card style={styles.resultCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="information-circle" size={24} color="#3498db" />
            Additional Information
          </Title>
          <View style={styles.infoGrid}>
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Age Group</Text>
              <Text style={styles.infoValue}>{enhancedAnalysis.age_group || 'Unknown'}</Text>
            </View>
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Size Category</Text>
              <Text style={styles.infoValue}>{enhancedAnalysis.size_category || 'Unknown'}</Text>
            </View>
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Area</Text>
              <Text style={styles.infoValue}>{safeAnalysisResult.area_cm2 ? `${safeAnalysisResult.area_cm2} cm²` : 'Not calculated'}</Text>
            </View>
            <View style={styles.infoItem}>
              <Text style={styles.infoLabel}>Confidence</Text>
              <Text style={styles.infoValue}>{safeAnalysisResult.model_confidence ? `${(safeAnalysisResult.model_confidence * 100).toFixed(1)}%` : 'Not available'}</Text>
            </View>
          </View>
        </Card.Content>
      </Card>
    </View>
  );

  const renderRecommendationsTab = () => (
    <View>
      <Card style={styles.resultCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="warning" size={24} color="#f39c12" />
            Precautions
          </Title>
          <List.Section>
            {enhancedAnalysis.precautions?.map((precaution, index) => (
              <List.Item
                key={index}
                title={precaution}
                left={() => <List.Icon icon="check" color="#27ae60" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      <Card style={styles.resultCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="medical" size={24} color="#27ae60" />
            Treatment Recommendations
          </Title>
          <List.Section>
            {enhancedAnalysis.treatment_recommendations?.map((treatment, index) => (
              <List.Item
                key={index}
                title={treatment}
                left={() => <List.Icon icon="pill" color="#3498db" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      <Card style={styles.resultCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="alert-circle" size={24} color="#e74c3c" />
            Risk Factors
          </Title>
          <List.Section>
            {enhancedAnalysis.risk_factors?.map((risk, index) => (
              <List.Item
                key={index}
                title={risk}
                left={() => <List.Icon icon="alert" color="#e74c3c" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>
    </View>
  );

  const renderProgressTab = () => (
    <View>
      <Card style={styles.resultCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="calendar" size={24} color="#9b59b6" />
            Follow-up Schedule
          </Title>
          <List.Section>
            {enhancedAnalysis.follow_up_schedule?.map((followUp, index) => (
              <List.Item
                key={index}
                title={followUp}
                left={() => <List.Icon icon="clock" color="#9b59b6" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      <Card style={styles.resultCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="trending-up" size={24} color="#27ae60" />
            Healing Stages
          </Title>
          {enhancedAnalysis.healing_stages && Object.entries(enhancedAnalysis.healing_stages).map(([stage, info]) => (
            <View key={stage} style={styles.stageItem}>
              <Text style={styles.stageTitle}>{stage.charAt(0).toUpperCase() + stage.slice(1)}</Text>
              <Text style={styles.stageDuration}>{info.duration} days</Text>
              <Text style={styles.stageDescription}>{info.description}</Text>
            </View>
          ))}
        </Card.Content>
      </Card>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.imageCard}>
        <Card.Content>
          <Title>Wound Image</Title>
          {imageUri && (
            <Image source={{ uri: imageUri }} style={styles.woundImage} />
          )}
        </Card.Content>
      </Card>

      <View style={styles.tabContainer}>
        <Button
          mode={activeTab === 'analysis' ? 'contained' : 'outlined'}
          onPress={() => setActiveTab('analysis')}
          style={styles.tabButton}
        >
          Analysis
        </Button>
        <Button
          mode={activeTab === 'recommendations' ? 'contained' : 'outlined'}
          onPress={() => setActiveTab('recommendations')}
          style={styles.tabButton}
        >
          Recommendations
        </Button>
        <Button
          mode={activeTab === 'progress' ? 'contained' : 'outlined'}
          onPress={() => setActiveTab('progress')}
          style={styles.tabButton}
        >
          Progress
        </Button>
      </View>

      {activeTab === 'analysis' && renderAnalysisTab()}
      {activeTab === 'recommendations' && renderRecommendationsTab()}
      {activeTab === 'progress' && renderProgressTab()}

      <Card style={styles.reportCard}>
        <Card.Content>
          <Title>Generate Reports</Title>
          <Paragraph>Download detailed reports for patient and clinician use</Paragraph>
          
          <View style={styles.reportButtons}>
            <Button
              mode="contained"
              onPress={() => handleGenerateReport('patient')}
              style={[styles.reportButton, { backgroundColor: '#27ae60' }]}
              loading={isGeneratingReport}
              disabled={isGeneratingReport}
              icon="account"
            >
              Patient Report
            </Button>
            
            <Button
              mode="contained"
              onPress={() => handleGenerateReport('clinician')}
              style={[styles.reportButton, { backgroundColor: '#9b59b6' }]}
              loading={isGeneratingReport}
              disabled={isGeneratingReport}
              icon="doctor"
            >
              Clinician Report
            </Button>
          </View>
        </Card.Content>
      </Card>

      <View style={styles.actionButtons}>
        <Button
          mode="outlined"
          onPress={() => navigation.navigate('History', { patientId: patientInfo.id })}
          style={styles.actionButton}
          icon="history"
        >
          View History
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
  imageCard: {
    margin: 15,
    elevation: 4,
  },
  woundImage: {
    width: width - 60,
    height: 200,
    resizeMode: 'cover',
    borderRadius: 10,
    marginTop: 10,
  },
  tabContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 15,
    marginBottom: 15,
  },
  tabButton: {
    flex: 1,
    marginHorizontal: 5,
  },
  resultCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    borderLeftWidth: 5,
    borderLeftColor: '#667eea',
  },
  cardTitle: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  chip: {
    alignSelf: 'flex-start',
  },
  chipText: {
    color: 'white',
    fontWeight: 'bold',
  },
  infoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  infoItem: {
    width: '48%',
    marginBottom: 15,
  },
  infoLabel: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 5,
  },
  infoValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  listItemText: {
    fontSize: 16,
    lineHeight: 22,
  },
  stageItem: {
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  stageTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  stageDuration: {
    fontSize: 16,
    color: '#667eea',
    fontWeight: '600',
    marginTop: 5,
  },
  stageDescription: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  reportCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  reportButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  reportButton: {
    flex: 0.48,
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
});
