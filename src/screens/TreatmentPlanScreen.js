import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
  List,
  Divider,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { generateTreatmentPlan, generateFollowUpSchedule, storePatientData, getPatientHistory } from '../utils/patientUtils';

export default function TreatmentPlanScreen({ navigation, route }) {
  const { analysisData, imageUri, questionnaireData } = route.params || {};
  
  const [treatmentPlan, setTreatmentPlan] = useState(null);
  const [patientHistory, setPatientHistory] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    if (analysisData) {
      generatePlan();
      loadPatientHistory();
    }
  }, [analysisData]);

  const generatePlan = () => {
    setIsGenerating(true);
    
    // Simulate plan generation
    setTimeout(() => {
      const plan = generateTreatmentPlan({
        woundType: analysisData.woundType,
        area: analysisData.area,
        severity: analysisData.severity,
        patientAge: questionnaireData?.age ? parseInt(questionnaireData.age) : 45,
        patientGender: questionnaireData?.gender || 'Unknown'
      });

      const followUpSchedule = generateFollowUpSchedule(analysisData.woundType, analysisData.severity);
      
      const fullPlan = {
        recommendations: plan,
        followUpSchedule: followUpSchedule.map(day => `Day ${day}`),
        precautions: generatePrecautions(analysisData, questionnaireData),
        riskFactors: generateRiskFactors(analysisData, questionnaireData),
        lifestyleRecommendations: generateLifestyleRecommendations(questionnaireData),
        medicationConsiderations: generateMedicationConsiderations(questionnaireData),
        healingStages: {
          inflammatory: { duration: 3, description: 'Redness, swelling, pain - normal healing response' },
          proliferative: { duration: 14, description: 'New tissue formation and wound contraction' },
          maturation: { duration: 7, description: 'Scar formation and tissue remodeling' }
        }
      };

      setTreatmentPlan(fullPlan);
      setIsGenerating(false);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }, 2000);
  };

  const loadPatientHistory = () => {
    // Load patient history (in a real app, this would come from a database)
    const history = getPatientHistory(analysisData.patientId);
    
    // Add mock history if none exists
    if (history.length === 0) {
      const mockHistory = [
        {
          id: 1,
          date: '2024-01-10',
          description: 'Initial wound assessment - minor cut on left hand',
          outcome: 'Healed successfully',
          woundType: 'cut',
          area: 2.5
        },
        {
          id: 2,
          date: '2024-01-05',
          description: 'Follow-up visit - wound healing progressing well',
          outcome: 'Good progress',
          woundType: 'cut',
          area: 1.8
        }
      ];
      setPatientHistory(mockHistory);
    } else {
      setPatientHistory(history);
    }
  };

  // Generate precautions based on questionnaire data
  const generatePrecautions = (analysis, questionnaire) => {
    const basePrecautions = [
      'Keep the wound clean and dry',
      'Monitor for signs of infection (redness, swelling, pus)',
      'Follow healthcare provider instructions',
      'Take prescribed medications as directed',
      'Avoid activities that may cause trauma to the wound'
    ];

    if (!questionnaire) return basePrecautions;

    const additionalPrecautions = [];

    // Age-specific precautions
    const age = parseInt(questionnaire.age) || 0;
    if (age >= 65) {
      additionalPrecautions.push('Monitor healing progress more frequently due to age');
      additionalPrecautions.push('Ensure adequate nutrition and hydration');
      additionalPrecautions.push('Be extra cautious with mobility to prevent falls');
    } else if (age < 18) {
      additionalPrecautions.push('Ensure proper supervision during wound care');
      additionalPrecautions.push('Monitor for signs of infection more closely');
      additionalPrecautions.push('Keep wound care supplies out of reach of children');
    }

    // Gender-specific precautions
    if (questionnaire.gender === 'Female') {
      additionalPrecautions.push('Consider hormonal factors that may affect healing');
      additionalPrecautions.push('Monitor for any changes during menstrual cycle');
    }

    // Diabetes precautions
    if (questionnaire.diabetes === 'Yes') {
      additionalPrecautions.push('Monitor blood sugar levels closely');
      additionalPrecautions.push('Check feet daily for any changes');
      additionalPrecautions.push('Ensure proper circulation to wound area');
    }

    // Allergy precautions
    if (questionnaire.allergies && questionnaire.allergies.length > 0) {
      additionalPrecautions.push('Inform healthcare providers of all known allergies');
      additionalPrecautions.push('Avoid any products containing known allergens');
    }

    // Medication interactions
    if (questionnaire.medications) {
      additionalPrecautions.push('Discuss current medications with healthcare provider');
      additionalPrecautions.push('Be aware of potential drug interactions');
    }

    // Lifestyle precautions
    if (questionnaire.smoking) {
      additionalPrecautions.push('Avoid smoking as it impairs wound healing');
      additionalPrecautions.push('Consider smoking cessation support');
    }

    if (questionnaire.alcohol) {
      additionalPrecautions.push('Limit alcohol consumption during healing');
    }

    return [...basePrecautions, ...additionalPrecautions];
  };

  // Generate risk factors based on questionnaire data
  const generateRiskFactors = (analysis, questionnaire) => {
    const baseRisks = [
      'Risk of infection increases with wound size',
      'Delayed healing possible with certain conditions',
      'Monitor for complications closely',
      'Follow up appointments are crucial'
    ];

    if (!questionnaire) return baseRisks;

    const additionalRisks = [];

    // Age-specific risks
    const age = parseInt(questionnaire.age) || 0;
    if (age >= 65) {
      additionalRisks.push('Higher risk of delayed healing due to age');
      additionalRisks.push('Increased risk of complications in elderly patients');
      additionalRisks.push('May require more frequent monitoring');
    } else if (age < 18) {
      additionalRisks.push('Children may require specialized wound care');
      additionalRisks.push('Risk of accidental trauma to wound area');
      additionalRisks.push('May need parental supervision for proper care');
    }

    // Gender-specific risks
    if (questionnaire.gender === 'Female') {
      additionalRisks.push('Hormonal changes may affect healing process');
      additionalRisks.push('Pregnancy considerations if applicable');
    }

    // Diabetes risks
    if (questionnaire.diabetes === 'Yes') {
      additionalRisks.push('Higher risk of delayed healing due to diabetes');
      additionalRisks.push('Increased risk of infection');
      additionalRisks.push('Potential for diabetic complications');
    }

    // Other diseases risks
    if (questionnaire.otherDiseases && questionnaire.otherDiseases.length > 0) {
      additionalRisks.push('Underlying medical conditions may affect healing');
      additionalRisks.push('May require specialized care coordination');
    }

    // Lifestyle risks
    if (questionnaire.smoking) {
      additionalRisks.push('Smoking significantly impairs wound healing');
      additionalRisks.push('Higher risk of complications');
    }

    if (questionnaire.exercise === 'Never' || questionnaire.exercise === 'Rarely') {
      additionalRisks.push('Poor circulation may affect healing');
    }

    return [...baseRisks, ...additionalRisks];
  };

  // Generate lifestyle recommendations
  const generateLifestyleRecommendations = (questionnaire) => {
    if (!questionnaire) return [];

    const recommendations = [];

    if (questionnaire.diabetes === 'Yes') {
      recommendations.push('Maintain optimal blood glucose control');
      recommendations.push('Follow diabetic diet recommendations');
      recommendations.push('Regular monitoring of blood sugar levels');
    }

    if (questionnaire.smoking) {
      recommendations.push('Consider smoking cessation programs');
      recommendations.push('Avoid smoking during healing period');
    }

    if (questionnaire.exercise === 'Never' || questionnaire.exercise === 'Rarely') {
      recommendations.push('Gradually increase physical activity');
      recommendations.push('Consult healthcare provider before starting exercise');
    }

    if (questionnaire.alcohol) {
      recommendations.push('Limit alcohol consumption during healing');
    }

    return recommendations;
  };

  // Generate medication considerations
  const generateMedicationConsiderations = (questionnaire) => {
    if (!questionnaire) return [];

    const considerations = [];

    if (questionnaire.medications) {
      considerations.push('Review current medications with healthcare provider');
      considerations.push('Be aware of potential drug interactions');
    }

    if (questionnaire.allergies && questionnaire.allergies.length > 0) {
      considerations.push('Inform all healthcare providers of allergies');
      considerations.push('Carry allergy information card');
    }

    if (questionnaire.otherDiseases && questionnaire.otherDiseases.length > 0) {
      considerations.push('Coordinate care with specialists if needed');
      considerations.push('Monitor for disease-specific complications');
    }

    return considerations;
  };

  const proceedToReports = () => {
    if (!treatmentPlan) return;
    
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    navigation.navigate('Reports', {
      analysisData,
      treatmentPlan,
      imageUri,
      patientHistory,
    });
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

  const getSeverityColor = (severity) => {
    const colors = {
      Mild: '#27ae60',
      Moderate: '#f39c12',
      Severe: '#e74c3c',
    };
    return colors[severity] || colors.Moderate;
  };

  if (isGenerating) {
    return (
      <View style={styles.container}>
        <Card style={styles.generatingCard}>
          <Card.Content>
            <Title>🔄 Generating Treatment Plan</Title>
            <View style={styles.generatingContent}>
              <Ionicons name="medical" size={60} color="#667eea" />
              <Text style={styles.generatingText}>
                Creating personalized treatment recommendations based on wound analysis...
              </Text>
            </View>
          </Card.Content>
        </Card>
      </View>
    );
  }

  if (!treatmentPlan) {
    return (
      <View style={styles.container}>
        <Card style={styles.errorCard}>
          <Card.Content>
            <Title>Error</Title>
            <Paragraph>Unable to generate treatment plan. Please try again.</Paragraph>
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
          <Title>📋 Treatment Plan</Title>
          <Paragraph>
            Personalized treatment recommendations based on wound analysis.
          </Paragraph>
          <View style={styles.woundInfo}>
            <Chip 
              style={[styles.chip, { backgroundColor: getWoundTypeColor(analysisData.woundType) }]}
              textStyle={styles.chipText}
            >
              {analysisData.woundType.toUpperCase()}
            </Chip>
            <Chip 
              style={[styles.chip, { backgroundColor: getSeverityColor(analysisData.severity) }]}
              textStyle={styles.chipText}
            >
              {analysisData.severity.toUpperCase()}
            </Chip>
          </View>
        </Card.Content>
      </Card>

      {/* Important Medical Disclaimer */}
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
            This treatment plan is for informational purposes only and does not replace professional medical advice. 
            Always consult with a qualified healthcare provider before taking any medications or following treatment recommendations.
          </Paragraph>
          <Paragraph style={styles.disclaimerText}>
            <Text style={styles.boldText}>Emergency:</Text> If you experience severe symptoms, seek immediate medical attention.
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.recommendationsCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="medical" size={24} color="#27ae60" />
            Treatment Recommendations
          </Title>
          <Paragraph style={styles.recommendationsNote}>
            <Text style={styles.noteText}>Note: These are general recommendations. Please consult your doctor for personalized treatment.</Text>
          </Paragraph>
          <List.Section>
            {treatmentPlan.recommendations.map((recommendation, index) => (
              <List.Item
                key={index}
                title={recommendation}
                left={() => <List.Icon icon="pill" color="#3498db" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      <Card style={styles.precautionsCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="shield-checkmark" size={24} color="#f39c12" />
            Important Precautions
          </Title>
          <List.Section>
            {treatmentPlan.precautions.map((precaution, index) => (
              <List.Item
                key={index}
                title={precaution}
                left={() => <List.Icon icon="shield-check" color="#f39c12" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      <Card style={styles.scheduleCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="calendar" size={24} color="#9b59b6" />
            Follow-up Schedule
          </Title>
          <List.Section>
            {treatmentPlan.followUpSchedule.map((appointment, index) => (
              <List.Item
                key={index}
                title={appointment}
                left={() => <List.Icon icon="clock" color="#9b59b6" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      <Card style={styles.stagesCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="trending-up" size={24} color="#27ae60" />
            Healing Stages
          </Title>
          {treatmentPlan.healingStages && Object.entries(treatmentPlan.healingStages).map(([stage, info]) => (
            <View key={stage} style={styles.stageItem}>
              <Text style={styles.stageTitle}>{stage.charAt(0).toUpperCase() + stage.slice(1)}</Text>
              <Text style={styles.stageDuration}>{info.duration} days</Text>
              <Text style={styles.stageDescription}>{info.description}</Text>
            </View>
          ))}
        </Card.Content>
      </Card>

      <Card style={styles.riskCard}>
        <Card.Content>
          <Title style={styles.cardTitle}>
            <Ionicons name="warning-outline" size={24} color="#e74c3c" />
            Risk Factors
          </Title>
          <List.Section>
            {treatmentPlan.riskFactors.map((risk, index) => (
              <List.Item
                key={index}
                title={risk}
                left={() => <List.Icon icon="alert-circle" color="#e74c3c" />}
                titleStyle={styles.listItemText}
              />
            ))}
          </List.Section>
        </Card.Content>
      </Card>

      {/* Lifestyle Recommendations */}
      {treatmentPlan.lifestyleRecommendations && treatmentPlan.lifestyleRecommendations.length > 0 && (
        <Card style={styles.lifestyleCard}>
          <Card.Content>
            <Title style={styles.cardTitle}>
              <Ionicons name="heart" size={24} color="#e67e22" />
              Lifestyle Recommendations
            </Title>
            <List.Section>
              {treatmentPlan.lifestyleRecommendations.map((recommendation, index) => (
                <List.Item
                  key={index}
                  title={recommendation}
                  left={() => <List.Icon icon="heart" color="#e67e22" />}
                  titleStyle={styles.listItemText}
                />
              ))}
            </List.Section>
          </Card.Content>
        </Card>
      )}

      {/* Medication Considerations */}
      {treatmentPlan.medicationConsiderations && treatmentPlan.medicationConsiderations.length > 0 && (
        <Card style={styles.medicationCard}>
          <Card.Content>
            <Title style={styles.cardTitle}>
              <Ionicons name="medical" size={24} color="#8e44ad" />
              Medication Considerations
            </Title>
            <List.Section>
              {treatmentPlan.medicationConsiderations.map((consideration, index) => (
                <List.Item
                  key={index}
                  title={consideration}
                  left={() => <List.Icon icon="pill" color="#8e44ad" />}
                  titleStyle={styles.listItemText}
                />
              ))}
            </List.Section>
          </Card.Content>
        </Card>
      )}

      {patientHistory.length > 0 && (
        <Card style={styles.historyCard}>
          <Card.Content>
            <Title style={styles.cardTitle}>
              <Ionicons name="history" size={24} color="#34495e" />
              Patient History
            </Title>
            <Paragraph style={styles.historySubtitle}>
              Previous wound treatments and outcomes
            </Paragraph>
            {patientHistory.map((record, index) => (
              <View key={index} style={styles.historyItem}>
                <Text style={styles.historyDate}>{record.date}</Text>
                <Text style={styles.historyDescription}>{record.description}</Text>
                <Text style={styles.historyOutcome}>Outcome: {record.outcome}</Text>
                <Text style={styles.historyDetails}>
                  Type: {record.woundType} | Area: {record.area} cm²
                </Text>
              </View>
            ))}
          </Card.Content>
        </Card>
      )}

      <Card style={styles.actionCard}>
        <Card.Content>
          <Title>Next Steps</Title>
          <Paragraph>
            Review the treatment plan and generate comprehensive reports for 
            patient and clinician use.
          </Paragraph>
          
          <View style={styles.actionButtons}>
            <Button
              mode="outlined"
              onPress={() => navigation.goBack()}
              style={styles.actionButton}
              icon="arrow-left"
            >
              Back to Analysis
            </Button>
            
            <Button
              mode="contained"
              onPress={proceedToReports}
              style={[styles.actionButton, { backgroundColor: '#667eea' }]}
              icon="file-document"
            >
              Generate Reports
            </Button>
          </View>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  generatingCard: {
    margin: 15,
    elevation: 4,
  },
  generatingContent: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  generatingText: {
    fontSize: 16,
    color: '#2c3e50',
    textAlign: 'center',
    marginTop: 20,
    lineHeight: 24,
  },
  headerCard: {
    margin: 15,
    elevation: 4,
  },
  woundInfo: {
    flexDirection: 'row',
    marginTop: 15,
    gap: 10,
  },
  chip: {
    alignSelf: 'flex-start',
  },
  chipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  recommendationsCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  precautionsCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  scheduleCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  stagesCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  riskCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  historyCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  cardTitle: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
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
  historySubtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 15,
  },
  historyItem: {
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  historyDate: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#667eea',
  },
  historyDescription: {
    fontSize: 16,
    color: '#2c3e50',
    marginTop: 5,
  },
  historyOutcome: {
    fontSize: 14,
    color: '#27ae60',
    marginTop: 5,
    fontStyle: 'italic',
  },
  historyDetails: {
    fontSize: 12,
    color: '#7f8c8d',
    marginTop: 5,
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
  lifestyleCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
    backgroundColor: '#fff8e6',
  },
  medicationCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
    backgroundColor: '#f0e6ff',
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
  recommendationsNote: {
    marginBottom: 15,
    padding: 10,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#3498db',
  },
  noteText: {
    fontSize: 14,
    color: '#2c3e50',
    fontStyle: 'italic',
  },
});
