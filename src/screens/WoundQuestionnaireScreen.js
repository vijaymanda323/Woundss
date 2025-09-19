import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Checkbox,
  TextInput,
  RadioButton,
  Divider,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

export default function WoundQuestionnaireScreen({ navigation, route }) {
  const { analysisData } = route.params || {};
  
  const [currentStep, setCurrentStep] = useState(0);
  const [questionnaireData, setQuestionnaireData] = useState({
    age: '',
    gender: '',
    woundFormation: '',
    allergies: [],
    diabetes: false,
    diabetesType: '',
    otherDiseases: [],
    medications: '',
    smoking: false,
    alcohol: false,
    exercise: '',
    diet: '',
    stress: '',
  });

  const questionnaireSteps = [
    {
      id: 'age',
      title: 'What is your age?',
      type: 'text',
      placeholder: 'Enter your age in years',
      required: true,
      inputType: 'numeric',
    },
    {
      id: 'gender',
      title: 'What is your gender?',
      type: 'radio',
      options: ['Male', 'Female', 'Other', 'Prefer not to say'],
      required: true,
    },
    {
      id: 'woundFormation',
      title: 'How was the wound formed?',
      type: 'radio',
      options: [
        'Accidental cut or injury',
        'Surgical procedure',
        'Burn (heat, chemical, electrical)',
        'Pressure or friction',
        'Insect bite or sting',
        'Animal bite',
        'Sports injury',
        'Other',
      ],
      required: true,
    },
    {
      id: 'allergies',
      title: 'Do you have any known allergies?',
      type: 'checkbox',
      options: [
        'Medications (antibiotics, painkillers)',
        'Latex or rubber',
        'Adhesive tapes or bandages',
        'Topical creams or ointments',
        'Food allergies',
        'Environmental allergies',
        'No known allergies',
      ],
      required: true,
    },
    {
      id: 'diabetes',
      title: 'Do you have diabetes?',
      type: 'radio',
      options: ['Yes', 'No'],
      required: true,
    },
    {
      id: 'diabetesType',
      title: 'What type of diabetes do you have?',
      type: 'radio',
      options: ['Type 1', 'Type 2', 'Gestational', 'Pre-diabetes'],
      required: false,
      showIf: 'diabetes',
      showValue: 'Yes',
    },
    {
      id: 'otherDiseases',
      title: 'Do you have any other medical conditions?',
      type: 'checkbox',
      options: [
        'Heart disease',
        'High blood pressure',
        'Kidney disease',
        'Liver disease',
        'Autoimmune disorders',
        'Cancer (current or history)',
        'Blood clotting disorders',
        'Skin conditions',
        'None of the above',
      ],
      required: true,
    },
    {
      id: 'medications',
      title: 'Are you currently taking any medications?',
      type: 'text',
      placeholder: 'Please list any medications you are currently taking...',
      required: false,
    },
    {
      id: 'lifestyle',
      title: 'Lifestyle Information',
      type: 'lifestyle',
      required: false,
    },
  ];

  const handleAnswer = (questionId, answer) => {
    setQuestionnaireData(prev => ({
      ...prev,
      [questionId]: answer,
    }));
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleCheckboxChange = (questionId, option) => {
    setQuestionnaireData(prev => {
      const currentAnswers = prev[questionId] || [];
      const newAnswers = currentAnswers.includes(option)
        ? currentAnswers.filter(item => item !== option)
        : [...currentAnswers, option];
      
      return {
        ...prev,
        [questionId]: newAnswers,
      };
    });
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleLifestyleChange = (field, value) => {
    setQuestionnaireData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const getFilteredSteps = () => {
    return questionnaireSteps.filter(step => {
      // If step has conditional logic, check if it should be shown
      if (step.showIf && step.showValue) {
        const dependentAnswer = questionnaireData[step.showIf];
        return dependentAnswer === step.showValue;
      }
      return true;
    });
  };

  const nextStep = () => {
    const filteredSteps = getFilteredSteps();
    const currentQuestion = filteredSteps[currentStep];
    
    // Check if current step is required and answered
    if (currentQuestion.required) {
      const answer = questionnaireData[currentQuestion.id];
      if (!answer || (Array.isArray(answer) && answer.length === 0)) {
        Alert.alert('Required Field', 'Please answer this question before proceeding.');
        return;
      }
    }

    // Move to next step
    if (currentStep < filteredSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      completeQuestionnaire();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const completeQuestionnaire = () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    
    // Navigate to treatment plan with questionnaire data
    navigation.navigate('TreatmentPlan', {
      analysisData,
      questionnaireData,
    });
  };

  const renderQuestion = (question) => {
    const currentAnswer = questionnaireData[question.id];

    switch (question.type) {
      case 'radio':
        return (
          <View style={styles.optionsContainer}>
            {question.options.map((option, index) => (
              <TouchableOpacity
                key={index}
                style={styles.radioOption}
                onPress={() => handleAnswer(question.id, option)}
              >
                <RadioButton
                  value={option}
                  status={currentAnswer === option ? 'checked' : 'unchecked'}
                  onPress={() => handleAnswer(question.id, option)}
                />
                <Text style={styles.optionText}>{option}</Text>
              </TouchableOpacity>
            ))}
          </View>
        );

      case 'checkbox':
        return (
          <View style={styles.optionsContainer}>
            {question.options.map((option, index) => (
              <TouchableOpacity
                key={index}
                style={styles.checkboxOption}
                onPress={() => handleCheckboxChange(question.id, option)}
              >
                <Checkbox
                  status={currentAnswer?.includes(option) ? 'checked' : 'unchecked'}
                  onPress={() => handleCheckboxChange(question.id, option)}
                />
                <Text style={styles.optionText}>{option}</Text>
              </TouchableOpacity>
            ))}
          </View>
        );

      case 'text':
        return (
          <TextInput
            style={styles.textInput}
            placeholder={question.placeholder}
            value={currentAnswer || ''}
            onChangeText={(text) => handleAnswer(question.id, text)}
            multiline={question.id !== 'age'}
            numberOfLines={question.id === 'age' ? 1 : 4}
            mode="outlined"
            keyboardType={question.inputType === 'numeric' ? 'numeric' : 'default'}
          />
        );

      case 'lifestyle':
        return (
          <View style={styles.lifestyleContainer}>
            <View style={styles.lifestyleSection}>
              <Text style={styles.lifestyleLabel}>Do you smoke?</Text>
              <View style={styles.radioRow}>
                <TouchableOpacity
                  style={styles.radioOption}
                  onPress={() => handleLifestyleChange('smoking', true)}
                >
                  <RadioButton
                    value="yes"
                    status={questionnaireData.smoking ? 'checked' : 'unchecked'}
                    onPress={() => handleLifestyleChange('smoking', true)}
                  />
                  <Text style={styles.optionText}>Yes</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.radioOption}
                  onPress={() => handleLifestyleChange('smoking', false)}
                >
                  <RadioButton
                    value="no"
                    status={!questionnaireData.smoking ? 'checked' : 'unchecked'}
                    onPress={() => handleLifestyleChange('smoking', false)}
                  />
                  <Text style={styles.optionText}>No</Text>
                </TouchableOpacity>
              </View>
            </View>

            <View style={styles.lifestyleSection}>
              <Text style={styles.lifestyleLabel}>Do you consume alcohol?</Text>
              <View style={styles.radioRow}>
                <TouchableOpacity
                  style={styles.radioOption}
                  onPress={() => handleLifestyleChange('alcohol', true)}
                >
                  <RadioButton
                    value="yes"
                    status={questionnaireData.alcohol ? 'checked' : 'unchecked'}
                    onPress={() => handleLifestyleChange('alcohol', true)}
                  />
                  <Text style={styles.optionText}>Yes</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.radioOption}
                  onPress={() => handleLifestyleChange('alcohol', false)}
                >
                  <RadioButton
                    value="no"
                    status={!questionnaireData.alcohol ? 'checked' : 'unchecked'}
                    onPress={() => handleLifestyleChange('alcohol', false)}
                  />
                  <Text style={styles.optionText}>No</Text>
                </TouchableOpacity>
              </View>
            </View>

            <View style={styles.lifestyleSection}>
              <Text style={styles.lifestyleLabel}>Exercise frequency:</Text>
              <View style={styles.optionsContainer}>
                {['Daily', '3-4 times per week', '1-2 times per week', 'Rarely', 'Never'].map((option, index) => (
                  <TouchableOpacity
                    key={index}
                    style={styles.radioOption}
                    onPress={() => handleLifestyleChange('exercise', option)}
                  >
                    <RadioButton
                      value={option}
                      status={questionnaireData.exercise === option ? 'checked' : 'unchecked'}
                      onPress={() => handleLifestyleChange('exercise', option)}
                    />
                    <Text style={styles.optionText}>{option}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </View>
        );

      default:
        return null;
    }
  };

  const filteredSteps = getFilteredSteps();
  const currentQuestion = filteredSteps[currentStep];
  const progress = ((currentStep + 1) / filteredSteps.length) * 100;

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Title style={styles.headerTitle}>Medical Questionnaire</Title>
        <Paragraph style={styles.headerSubtitle}>
          Help us create a personalized treatment plan
        </Paragraph>
        
        {/* Progress Bar */}
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${progress}%` }]} />
          </View>
          <Text style={styles.progressText}>
            Step {currentStep + 1} of {filteredSteps.length}
          </Text>
        </View>
      </View>

      {/* Question Card */}
      <Card style={styles.questionCard}>
        <Card.Content>
          <Title style={styles.questionTitle}>
            {currentQuestion.title}
          </Title>
          
          {renderQuestion(currentQuestion)}
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
            This questionnaire and treatment plan are for informational purposes only. 
            They do not replace professional medical advice, diagnosis, or treatment.
          </Paragraph>
          <Paragraph style={styles.disclaimerText}>
            <Text style={styles.boldText}>Always consult with a qualified healthcare provider</Text> before taking any medications or following treatment recommendations.
          </Paragraph>
        </Card.Content>
      </Card>

      {/* Caution Box for Severe Conditions */}
      {analysisData?.severity === 'Severe' && (
        <Card style={styles.cautionCard}>
          <Card.Content>
            <View style={styles.cautionHeader}>
              <Ionicons name="warning" size={24} color="#e74c3c" />
              <Title style={styles.cautionTitle}>Important Caution</Title>
            </View>
            <Paragraph style={styles.cautionText}>
              Your wound has been classified as <Text style={styles.severityText}>SEVERE</Text>. 
              Please visit a doctor immediately for proper medical attention. 
              This questionnaire is for informational purposes only and does not replace professional medical care.
            </Paragraph>
            <Button
              mode="contained"
              onPress={() => navigation.navigate('DoctorAppointment', { 
                severity: analysisData.severity, 
                woundType: analysisData.woundType 
              })}
              style={styles.emergencyButton}
              icon="medical"
            >
              Book Emergency Appointment
            </Button>
          </Card.Content>
        </Card>
      )}

      {/* Navigation Buttons */}
      <Card style={styles.navigationCard}>
        <Card.Content>
          <View style={styles.navigationButtons}>
            {currentStep > 0 && (
              <Button
                mode="outlined"
                onPress={prevStep}
                style={styles.navButton}
                icon="arrow-left"
              >
                Previous
              </Button>
            )}
            
            <Button
              mode="contained"
              onPress={nextStep}
              style={[styles.navButton, styles.nextButton]}
              icon="arrow-right"
            >
              {currentStep === filteredSteps.length - 1 ? 'Complete' : 'Next'}
            </Button>
          </View>
        </Card.Content>
      </Card>

      {/* Skip Option */}
      <Card style={styles.skipCard}>
        <Card.Content>
          <Button
            mode="text"
            onPress={() => navigation.navigate('TreatmentPlan', { 
              analysisData, 
              questionnaireData: null 
            })}
            style={styles.skipButton}
            textColor="#7f8c8d"
          >
            Skip Questionnaire
          </Button>
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
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 20,
  },
  progressContainer: {
    alignItems: 'center',
  },
  progressBar: {
    width: '100%',
    height: 8,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#667eea',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 14,
    color: '#7f8c8d',
  },
  questionCard: {
    margin: 15,
    elevation: 2,
  },
  questionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 20,
  },
  optionsContainer: {
    marginTop: 10,
  },
  radioOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  checkboxOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  optionText: {
    fontSize: 16,
    color: '#2c3e50',
    marginLeft: 8,
  },
  textInput: {
    marginTop: 10,
  },
  lifestyleContainer: {
    marginTop: 10,
  },
  lifestyleSection: {
    marginBottom: 20,
  },
  lifestyleLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 10,
  },
  radioRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  cautionCard: {
    margin: 15,
    marginTop: 0,
    borderLeftWidth: 4,
    borderLeftColor: '#e74c3c',
    elevation: 4,
    backgroundColor: '#fff5f5',
  },
  cautionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  cautionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#e74c3c',
    marginLeft: 8,
  },
  cautionText: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 16,
    lineHeight: 20,
  },
  severityText: {
    fontWeight: 'bold',
    color: '#e74c3c',
  },
  emergencyButton: {
    backgroundColor: '#e74c3c',
    paddingVertical: 4,
  },
  navigationCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
  },
  navigationButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  navButton: {
    flex: 1,
    marginHorizontal: 5,
  },
  nextButton: {
    backgroundColor: '#667eea',
  },
  skipCard: {
    margin: 15,
    marginTop: 0,
    elevation: 1,
  },
  skipButton: {
    alignSelf: 'center',
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
});
