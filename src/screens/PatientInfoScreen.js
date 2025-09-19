import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  Platform,
} from 'react-native';
import {
  TextInput,
  Button,
  Card,
  Title,
  Paragraph,
  RadioButton,
  Divider,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

export default function PatientInfoScreen({ navigation, route }) {
  const { imageUri } = route.params || {};
  
  const [patientInfo, setPatientInfo] = useState({
    id: '',
    name: '',
    age: '',
    gender: '',
    injuryDate: '',
    injuryType: '',
    notes: '',
  });

  const [isLoading, setIsLoading] = useState(false);

  const injuryTypes = [
    'Burn',
    'Cut/Laceration',
    'Surgical Wound',
    'Chronic Wound',
    'Diabetic Ulcer',
    'Pressure Sore',
    'Other',
  ];

  const handleInputChange = (field, value) => {
    setPatientInfo(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const validateForm = () => {
    if (!patientInfo.id.trim()) {
      Alert.alert('Validation Error', 'Patient ID is required');
      return false;
    }
    if (!patientInfo.name.trim()) {
      Alert.alert('Validation Error', 'Patient name is required');
      return false;
    }
    if (patientInfo.age && (isNaN(patientInfo.age) || patientInfo.age < 0 || patientInfo.age > 150)) {
      Alert.alert('Validation Error', 'Please enter a valid age');
      return false;
    }
    return true;
  };

  const proceedToAnalysis = async () => {
    if (!validateForm()) return;

    setIsLoading(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      // Navigate to analysis screen with patient info and image
      navigation.navigate('Analysis', {
        imageUri,
        patientInfo,
      });
    } catch (error) {
      console.error('Error proceeding to analysis:', error);
      Alert.alert('Error', 'Failed to proceed to analysis');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content>
          <Title>Patient Information</Title>
          <Paragraph>
            Please provide patient details for accurate wound analysis and treatment recommendations.
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.formCard}>
        <Card.Content>
          <TextInput
            label="Patient ID *"
            value={patientInfo.id}
            onChangeText={(text) => handleInputChange('id', text)}
            style={styles.input}
            mode="outlined"
            autoCapitalize="none"
          />

          <TextInput
            label="Patient Name *"
            value={patientInfo.name}
            onChangeText={(text) => handleInputChange('name', text)}
            style={styles.input}
            mode="outlined"
            autoCapitalize="words"
          />

          <TextInput
            label="Age"
            value={patientInfo.age}
            onChangeText={(text) => handleInputChange('age', text)}
            style={styles.input}
            mode="outlined"
            keyboardType="numeric"
            placeholder="Enter age in years"
          />

          <View style={styles.genderContainer}>
            <Text style={styles.genderLabel}>Gender</Text>
            <View style={styles.radioGroup}>
              <View style={styles.radioOption}>
                <RadioButton
                  value="male"
                  status={patientInfo.gender === 'male' ? 'checked' : 'unchecked'}
                  onPress={() => handleInputChange('gender', 'male')}
                />
                <Text style={styles.radioLabel}>Male</Text>
              </View>
              <View style={styles.radioOption}>
                <RadioButton
                  value="female"
                  status={patientInfo.gender === 'female' ? 'checked' : 'unchecked'}
                  onPress={() => handleInputChange('gender', 'female')}
                />
                <Text style={styles.radioLabel}>Female</Text>
              </View>
              <View style={styles.radioOption}>
                <RadioButton
                  value="other"
                  status={patientInfo.gender === 'other' ? 'checked' : 'unchecked'}
                  onPress={() => handleInputChange('gender', 'other')}
                />
                <Text style={styles.radioLabel}>Other</Text>
              </View>
            </View>
          </View>

          <TextInput
            label="Injury Date"
            value={patientInfo.injuryDate}
            onChangeText={(text) => handleInputChange('injuryDate', text)}
            style={styles.input}
            mode="outlined"
            placeholder="YYYY-MM-DD"
          />

          <View style={styles.injuryTypeContainer}>
            <Text style={styles.injuryTypeLabel}>Injury Type</Text>
            {injuryTypes.map((type, index) => (
              <View key={index} style={styles.radioOption}>
                <RadioButton
                  value={type}
                  status={patientInfo.injuryType === type ? 'checked' : 'unchecked'}
                  onPress={() => handleInputChange('injuryType', type)}
                />
                <Text style={styles.radioLabel}>{type}</Text>
              </View>
            ))}
          </View>

          <TextInput
            label="Additional Notes"
            value={patientInfo.notes}
            onChangeText={(text) => handleInputChange('notes', text)}
            style={styles.input}
            mode="outlined"
            multiline
            numberOfLines={3}
            placeholder="Any additional information about the wound or patient..."
          />
        </Card.Content>
      </Card>

      <View style={styles.buttonContainer}>
        <Button
          mode="outlined"
          onPress={() => navigation.goBack()}
          style={styles.backButton}
          icon="arrow-left"
        >
          Back
        </Button>
        
        <Button
          mode="contained"
          onPress={proceedToAnalysis}
          style={styles.continueButton}
          loading={isLoading}
          disabled={isLoading}
          icon="arrow-right"
        >
          Continue to Analysis
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
  formCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  input: {
    marginBottom: 15,
  },
  genderContainer: {
    marginBottom: 20,
  },
  genderLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#2c3e50',
    marginBottom: 10,
  },
  radioGroup: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  radioOption: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  radioLabel: {
    marginLeft: 8,
    fontSize: 16,
    color: '#2c3e50',
  },
  injuryTypeContainer: {
    marginBottom: 20,
  },
  injuryTypeLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#2c3e50',
    marginBottom: 10,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 15,
    paddingTop: 0,
  },
  backButton: {
    flex: 0.4,
    borderColor: '#667eea',
  },
  continueButton: {
    flex: 0.55,
    backgroundColor: '#667eea',
  },
});




