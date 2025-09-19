import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Dimensions,
  TextInput,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
  List,
  Divider,
  TextInput as RNTextInput,
  Portal,
  Dialog,
  RadioButton,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { getPatientHistory } from '../services/apiService';

const { width } = Dimensions.get('window');

export default function DoctorAppointmentScreen({ navigation, route }) {
  const { patientInfo, analysisData, treatmentPlan } = route.params || {};
  
  const [selectedDoctorType, setSelectedDoctorType] = useState(null);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [patientDetails, setPatientDetails] = useState(null);
  const [showPatientIdDialog, setShowPatientIdDialog] = useState(false);
  const [isLoadingPatient, setIsLoadingPatient] = useState(false);

  const doctorTypes = [
    {
      id: 'dermatologist',
      name: 'Dermatologist',
      description: 'Skin and wound care specialist',
      icon: 'medical',
      color: '#3498db',
    },
    {
      id: 'general',
      name: 'General Practitioner',
      description: 'Primary care physician',
      icon: 'person',
      color: '#2ecc71',
    },
    {
      id: 'surgeon',
      name: 'Surgeon',
      description: 'Surgical wound specialist',
      icon: 'cut',
      color: '#e74c3c',
    },
    {
      id: 'emergency',
      name: 'Emergency Medicine',
      description: 'Emergency wound care',
      icon: 'medical',
      color: '#f39c12',
    },
    {
      id: 'plastic',
      name: 'Plastic Surgeon',
      description: 'Cosmetic wound repair',
      icon: 'star',
      color: '#9b59b6',
    },
  ];

  const doctors = {
    dermatologist: [
      {
        id: 1,
        name: 'Dr. Sarah Johnson',
        specialty: 'Dermatology & Wound Care',
        location: 'City Medical Center, Downtown',
        address: '123 Main Street, Downtown',
        phone: '+1 (555) 123-4567',
        videoCallId: 'dr-sarah-johnson',
        rating: 4.9,
        experience: '15 years',
        availability: ['Monday', 'Wednesday', 'Friday'],
        times: ['9:00 AM', '10:30 AM', '2:00 PM', '3:30 PM'],
        image: '👩‍⚕️',
        consultationFee: '$150',
      },
      {
        id: 2,
        name: 'Dr. Michael Chen',
        specialty: 'Dermatology & Skin Care',
        location: 'Regional Hospital, Midtown',
        address: '456 Oak Avenue, Midtown',
        phone: '+1 (555) 234-5678',
        videoCallId: 'dr-michael-chen',
        rating: 4.8,
        experience: '12 years',
        availability: ['Tuesday', 'Thursday', 'Saturday'],
        times: ['8:00 AM', '11:00 AM', '1:00 PM', '4:00 PM'],
        image: '👨‍⚕️',
        consultationFee: '$140',
      },
    ],
    general: [
      {
        id: 3,
        name: 'Dr. Emily Rodriguez',
        specialty: 'General Practice & Wound Management',
        location: 'University Medical Center, Uptown',
        address: '789 Pine Street, Uptown',
        phone: '+1 (555) 345-6789',
        videoCallId: 'dr-emily-rodriguez',
        rating: 4.9,
        experience: '18 years',
        availability: ['Monday', 'Tuesday', 'Thursday', 'Friday'],
        times: ['9:30 AM', '11:30 AM', '2:30 PM', '4:30 PM'],
        image: '👩‍⚕️',
        consultationFee: '$120',
      },
      {
        id: 4,
        name: 'Dr. James Wilson',
        specialty: 'Family Medicine',
        location: 'Community Health Center',
        address: '321 Elm Street, Suburb',
        phone: '+1 (555) 456-7890',
        videoCallId: 'dr-james-wilson',
        rating: 4.7,
        experience: '10 years',
        availability: ['Monday', 'Wednesday', 'Friday'],
        times: ['8:30 AM', '10:00 AM', '1:30 PM', '3:00 PM'],
        image: '👨‍⚕️',
        consultationFee: '$100',
      },
    ],
    surgeon: [
      {
        id: 5,
        name: 'Dr. Robert Kim',
        specialty: 'General Surgery & Wound Care',
        location: 'Surgical Center, Downtown',
        address: '654 Surgery Lane, Downtown',
        phone: '+1 (555) 567-8901',
        videoCallId: 'dr-robert-kim',
        rating: 4.9,
        experience: '20 years',
        availability: ['Tuesday', 'Thursday', 'Saturday'],
        times: ['9:00 AM', '11:00 AM', '2:00 PM', '4:00 PM'],
        image: '👨‍⚕️',
        consultationFee: '$200',
      },
    ],
    emergency: [
      {
        id: 6,
        name: 'Dr. Lisa Thompson',
        specialty: 'Emergency Medicine & Trauma',
        location: 'Emergency Medical Center',
        address: '987 Emergency Blvd, City Center',
        phone: '+1 (555) 678-9012',
        videoCallId: 'dr-lisa-thompson',
        rating: 4.8,
        experience: '14 years',
        availability: ['24/7 Emergency'],
        times: ['Immediate', 'Within 1 hour', 'Within 2 hours'],
        image: '👩‍⚕️',
        consultationFee: '$180',
      },
    ],
    plastic: [
      {
        id: 7,
        name: 'Dr. Amanda Davis',
        specialty: 'Plastic Surgery & Cosmetic Repair',
        location: 'Plastic Surgery Institute',
        address: '147 Beauty Street, Uptown',
        phone: '+1 (555) 789-0123',
        videoCallId: 'dr-amanda-davis',
        rating: 4.9,
        experience: '16 years',
        availability: ['Monday', 'Wednesday', 'Friday'],
        times: ['10:00 AM', '12:00 PM', '3:00 PM', '5:00 PM'],
        image: '👩‍⚕️',
        consultationFee: '$250',
      },
    ],
  };

  const availableDates = [
    'Today',
    'Tomorrow',
    'Day After Tomorrow',
    'Next Week',
  ];

  useEffect(() => {
    // Auto-fill patient ID if available from route params
    if (patientInfo?.id) {
      setPatientId(patientInfo.id);
      fetchPatientDetails(patientInfo.id);
    }
  }, [patientInfo]);

  const fetchPatientDetails = async (id) => {
    if (!id) return;
    
    setIsLoadingPatient(true);
    try {
      // First, try to get patient details from the patient details endpoint
      const patientDetailsResponse = await fetch(`http://10.81.160.244:5000/patient/${id}/details`);
      if (patientDetailsResponse.ok) {
        const patientData = await patientDetailsResponse.json();
        const patient = patientData.patient;
        
        // Get the latest analysis record for wound information
        const historyResponse = await fetch(`http://10.81.160.244:5000/patient/${id}/history/days`);
        let latestRecord = null;
        if (historyResponse.ok) {
          const historyData = await historyResponse.json();
          if (historyData.days && historyData.days.length > 0) {
            latestRecord = historyData.days[0].records[0]; // Get first record from most recent day
          }
        }
        
        setPatientDetails({
          id: id,
          name: patient.name || 'Unknown Patient',
          age: patient.age || patient.date_of_birth ? calculateAge(patient.date_of_birth) : 'N/A',
          gender: patient.gender || 'N/A',
          woundType: latestRecord?.wound_type || 'Unknown',
          area: latestRecord?.area_cm2 || 0,
          healingTime: latestRecord?.days_to_heal || 'N/A',
          lastVisit: latestRecord?.timestamp || latestRecord?.created_at || 'N/A',
          notes: latestRecord?.notes || 'No notes available',
        });
        
        console.log('Patient details set:', {
          id: id,
          name: patient.name,
          age: patient.age || patient.date_of_birth ? calculateAge(patient.date_of_birth) : 'N/A',
          gender: patient.gender,
          date_of_birth: patient.date_of_birth
        });
      } else {
        // Fallback: try to get from history records
        const history = await getPatientHistory(id);
        if (history && history.length > 0) {
          const latestRecord = history[0];
          setPatientDetails({
            id: id,
            name: 'Patient Found (Limited Info)',
            age: 'N/A',
            gender: 'N/A',
            woundType: latestRecord.wound_type || 'Unknown',
            area: latestRecord.area_cm2 || 0,
            healingTime: latestRecord.estimated_days_to_cure || 'N/A',
            lastVisit: latestRecord.timestamp || 'N/A',
            notes: latestRecord.notes || 'No notes available',
          });
        } else {
          setPatientDetails({
            id: id,
            name: 'Patient Not Found',
            age: 'N/A',
            gender: 'N/A',
            woundType: 'Unknown',
            area: 0,
            healingTime: 'N/A',
            lastVisit: 'N/A',
            notes: 'No previous records found',
          });
        }
      }
    } catch (error) {
      console.error('Error fetching patient details:', error);
      Alert.alert('Error', 'Failed to fetch patient details');
    } finally {
      setIsLoadingPatient(false);
    }
  };

  // Helper function to calculate age from date of birth
  const calculateAge = (dateOfBirth) => {
    if (!dateOfBirth || dateOfBirth === '' || dateOfBirth === null) return 'N/A';
    try {
      const birthDate = new Date(dateOfBirth);
      // Check if the date is valid
      if (isNaN(birthDate.getTime())) return 'N/A';
      
      const today = new Date();
      let age = today.getFullYear() - birthDate.getFullYear();
      const monthDiff = today.getMonth() - birthDate.getMonth();
      if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
      }
      return age.toString();
    } catch (error) {
      console.log('Age calculation error:', error);
      return 'N/A';
    }
  };

  const handleDoctorTypeSelection = (type) => {
    setSelectedDoctorType(type);
    setSelectedDoctor(null);
    setSelectedDate(null);
    setSelectedTime(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleDoctorSelection = (doctor) => {
    setSelectedDoctor(doctor);
    setSelectedDate(null);
    setSelectedTime(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleDateSelection = (date) => {
    setSelectedDate(date);
    setSelectedTime(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleTimeSelection = (time) => {
    setSelectedTime(time);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleCallDoctor = (doctor) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      'Call Doctor',
      `Calling ${doctor.name} at ${doctor.phone}`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Call', onPress: () => {
          // In a real app, this would initiate a phone phone
          Alert.alert('Call Initiated', `Connecting to ${doctor.phone}...`);
        }},
      ]
    );
  };

  const handleVideoCall = (doctor) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      'Video Call',
      `Starting video consultation with ${doctor.name}`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Start Call', onPress: () => {
          // In a real app, this would initiate a video phone
          Alert.alert('Video Call Started', `Connecting to ${doctor.videoCallId}...`);
        }},
      ]
    );
  };

  const handleBookAppointment = () => {
    if (!selectedDoctorType || !selectedDoctor || !selectedDate || !selectedTime) {
      Alert.alert('Incomplete Selection', 'Please select a doctor type, doctor, date, and time.');
      return;
    }

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    
    Alert.alert(
      'Appointment Booked!',
      `Your appointment with ${selectedDoctor.name} (${selectedDoctorType.name}) is confirmed for ${selectedDate} at ${selectedTime}.\n\nConsultation Fee: ${selectedDoctor.consultationFee}`,
      [
        {
          text: 'OK',
          onPress: () => navigation.goBack(),
        },
      ]
    );
  };

  const handlePatientIdSubmit = () => {
    if (!patientId.trim()) {
      Alert.alert('Invalid Input', 'Please enter a valid patient ID.');
      return;
    }
    setShowPatientIdDialog(false);
    fetchPatientDetails(patientId.trim());
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'severe':
        return '#e74c3c';
      case 'moderate':
        return '#f39c12';
      case 'mild':
        return '#f1c40f';
      default:
        return '#95a5a6';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'severe':
        return 'alert-circle';
      case 'moderate':
        return 'warning';
      case 'mild':
        return 'checkmark-circle';
      default:
        return 'help-circle';
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Title style={styles.headerTitle}>Book Doctor Appointment</Title>
        <Paragraph style={styles.headerSubtitle}>
          Connect with medical professionals for expert wound care
        </Paragraph>
      </View>

      {/* Patient ID Section */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <Title style={styles.sectionTitle}>
            <Ionicons name="person" size={24} color="#3498db" />
            Patient Information
          </Title>
          
          {patientDetails ? (
            <View style={styles.patientDetailsCard}>
              <View style={styles.patientHeader}>
                <Text style={styles.patientName}>{patientDetails.name}</Text>
                <Chip style={styles.patientIdChip}>ID: {patientDetails.id}</Chip>
              </View>
              
              <View style={styles.patientInfo}>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Age:</Text>
                  <Text style={styles.infoValue}>{patientDetails.age}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Gender:</Text>
                  <Text style={styles.infoValue}>{patientDetails.gender}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Wound Type:</Text>
                  <Text style={styles.infoValue}>{patientDetails.woundType}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Area:</Text>
                  <Text style={styles.infoValue}>{patientDetails.area} cm²</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Healing Time:</Text>
                  <Text style={styles.infoValue}>{patientDetails.healingTime} days</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Last Visit:</Text>
                  <Text style={styles.infoValue}>{new Date(patientDetails.lastVisit).toLocaleDateString()}</Text>
                </View>
              </View>
              
              <Button
                mode="outlined"
                onPress={() => setShowPatientIdDialog(true)}
                style={styles.changePatientButton}
                icon="pencil"
              >
                Change Patient ID
              </Button>
            </View>
          ) : (
            <View style={styles.noPatientCard}>
              <Text style={styles.noPatientText}>No patient information available</Text>
              <Button
                mode="contained"
                onPress={() => setShowPatientIdDialog(true)}
                style={styles.addPatientButton}
                icon="plus"
              >
                Add Patient ID
              </Button>
            </View>
          )}
        </Card.Content>
      </Card>

      {/* Severity Alert */}
      {analysisData?.severity && (analysisData.severity === 'moderate' || analysisData.severity === 'severe') && (
        <Card style={[styles.alertCard, { borderLeftColor: getSeverityColor(analysisData.severity) }]}>
          <Card.Content>
            <View style={styles.alertContent}>
              <Ionicons 
                name={getSeverityIcon(analysisData.severity)} 
                size={24} 
                color={getSeverityColor(analysisData.severity)} 
              />
              <View style={styles.alertText}>
                <Text style={[styles.alertTitle, { color: getSeverityColor(analysisData.severity) }]}>
                  {analysisData.severity?.toUpperCase()} Severity Detected
                </Text>
                <Text style={styles.alertDescription}>
                  Your wound requires professional medical attention. We recommend booking an appointment immediately.
                </Text>
                {analysisData.woundType && (
                  <Chip 
                    style={[styles.woundTypeChip, { backgroundColor: getSeverityColor(analysisData.severity) + '20' }]}
                    textStyle={{ color: getSeverityColor(analysisData.severity) }}
                  >
                    {analysisData.woundType.toUpperCase()}
                  </Chip>
                )}
              </View>
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Doctor Type Selection */}
      <Card style={styles.sectionCard}>
        <Card.Content>
          <Title style={styles.sectionTitle}>
            <Ionicons name="medical" size={24} color="#3498db" />
            Select Doctor Type
          </Title>
          <Paragraph style={styles.sectionDescription}>
            Choose the type of medical professional you need
          </Paragraph>
          
          <View style={styles.doctorTypeGrid}>
            {doctorTypes.map((type) => (
              <TouchableOpacity
                key={type.id}
                style={[
                  styles.doctorTypeCard,
                  selectedDoctorType?.id === type.id && styles.selectedDoctorTypeCard,
                  { borderLeftColor: type.color }
                ]}
                onPress={() => handleDoctorTypeSelection(type)}
              >
                <View style={styles.doctorTypeHeader}>
                  <Ionicons name={type.icon} size={24} color={type.color} />
                  <Text style={styles.doctorTypeName}>{type.name}</Text>
                  {selectedDoctorType?.id === type.id && (
                    <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
                  )}
                </View>
                <Text style={styles.doctorTypeDescription}>{type.description}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </Card.Content>
      </Card>

      {/* Doctor Selection */}
      {selectedDoctorType && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>
              <Ionicons name="people" size={24} color="#3498db" />
              Select a Doctor
            </Title>
            <Paragraph style={styles.sectionDescription}>
              Choose from our network of qualified {selectedDoctorType.name.toLowerCase()} specialists
            </Paragraph>
            
            {doctors[selectedDoctorType.id]?.map((doctor) => (
              <TouchableOpacity
                key={doctor.id}
                style={[
                  styles.doctorCard,
                  selectedDoctor?.id === doctor.id && styles.selectedDoctorCard,
                ]}
                onPress={() => handleDoctorSelection(doctor)}
              >
                <View style={styles.doctorHeader}>
                  <Text style={styles.doctorEmoji}>{doctor.image}</Text>
                  <View style={styles.doctorInfo}>
                    <Text style={styles.doctorName}>{doctor.name}</Text>
                    <Text style={styles.doctorSpecialty}>{doctor.specialty}</Text>
                    <View style={styles.doctorRating}>
                      <Ionicons name="star" size={16} color="#f39c12" />
                      <Text style={styles.ratingText}>{doctor.rating}</Text>
                      <Text style={styles.experienceText}>• {doctor.experience}</Text>
                      <Text style={styles.feeText}>• {doctor.consultationFee}</Text>
                    </View>
                  </View>
                  {selectedDoctor?.id === doctor.id && (
                    <Ionicons name="checkmark-circle" size={24} color="#27ae60" />
                  )}
                </View>
                
                <View style={styles.doctorDetails}>
                  <View style={styles.detailRow}>
                    <Ionicons name="location" size={16} color="#7f8c8d" />
                    <Text style={styles.detailText}>{doctor.location}</Text>
                  </View>
                  <View style={styles.detailRow}>
                    <Ionicons name="calendar" size={16} color="#7f8c8d" />
                    <Text style={styles.detailText}>
                      Available: {doctor.availability.join(', ')}
                    </Text>
                  </View>
                </View>

                {/* Call and Video Call Options */}
                <View style={styles.doctorActions}>
                  <TouchableOpacity
                    style={styles.actionButton}
                    onPress={() => handleCallDoctor(doctor)}
                  >
                    <Ionicons name="phone" size={20} color="#27ae60" />
                    <Text style={styles.actionButtonText}>Call</Text>
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={styles.actionButton}
                    onPress={() => handleVideoCall(doctor)}
                  >
                    <Ionicons name="videocam" size={20} color="#3498db" />
                    <Text style={styles.actionButtonText}>Video Call</Text>
                  </TouchableOpacity>
                </View>
              </TouchableOpacity>
            ))}
          </Card.Content>
        </Card>
      )}

      {/* Date Selection */}
      {selectedDoctor && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>
              <Ionicons name="calendar" size={24} color="#3498db" />
              Select Date
            </Title>
            
            <View style={styles.dateGrid}>
              {availableDates.map((date) => (
                <TouchableOpacity
                  key={date}
                  style={[
                    styles.dateButton,
                    selectedDate === date && styles.selectedDateButton,
                  ]}
                  onPress={() => handleDateSelection(date)}
                >
                  <Text
                    style={[
                      styles.dateButtonText,
                      selectedDate === date && styles.selectedDateButtonText,
                    ]}
                  >
                    {date}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Time Selection */}
      {selectedDoctor && selectedDate && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>
              <Ionicons name="time" size={24} color="#3498db" />
              Select Time
            </Title>
            
            <View style={styles.timeGrid}>
              {selectedDoctor.times.map((time) => (
                <TouchableOpacity
                  key={time}
                  style={[
                    styles.timeButton,
                    selectedTime === time && styles.selectedTimeButton,
                  ]}
                  onPress={() => handleTimeSelection(time)}
                >
                  <Text
                    style={[
                      styles.timeButtonText,
                      selectedTime === time && styles.selectedTimeButtonText,
                    ]}
                  >
                    {time}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Book Appointment Button */}
      {selectedDoctorType && selectedDoctor && selectedDate && selectedTime && (
        <Card style={styles.actionCard}>
          <Card.Content>
            <Button
              mode="contained"
              onPress={handleBookAppointment}
              style={styles.bookButton}
              icon="calendar-check"
            >
              Book Appointment - {selectedDoctor.consultationFee}
            </Button>
          </Card.Content>
        </Card>
      )}

      {/* Emergency Contact */}
      <Card style={styles.emergencyCard}>
        <Card.Content>
          <Title style={styles.emergencyTitle}>
            <Ionicons name="phone" size={24} color="#e74c3c" />
            Emergency Contact
          </Title>
          <Paragraph style={styles.emergencyText}>
            If this is a medical emergency, phone 911 immediately or go to your nearest emergency room.
          </Paragraph>
          <Button
            mode="outlined"
            onPress={() => Alert.alert('Emergency', 'Calling 911...')}
            style={styles.emergencyButton}
            textColor="#e74c3c"
            icon="phone"
          >
            Call Emergency Services
          </Button>
        </Card.Content>
      </Card>

      {/* Patient ID Dialog */}
      <Portal>
        <Dialog visible={showPatientIdDialog} onDismiss={() => setShowPatientIdDialog(false)}>
          <Dialog.Title>Enter Patient ID</Dialog.Title>
          <Dialog.Content>
            <RNTextInput
              label="Patient ID"
              value={patientId}
              onChangeText={setPatientId}
              style={styles.dialogInput}
              mode="outlined"
              placeholder="Enter patient ID"
            />
            <Paragraph style={styles.dialogDescription}>
              Enter the patient ID to load their medical history and wound information.
            </Paragraph>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowPatientIdDialog(false)}>Cancel</Button>
            <Button onPress={handlePatientIdSubmit} loading={isLoadingPatient}>
              Load Patient
            </Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
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
  },
  sectionCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  sectionDescription: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 20,
  },
  patientDetailsCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  patientHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  patientName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  patientIdChip: {
    backgroundColor: '#3498db',
  },
  patientInfo: {
    marginBottom: 16,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  infoLabel: {
    fontSize: 14,
    color: '#7f8c8d',
    fontWeight: '500',
  },
  infoValue: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: 'bold',
  },
  changePatientButton: {
    borderColor: '#3498db',
  },
  noPatientCard: {
    alignItems: 'center',
    padding: 20,
  },
  noPatientText: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 16,
  },
  addPatientButton: {
    backgroundColor: '#3498db',
  },
  alertCard: {
    margin: 15,
    marginTop: 0,
    borderLeftWidth: 4,
    elevation: 4,
  },
  alertContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  alertText: {
    flex: 1,
    marginLeft: 12,
  },
  alertTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  alertDescription: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 12,
  },
  woundTypeChip: {
    alignSelf: 'flex-start',
  },
  doctorTypeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  doctorTypeCard: {
    width: (width - 60) / 2,
    borderWidth: 1,
    borderColor: '#e9ecef',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    backgroundColor: '#fff',
    borderLeftWidth: 4,
  },
  selectedDoctorTypeCard: {
    borderColor: '#27ae60',
    backgroundColor: '#f8fff8',
  },
  doctorTypeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  doctorTypeName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginLeft: 8,
    flex: 1,
  },
  doctorTypeDescription: {
    fontSize: 12,
    color: '#7f8c8d',
  },
  doctorCard: {
    borderWidth: 1,
    borderColor: '#e9ecef',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    backgroundColor: '#fff',
  },
  selectedDoctorCard: {
    borderColor: '#27ae60',
    backgroundColor: '#f8fff8',
  },
  doctorHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  doctorEmoji: {
    fontSize: 32,
    marginRight: 12,
  },
  doctorInfo: {
    flex: 1,
  },
  doctorName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 4,
  },
  doctorSpecialty: {
    fontSize: 14,
    color: '#3498db',
    marginBottom: 4,
  },
  doctorRating: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  ratingText: {
    fontSize: 14,
    color: '#2c3e50',
    marginLeft: 4,
    marginRight: 8,
  },
  experienceText: {
    fontSize: 14,
    color: '#7f8c8d',
    marginRight: 8,
  },
  feeText: {
    fontSize: 14,
    color: '#27ae60',
    fontWeight: 'bold',
  },
  doctorDetails: {
    marginLeft: 44,
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  detailText: {
    fontSize: 14,
    color: '#7f8c8d',
    marginLeft: 8,
  },
  doctorActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e9ecef',
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#f8f9fa',
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  actionButtonText: {
    fontSize: 14,
    color: '#2c3e50',
    marginLeft: 6,
    fontWeight: '500',
  },
  dateGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  dateButton: {
    width: (width - 60) / 2,
    padding: 16,
    borderWidth: 1,
    borderColor: '#e9ecef',
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12,
    backgroundColor: '#fff',
  },
  selectedDateButton: {
    borderColor: '#3498db',
    backgroundColor: '#e3f2fd',
  },
  dateButtonText: {
    fontSize: 16,
    color: '#2c3e50',
  },
  selectedDateButtonText: {
    color: '#3498db',
    fontWeight: 'bold',
  },
  timeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  timeButton: {
    width: (width - 60) / 2,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e9ecef',
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12,
    backgroundColor: '#fff',
  },
  selectedTimeButton: {
    borderColor: '#3498db',
    backgroundColor: '#e3f2fd',
  },
  timeButtonText: {
    fontSize: 14,
    color: '#2c3e50',
  },
  selectedTimeButtonText: {
    color: '#3498db',
    fontWeight: 'bold',
  },
  actionCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  bookButton: {
    backgroundColor: '#27ae60',
    paddingVertical: 8,
  },
  emergencyCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
    backgroundColor: '#fff5f5',
  },
  emergencyTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#e74c3c',
    marginBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
  },
  emergencyText: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 16,
  },
  emergencyButton: {
    borderColor: '#e74c3c',
  },
  dialogInput: {
    marginBottom: 16,
  },
  dialogDescription: {
    fontSize: 14,
    color: '#7f8c8d',
  },
});