import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { Card, Title, Paragraph, Button, IconButton } from 'react-native-paper';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

export default function HomeScreen({ navigation }) {
  const features = [
    {
      icon: 'medical',
      title: 'Appointment to Doctor',
      description: 'Book appointment with medical professionals',
      color: '#3498db',
      action: () => navigation.navigate('DoctorAppointment'),
    },
    {
      icon: 'cloud-upload',
      title: 'Upload Photo',
      description: 'Upload wound images from gallery or files',
      color: '#2ecc71',
      action: () => navigation.navigate('PhotoUpload'),
    },
    {
      icon: 'analytics',
      title: 'AI Analysis',
      description: 'Advanced wound analysis and classification',
      color: '#9b59b6',
      action: () => navigation.navigate('Analysis'),
    },
    {
      icon: 'medical',
      title: 'Treatment Plan',
      description: 'Personalized treatment recommendations',
      color: '#27ae60',
      action: () => navigation.navigate('PhotoUpload'),
    },
    {
      icon: 'document-text',
      title: 'Reports',
      description: 'Generate patient and clinician reports',
      color: '#e67e22',
      action: () => navigation.navigate('Reports'),
    },
    {
      icon: 'time',
      title: 'History',
      description: 'Track healing progress over time',
      color: '#e74c3c',
      action: () => navigation.navigate('History', { patientId: 'test_patient' }),
    },
    {
      icon: 'person',
      title: 'Patient Info',
      description: 'Manage patient information',
      color: '#34495e',
      action: () => navigation.navigate('PatientInfo'),
    },
  ];

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Ionicons name="medical" size={60} color="white" />
          <Text style={styles.headerTitle}>Wound Healing Tracker</Text>
          <Text style={styles.headerSubtitle}>
            AI-powered wound analysis for better patient care
          </Text>
        </View>
      </LinearGradient>

      <View style={styles.content}>
        <Card style={styles.welcomeCard}>
          <Card.Content>
            <Title>Welcome to Wound Healing Tracker</Title>
            <Paragraph style={styles.welcomeText}>
              This advanced application uses artificial intelligence to analyze wound images,
              predict healing times, and provide comprehensive treatment recommendations.
              Available on web, Android, and iOS platforms.
            </Paragraph>
          </Card.Content>
        </Card>

        <Text style={styles.featuresTitle}>Features</Text>
        
        <View style={styles.featuresGrid}>
          {features.map((feature, index) => (
            <TouchableOpacity
              key={index}
              style={[styles.featureCard, { backgroundColor: feature.color }]}
              onPress={feature.action}
            >
              <Ionicons name={feature.icon} size={40} color="white" />
              <Text style={styles.featureTitle}>{feature.title}</Text>
              <Text style={styles.featureDescription}>{feature.description}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <Card style={styles.quickStartCard}>
          <Card.Content>
            <Title>Quick Start</Title>
            <Paragraph>
              1. Take a photo of the wound{'\n'}
              2. Enter patient information{'\n'}
              3. Get AI analysis and recommendations{'\n'}
              4. Generate professional reports
            </Paragraph>
            <Button
              mode="contained"
              onPress={() => navigation.navigate('Camera')}
              style={styles.startButton}
            >
              Start Analysis
            </Button>
          </Card.Content>
        </Card>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    paddingTop: 60,
    paddingBottom: 40,
    paddingHorizontal: 20,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 10,
    textAlign: 'center',
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.9)',
    marginTop: 5,
    textAlign: 'center',
  },
  content: {
    padding: 20,
  },
  welcomeCard: {
    marginBottom: 20,
    elevation: 4,
  },
  welcomeText: {
    marginTop: 10,
    lineHeight: 22,
  },
  featuresTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  featureCard: {
    width: (width - 60) / 2,
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
    marginBottom: 15,
    elevation: 3,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 10,
    textAlign: 'center',
  },
  featureDescription: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.9)',
    marginTop: 5,
    textAlign: 'center',
  },
  quickStartCard: {
    elevation: 4,
  },
  startButton: {
    marginTop: 15,
    backgroundColor: '#667eea',
  },
});
