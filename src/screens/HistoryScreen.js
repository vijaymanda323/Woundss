import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Alert,
  Modal,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
  List,
  Searchbar,
  Portal,
  Dialog,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import { fetchWithTimeout } from '../services/apiService';
import * as Haptics from 'expo-haptics';

export default function HistoryScreen({ navigation, route }) {
  const { patientId: routePatientId } = route.params || {};
  const [patientId, setPatientId] = useState(routePatientId || 'test_patient');
  const [isManualChange, setIsManualChange] = useState(false);
  
  // Only update from route params if it's not a manual change
  React.useEffect(() => {
    console.log('HistoryScreen route params:', route.params);
    console.log('Current patientId state:', patientId);
    console.log('Route patientId:', routePatientId);
    console.log('Is manual change:', isManualChange);
    console.log('Is changing patient:', isChangingPatient);
    console.log('Ignore route params:', ignoreRouteParams);

    // Only allow route params to change patient ID if:
    // 1. It's a different patient ID
    // 2. No manual change is in progress
    // 3. Not currently changing patient
    // 4. Not ignoring route params
    if (routePatientId && 
        routePatientId !== patientId && 
        !isManualChange && 
        !isChangingPatient && 
        !ignoreRouteParams) {
      console.log('✅ Patient ID changed from route params:', routePatientId);
      setPatientId(routePatientId);
    } else {
      console.log('❌ Route params blocked - manual:', isManualChange, 'changing:', isChangingPatient, 'ignore:', ignoreRouteParams, 'current:', patientId);
    }
  }, [routePatientId, patientId, isManualChange, isChangingPatient, ignoreRouteParams]);
  
  // Handle screen focus to update patient ID (only if not manual change)
  React.useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      console.log('HistoryScreen focused, checking route params:', route.params);
      if (route.params?.patientId && 
          route.params.patientId !== patientId && 
          !isManualChange && 
          !isChangingPatient && 
          !ignoreRouteParams) {
        console.log('Screen focused with new patient ID:', route.params.patientId);
        setPatientId(route.params.patientId);
      }
    });

    return unsubscribe;
  }, [navigation, route.params, patientId, isManualChange, isChangingPatient, ignoreRouteParams]);
  
  // Cleanup timeout on unmount
  React.useEffect(() => {
    return () => {
      if (patientIdTimeout) {
        clearTimeout(patientIdTimeout);
      }
    };
  }, [patientIdTimeout]);
  const [patientHistory, setPatientHistory] = useState([]);
  const [daysData, setDaysData] = useState([]);
  const [patientDetails, setPatientDetails] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [patientIdInput, setPatientIdInput] = useState('');
  const [patientIdTimeout, setPatientIdTimeout] = useState(null);
  const [isChangingPatient, setIsChangingPatient] = useState(false);
  const [ignoreRouteParams, setIgnoreRouteParams] = useState(false);
  const [selectedDay, setSelectedDay] = useState(null);
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [updateData, setUpdateData] = useState({
    area_pixels: '',
    area_cm2: '',
    painLevel: 'none',
    redness: 'none',
    swelling: 'none',
    notes: '',
    healing_pct: '',
    days_to_heal: '',
    previousAreaCm2: '0',
  });

  // Mock patient history data
  const mockHistory = [
    {
      id: 1,
      patient_id: patientId || 'P001',
      filename: 'wound_001.jpg',
      timestamp: '2024-01-15T10:30:00Z',
      area_pixels: 1250,
      area_cm2: 5.2,
      wound_type: 'burn',
      healing_time_category: 'moderate_healing',
      estimated_days_to_cure: 21,
      notes: 'Initial burn assessment',
    },
    {
      id: 2,
      patient_id: patientId || 'P001',
      filename: 'wound_002.jpg',
      timestamp: '2024-01-08T14:20:00Z',
      area_pixels: 1500,
      area_cm2: 6.1,
      wound_type: 'burn',
      healing_time_category: 'moderate_healing',
      estimated_days_to_cure: 25,
      notes: 'Follow-up visit',
    },
    {
      id: 3,
      patient_id: patientId || 'P001',
      filename: 'wound_003.jpg',
      timestamp: '2024-01-01T09:15:00Z',
      area_pixels: 1800,
      area_cm2: 7.3,
      wound_type: 'burn',
      healing_time_category: 'slow_healing',
      estimated_days_to_cure: 30,
      notes: 'Initial visit',
    },
  ];

  useEffect(() => {
    loadPatientHistory();
  }, [patientId]);

  const handlePatientIdChange = (newPatientId) => {
    console.log('🎯 handlePatientIdChange called with:', newPatientId);
    console.log('🎯 Current patientId:', patientId);
    console.log('🎯 Current flags - manual:', isManualChange, 'changing:', isChangingPatient, 'ignore:', ignoreRouteParams);
    
    if (newPatientId && newPatientId.trim() && newPatientId.trim() !== patientId) {
      console.log('✅ Valid change detected! Changing patient ID to:', newPatientId);
      console.log('✅ Current patient ID:', patientId);

      // Clear any existing timeout
      if (patientIdTimeout) {
        clearTimeout(patientIdTimeout);
        console.log('🧹 Cleared existing timeout');
      }

      console.log('🛡️ Setting protection flags');
      setIsManualChange(true); // Mark as manual change
      setIsChangingPatient(true); // Mark as changing patient
      setIgnoreRouteParams(true); // Ignore route params during manual change
      setIsLoading(true); // Set loading state
      setPatientIdInput(''); // Clear input after setting
      // Clear previous data immediately to prevent blinking
      setDaysData([]);
      setPatientDetails(null);
      setPatientHistory([]);
      console.log('🧹 Cleared all previous data');

      // Debounce the patient ID change
      const timeout = setTimeout(() => {
        console.log('⏰ Timeout executed - setting patient ID to:', newPatientId.trim());
        setPatientId(newPatientId.trim());
        // Reset flags after a delay
        setTimeout(() => {
          console.log('🔄 Resetting protection flags');
          setIsManualChange(false);
          setIsChangingPatient(false);
          setIgnoreRouteParams(false);
        }, 3000); // Increased to 3 seconds
      }, 300); // 300ms delay

      setPatientIdTimeout(timeout);
      console.log('⏰ Set timeout for patient ID change');
    } else {
      console.log('❌ Invalid change - same patient ID or empty');
    }
  };

  const loadPatientHistory = async () => {
    setIsLoading(true);
    try {
      // Fetch patient details
      if (patientId) {
        const patientResponse = await fetchWithTimeout(`http://10.81.160.244:5000/patient/${patientId}/details`, {
          method: 'GET',
        }, 8000);
        if (patientResponse.ok) {
          const patientData = await patientResponse.json();
          setPatientDetails(patientData.patient);
        }
      }

      // Fetch day-wise history
      const currentPatientId = patientId || 'test_patient';
      console.log('Loading history for patient:', currentPatientId);
      
      const daysResponse = await fetchWithTimeout(`http://10.81.160.244:5000/patient/${currentPatientId}/history/days`, {
        method: 'GET',
      }, 8000);
      if (daysResponse.ok) {
        const daysData = await daysResponse.json();
        console.log('Days data received:', daysData);
        setDaysData(daysData.days || []);
        
        // Flatten all records for backward compatibility
        const allRecords = [];
        daysData.days?.forEach(day => {
          allRecords.push(...day.records);
        });
        setPatientHistory(allRecords);
        console.log('Set patient history:', allRecords.length, 'records');
      } else {
        console.log('Days response failed, using mock data');
        // Use mock data if no real data available
        setPatientHistory(mockHistory);
        setDaysData([]);
      }
    } catch (error) {
      console.error('Error loading patient history:', error);
      // Use mock data as fallback
      setPatientHistory(mockHistory);
      setDaysData([]);
    } finally {
      setIsLoading(false);
    }
  };

  const onRefresh = async () => {
    setIsRefreshing(true);
    await loadPatientHistory();
    setIsRefreshing(false);
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

  const getHealingCategoryColor = (category) => {
    const colors = {
      fast_healing: '#27ae60',
      moderate_healing: '#f39c12',
      slow_healing: '#e67e22',
      chronic_non_healing: '#e74c3c',
    };
    return colors[category] || '#95a5a6';
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const calculateHealingProgress = (currentArea, previousArea) => {
    if (!previousArea || previousArea === 0) {
      console.log(`No previous area - Current=${currentArea}, Previous=${previousArea}`);
      return { progress: 0, isImprovement: true };
    }
    
    // Calculate percentage change
    const progress = ((previousArea - currentArea) / previousArea) * 100;
    const isImprovement = currentArea < previousArea; // Smaller area = better healing
    
    console.log(`Healing Progress: Current=${currentArea.toFixed(2)} cm², Previous=${previousArea.toFixed(2)} cm², Progress=${progress.toFixed(1)}%, IsImprovement=${isImprovement}`);
    
    return { 
      progress: Math.abs(progress), 
      isImprovement 
    };
  };

  const handleUpdateDay = (day) => {
    console.log('Update Day clicked for:', day.date);
    console.log('Day data:', day);
    setSelectedDay(day);
    
    // Get previous day for comparison
    const previousDay = daysData.find(d => d.date < day.date);
    const previousAreaCm2 = previousDay ? (previousDay.avg_area / 1153).toFixed(2) : '0';
    
    setUpdateData({
      painLevel: 'none', // Default to none for better healing
      redness: 'none', // Default to none for better healing
      swelling: 'none', // Default to none for better healing
      area_cm2: (day.avg_area / 1153).toFixed(2), // Convert pixels to cm² (1153 pixels = 1 cm²)
      notes: day.records[0]?.notes || '',
      previousAreaCm2: previousAreaCm2, // Store previous area for display
    });
    console.log('Setting update modal to true');
    setShowUpdateModal(true);
  };

  const handleDeleteDay = (day) => {
    setSelectedDay(day);
    setShowDeleteDialog(true);
  };

  const confirmUpdateDay = async () => {
    try {
      // Calculate healing progress based on pain, redness, swelling, and area
      const painScore = ['none', 'mild', 'moderate', 'severe'].indexOf(updateData.painLevel);
      const rednessScore = ['none', 'mild', 'moderate', 'severe'].indexOf(updateData.redness);
      const swellingScore = ['none', 'mild', 'moderate', 'severe'].indexOf(updateData.swelling);
      const areaCm2 = parseFloat(updateData.area_cm2);
      
      console.log('Update Day Calculation:', {
        painLevel: updateData.painLevel,
        painScore,
        redness: updateData.redness,
        rednessScore,
        swelling: updateData.swelling,
        swellingScore,
        areaCm2
      });
      
      // Calculate overall healing percentage (lower scores = better healing)
      const painProgress = ((4 - painScore) / 4) * 100; // Invert pain (0=none=best, 3=severe=worst)
      const rednessProgress = ((4 - rednessScore) / 4) * 100; // Invert redness
      const swellingProgress = ((4 - swellingScore) / 4) * 100; // Invert swelling
      
      // Calculate area progress (assuming smaller area = better healing)
      const previousDay = daysData.find(day => day.date < selectedDay.date);
      let areaProgress = 0;
      if (previousDay) {
        const previousArea = previousDay.avg_area / 1153; // Convert to cm² (1153 pixels = 1 cm²)
        // Calculate improvement percentage (positive = improvement)
        areaProgress = ((previousArea - areaCm2) / previousArea) * 100;
        console.log('Area Progress:', { previousArea, areaCm2, areaProgress });
      }
      
      // Overall healing percentage (weighted average)
      // Use Math.max(0, areaProgress) to ensure negative area progress doesn't hurt overall score
      const overallHealing = (painProgress * 0.3 + rednessProgress * 0.3 + swellingProgress * 0.3 + Math.max(0, areaProgress) * 0.1);
      
      console.log('Overall Healing Calculation:', {
        painProgress,
        rednessProgress,
        swellingProgress,
        areaProgress,
        overallHealing
      });
      
      const response = await fetch(`http://10.81.160.244:5000/patient/${patientId || 'test_patient'}/day/${selectedDay.date}/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          area_pixels: Math.round(areaCm2 * 1153), // Convert back to pixels (1153 pixels = 1 cm²)
          area_cm2: areaCm2,
          notes: updateData.notes,
          healing_pct: Math.round(overallHealing),
          days_to_heal: Math.max(1, Math.round(21 - (overallHealing / 100) * 20)), // Estimate days to heal
          pain_level: updateData.painLevel,
          redness: updateData.redness,
          swelling: updateData.swelling
        }),
      });

      if (response.ok) {
        Alert.alert('Success', `Day data updated successfully!\nHealing Progress: ${Math.round(overallHealing)}%`);
        setShowUpdateModal(false);
        loadPatientHistory(); // Refresh data
      } else {
        Alert.alert('Error', 'Failed to update day data');
      }
    } catch (error) {
      console.error('Error updating day:', error);
      Alert.alert('Error', 'Failed to update day data');
    }
  };

  const confirmDeleteDay = async () => {
    try {
      const response = await fetch(`http://10.81.160.244:5000/patient/${patientId || 'test_patient'}/day/${selectedDay.date}/delete`, {
        method: 'DELETE',
      });

      if (response.ok) {
        Alert.alert('Success', 'Day data deleted successfully!');
        setShowDeleteDialog(false);
        loadPatientHistory(); // Refresh data
      } else {
        Alert.alert('Error', 'Failed to delete day data');
      }
    } catch (error) {
      console.error('Error deleting day:', error);
      Alert.alert('Error', 'Failed to delete day data');
    }
  };

  const filteredHistory = patientHistory.filter(record => {
    const filename = record.filename || record.image_path || '';
    const woundType = record.wound_type || record.predicted_label || '';
    const notes = record.notes || '';
    
    return filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
           woundType.toLowerCase().includes(searchQuery.toLowerCase()) ||
           notes.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const renderDayItem = (day, index) => {
    const previousDay = index < daysData.length - 1 ? daysData[index + 1] : null;
    const currentAreaCm2 = day.avg_area / 1153; // Convert pixels to cm²
    const previousAreaCm2 = previousDay ? previousDay.avg_area / 1153 : 0; // Convert pixels to cm²
    const healingProgress = previousDay ? 
      calculateHealingProgress(currentAreaCm2, previousAreaCm2) : { progress: 0, isImprovement: true };

    // Check if this is today's record
    const today = new Date().toISOString().split('T')[0]; // Get YYYY-MM-DD format
    const isToday = day.date === today;

    return (
      <Card key={day.date} style={[styles.dayCard, isToday && styles.todayCard]}>
        <Card.Content>
          <View style={styles.dayPatientHeader}>
            <Text style={styles.dayPatientId}>Patient ID: {patientId || 'test_patient'}</Text>
            {isToday && (
              <Text style={styles.todayIndicator}>📅 TODAY</Text>
            )}
          </View>
          <View style={styles.dayHeader}>
            <View style={styles.dayInfo}>
              <Title style={[styles.dayTitle, isToday && styles.todayTitle]}>
                {new Date(day.date).toLocaleDateString('en-US', { 
                  weekday: 'long', 
                  year: 'numeric', 
                  month: 'long', 
                  day: 'numeric' 
                })}
                {isToday && ' 🎯'}
              </Title>
              <Paragraph style={styles.daySubtitle}>
                Patient: <Text style={styles.patientIdHighlight}>{patientId || 'test_patient'}</Text> • {day.total_records} record{day.total_records !== 1 ? 's' : ''}
              </Paragraph>
            </View>
            <View style={styles.dayChips}>
              <Chip style={styles.dayChip}>
                Avg: {day.avg_area.toFixed(0)}px
              </Chip>
            </View>
          </View>

          <View style={styles.dayDetails}>
            <View style={styles.detailRow}>
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>Min Area</Text>
                <Text style={styles.detailValue}>{day.min_area} px</Text>
              </View>
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>Max Area</Text>
                <Text style={styles.detailValue}>{day.max_area} px</Text>
              </View>
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>Avg Area</Text>
                <Text style={styles.detailValue}>{day.avg_area.toFixed(0)} px</Text>
              </View>
            </View>

            <View style={styles.progressContainer}>
              <Text style={styles.progressLabel}>
                {healingProgress.isImprovement ? 'Healing Progress' : 'Wound Worsening'}
              </Text>
              <View style={styles.progressBar}>
                <View 
                  style={[
                    styles.progressFill, 
                    { 
                      width: `${Math.min(Math.max(healingProgress.progress, 1), 100)}%`,
                      backgroundColor: healingProgress.isImprovement ? '#27ae60' : '#e74c3c'
                    }
                  ]} 
                />
              </View>
              <Text style={[
                styles.progressText,
                { color: healingProgress.isImprovement ? '#27ae60' : '#e74c3c' }
              ]}>
                {healingProgress.isImprovement ? '↓' : '↑'} {healingProgress.progress.toFixed(1)}% 
                {healingProgress.isImprovement ? ' improvement' : ' worsening'}
              </Text>
            </View>
          </View>

          <View style={styles.dayActions}>
            <Button
              mode="outlined"
              onPress={() => handleUpdateDay(day)}
              style={[styles.dayActionButton, { borderColor: '#3498db' }]}
              icon="pencil"
              labelStyle={{ color: '#3498db', fontWeight: 'bold' }}
            >
              Update Day
            </Button>
            <Button
              mode="contained"
              onPress={() => handleDeleteDay(day)}
              style={[styles.dayActionButton, { backgroundColor: '#e74c3c' }]}
              icon="delete"
              labelStyle={{ fontWeight: 'bold' }}
            >
              Delete Day
            </Button>
          </View>
        </Card.Content>
      </Card>
    );
  };

  const renderHistoryItem = (record, index) => {
    const previousRecord = index < patientHistory.length - 1 ? patientHistory[index + 1] : null;
    const healingProgress = previousRecord ? 
      calculateHealingProgress(record.area_cm2 || 0, previousRecord.area_cm2 || 0) : { progress: 0, isImprovement: true };

    // Handle different field names from backend vs mock data
    const filename = record.filename || record.image_path || 'Unknown';
    const woundType = record.wound_type || record.predicted_label || 'unknown';
    const area = record.area_cm2 || 0;
    const healingTime = record.estimated_days_to_cure || 14;
    const category = record.healing_time_category || 'moderate_healing';
    const confidence = record.confidence || 0.8;
    const feedbackStatus = record.feedback_status || 'unknown';

    return (
      <Card key={record.id || index} style={styles.historyCard}>
        <Card.Content>
          <View style={styles.historyHeader}>
            <View style={styles.historyInfo}>
              <Title style={styles.historyTitle}>
                {filename}
              </Title>
              <Paragraph style={styles.historyDate}>
                {formatDate(record.timestamp)}
              </Paragraph>
            </View>
            <View style={styles.historyChips}>
              <Chip
                style={[styles.chip, { backgroundColor: getWoundTypeColor(woundType) }]}
                textStyle={styles.chipText}
              >
                {woundType.toUpperCase()}
              </Chip>
            </View>
          </View>

          <View style={styles.historyDetails}>
            <View style={styles.detailRow}>
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>Area</Text>
                <Text style={styles.detailValue}>{area} cm²</Text>
              </View>
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>Healing Time</Text>
                <Text style={styles.detailValue}>{healingTime} days</Text>
              </View>
              <View style={styles.detailItem}>
                <Text style={styles.detailLabel}>Confidence</Text>
                <Text style={styles.detailValue}>{(confidence * 100).toFixed(1)}%</Text>
              </View>
            </View>

            {healingProgress.progress > 0 && (
              <View style={styles.progressContainer}>
                <Text style={styles.progressLabel}>
                  {healingProgress.isImprovement ? 'Healing Progress' : 'Wound Worsening'}
                </Text>
                <View style={styles.progressBar}>
                  <View 
                    style={[
                      styles.progressFill, 
                      { 
                        width: `${Math.min(Math.max(healingProgress.progress, 1), 100)}%`,
                        backgroundColor: healingProgress.isImprovement ? '#27ae60' : '#e74c3c'
                      }
                    ]} 
                  />
                </View>
                <Text style={[
                  styles.progressText,
                  { color: healingProgress.isImprovement ? '#27ae60' : '#e74c3c' }
                ]}>
                  {healingProgress.isImprovement ? '↓' : '↑'} {healingProgress.progress.toFixed(1)}% 
                  {healingProgress.isImprovement ? ' improvement' : ' worsening'}
                </Text>
              </View>
            )}

            {record.notes && (
              <View style={styles.notesContainer}>
                <Text style={styles.notesLabel}>Notes</Text>
                <Text style={styles.notesText}>{record.notes}</Text>
              </View>
            )}
          </View>

          <View style={styles.historyActions}>
            <Button
              mode="outlined"
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                Alert.alert('View Details', 'This would show detailed analysis for this record.');
              }}
              style={styles.actionButton}
              icon="eye"
            >
              View Details
            </Button>
            <Button
              mode="contained"
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                navigation.navigate('PatientTracker', {
                  patientData: { id: record.patient_id },
                  analysisData: {
                    woundType: record.wound_type,
                    area: record.area_cm2,
                    healingTime: record.estimated_days_to_cure,
                    timestamp: record.timestamp,
                  },
                  treatmentPlan: {
                    medications: [
                      {
                        name: 'Antibiotic',
                        dosage: '500mg',
                        frequency: 'daily',
                        time: 'Morning',
                        notes: 'As prescribed by doctor',
                      }
                    ]
                  }
                });
              }}
              style={[styles.actionButton, { backgroundColor: '#27ae60' }]}
              icon="calendar-clock"
            >
              Track Progress
            </Button>
          </View>
        </Card.Content>
      </Card>
    );
  };

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={isRefreshing} onRefresh={onRefresh} />
      }
    >
      <Card style={styles.headerCard}>
        <Card.Content>
          <View style={styles.headerRow}>
            <View style={styles.headerText}>
              <Title>Patient History</Title>
              <Paragraph>
                {patientDetails ? 
                  `History for ${patientDetails.name} (ID: ${patientDetails.id})` : 
                  `History for Patient ID: ${patientId || 'test_patient'}`
                }
              </Paragraph>
            {isChangingPatient && (
              <Paragraph style={styles.changingIndicator}>
                🔄 Switching to patient: {patientId}
              </Paragraph>
            )}
            <Paragraph style={styles.currentPatientIndicator}>
              📋 Currently loaded: <Text style={styles.patientIdHighlight}>{patientId}</Text>
              {patientId === 'PMFQ4MV2L8F437O' && ' (Vijay - Real Patient)'}
              {patientId === 'test_patient' && ' (Test Patient)'}
            </Paragraph>
            </View>
            <Button
              mode="outlined"
              onPress={onRefresh}
              style={styles.refreshButton}
              icon="refresh"
              compact
            >
              Refresh
            </Button>
          </View>
          {patientDetails && (
            <View style={styles.patientInfo}>
              <Text style={styles.patientDetail}>Gender: {patientDetails.gender || 'Not specified'}</Text>
              <Text style={styles.patientDetail}>Contact: {patientDetails.contact || 'Not provided'}</Text>
              <Text style={styles.patientDetail}>Clinician: {patientDetails.clinician || 'Not assigned'}</Text>
            </View>
          )}
        </Card.Content>
      </Card>

      {isLoading && (
        <Card style={styles.loadingCard}>
          <Card.Content style={styles.loadingContent}>
            <ActivityIndicator size="large" color="#3498db" />
            <Text style={styles.loadingText}>
              {patientId ? `Loading history for ${patientId}...` : 'Loading patient history...'}
            </Text>
          </Card.Content>
        </Card>
      )}

      {/* Patient ID Input Card */}
      <Card style={styles.inputCard}>
        <Card.Content>
          <Title style={styles.inputTitle}>🔍 Search by Patient ID</Title>
          <Paragraph style={styles.inputSubtitle}>
            Enter a patient ID to view their wound healing history
          </Paragraph>
          <View style={styles.inputRow}>
            <TextInput
              label="Patient ID"
              value={patientIdInput}
              onChangeText={setPatientIdInput}
              style={styles.patientIdInput}
              mode="outlined"
              keyboardType="default"
              placeholder="Enter patient ID (e.g., PMFQ4MV2L8F437O)"
              autoFocus={false}
              returnKeyType="search"
              onSubmitEditing={() => handlePatientIdChange(patientIdInput)}
            />
            <Button
              mode="contained"
              onPress={() => handlePatientIdChange(patientIdInput)}
              style={styles.searchButton}
              disabled={!patientIdInput.trim() || isLoading}
              icon="magnify"
            >
              Search
            </Button>
          </View>
        <View style={styles.quickSearchRow}>
          <Button
            mode="outlined"
            onPress={() => {
              setPatientIdInput('test_patient');
              handlePatientIdChange('test_patient');
            }}
            style={styles.quickButton}
            icon="test-tube"
            compact
          >
            Test Patient
          </Button>
        </View>
        </Card.Content>
      </Card>

      <Card style={styles.searchCard}>
        <Card.Content>
          <Searchbar
            placeholder="Search records..."
            onChangeText={setSearchQuery}
            value={searchQuery}
            style={styles.searchbar}
          />
        </Card.Content>
      </Card>

      <View style={styles.statsContainer}>
        <Card style={styles.statCard}>
          <Card.Content>
            <Title style={styles.statTitle}>Total Records</Title>
            <Text style={styles.statValue}>{filteredHistory.length}</Text>
          </Card.Content>
        </Card>
        
        <Card style={styles.statCard}>
          <Card.Content>
            <Title style={styles.statTitle}>Total Days</Title>
            <Text style={styles.statValue}>{daysData.length}</Text>
          </Card.Content>
        </Card>
        
        <Card style={styles.statCard}>
          <Card.Content>
            <Title style={styles.statTitle}>Latest Analysis</Title>
            <Text style={styles.statValue}>
              {filteredHistory.length > 0 ? formatDate(filteredHistory[0].timestamp) : 'N/A'}
            </Text>
          </Card.Content>
        </Card>
      </View>

      {daysData.length > 0 && (
        <Card style={styles.sectionCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>📅 Day-wise Tracking</Title>
            <Paragraph style={styles.sectionSubtitle}>
              Patient ID: <Text style={styles.patientIdHighlight}>{patientId || 'test_patient'}</Text> - {daysData.length} days of data
            </Paragraph>
            <Paragraph style={styles.sectionSubtitle}>
              View and manage wound data by day. Click "Update Day" to modify data or "Delete Day" to restart tracking from that point.
            </Paragraph>
          </Card.Content>
        </Card>
      )}

      {daysData.length > 0 ? (
        <View style={styles.historyContainer}>
          <Card style={styles.debugCard}>
            <Card.Content>
              <Text style={styles.debugText}>
                📊 Showing {daysData.length} days of data for Patient ID: <Text style={styles.patientIdHighlight}>{patientId}</Text>
              </Text>
              <Text style={styles.debugSubtext}>
                Latest record: {daysData[0]?.date} ({daysData[0]?.total_records} records)
              </Text>
            </Card.Content>
          </Card>
          {daysData.map((day, index) => renderDayItem(day, index))}
        </View>
      ) : filteredHistory.length > 0 ? (
        <View style={styles.historyContainer}>
          <Text style={styles.debugText}>Showing {filteredHistory.length} individual records</Text>
          {filteredHistory.map((record, index) => renderHistoryItem(record, index))}
        </View>
      ) : (
        <Card style={styles.emptyCard}>
          <Card.Content>
            <View style={styles.emptyContainer}>
              <Ionicons name="document-outline" size={60} color="#bdc3c7" />
              <Title style={styles.emptyTitle}>No Records Found</Title>
              <Paragraph style={styles.emptyText}>
                {searchQuery ? 'No records match your search criteria.' : 'No analysis records found for this patient.'}
              </Paragraph>
              <Button
                mode="contained"
                onPress={() => navigation.navigate('Camera')}
                style={styles.emptyButton}
                icon="camera"
              >
                Start New Analysis
              </Button>
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Update Day Modal */}
      <Modal
        visible={showUpdateModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowUpdateModal(false)}
      >
        <View style={styles.modalOverlay}>
          <ScrollView 
            style={styles.modalScrollView}
            contentContainerStyle={styles.modalScrollContent}
            keyboardShouldPersistTaps="handled"
          >
            <View style={styles.modalContent}>
              <Title style={styles.modalTitle}>Update Day Data</Title>
              <Paragraph style={styles.modalSubtitle}>
                Update wound data for {selectedDay?.date}
              </Paragraph>
              {console.log('Update modal is visible, selectedDay:', selectedDay)}

            <Text style={styles.inputLabel}>Pain Level</Text>
            <View style={styles.radioGroup}>
              {['none', 'mild', 'moderate', 'severe'].map((level) => (
                <TouchableOpacity
                  key={level}
                  style={styles.radioItem}
                  onPress={() => setUpdateData(prev => ({ ...prev, painLevel: level }))}
                >
                  <View style={[
                    styles.radioCircle,
                    updateData.painLevel === level && styles.radioCircleSelected
                  ]}>
                    {updateData.painLevel === level && <View style={styles.radioInner} />}
                  </View>
                  <Text style={styles.radioLabel}>{level}</Text>
                </TouchableOpacity>
              ))}
            </View>
            
            <Text style={styles.inputLabel}>Redness</Text>
            <View style={styles.radioGroup}>
              {['none', 'mild', 'moderate', 'severe'].map((level) => (
                <TouchableOpacity
                  key={level}
                  style={styles.radioItem}
                  onPress={() => setUpdateData(prev => ({ ...prev, redness: level }))}
                >
                  <View style={[
                    styles.radioCircle,
                    updateData.redness === level && styles.radioCircleSelected
                  ]}>
                    {updateData.redness === level && <View style={styles.radioInner} />}
                  </View>
                  <Text style={styles.radioLabel}>{level}</Text>
                </TouchableOpacity>
              ))}
            </View>
            
            <Text style={styles.inputLabel}>Swelling</Text>
            <View style={styles.radioGroup}>
              {['none', 'mild', 'moderate', 'severe'].map((level) => (
                <TouchableOpacity
                  key={level}
                  style={styles.radioItem}
                  onPress={() => setUpdateData(prev => ({ ...prev, swelling: level }))}
                >
                  <View style={[
                    styles.radioCircle,
                    updateData.swelling === level && styles.radioCircleSelected
                  ]}>
                    {updateData.swelling === level && <View style={styles.radioInner} />}
                  </View>
                  <Text style={styles.radioLabel}>{level}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.inputLabel}>Wound Area (cm²)</Text>
            <View style={styles.areaInfoContainer}>
              <Text style={styles.currentAreaText}>
                Current: {updateData.area_cm2} cm²
              </Text>
              {updateData.previousAreaCm2 && updateData.previousAreaCm2 !== '0' && (
                <Text style={styles.previousAreaText}>
                  Previous: {updateData.previousAreaCm2} cm²
                </Text>
              )}
            </View>
            <TextInput
              label="Enter wound area in square centimeters"
              value={updateData.area_cm2}
              onChangeText={(text) => setUpdateData(prev => ({ ...prev, area_cm2: text }))}
              style={styles.modalInput}
              mode="outlined"
              keyboardType="numeric"
              placeholder="0.00"
              returnKeyType="done"
            />

            <TextInput
              label="Notes"
              value={updateData.notes}
              onChangeText={(text) => setUpdateData(prev => ({ ...prev, notes: text }))}
              style={styles.modalInput}
              mode="outlined"
              multiline
              numberOfLines={3}
            />

            <View style={styles.modalActions}>
              <Button
                mode="outlined"
                onPress={() => setShowUpdateModal(false)}
                style={styles.modalButton}
              >
                Cancel
              </Button>
              <Button
                mode="contained"
                onPress={confirmUpdateDay}
                style={[styles.modalButton, { backgroundColor: '#27ae60' }]}
              >
                Update
              </Button>
            </View>
            </View>
          </ScrollView>
        </View>
      </Modal>

      {/* Delete Day Dialog */}
      <Portal>
        <Dialog visible={showDeleteDialog} onDismiss={() => setShowDeleteDialog(false)}>
          <Dialog.Title>Delete Day Data</Dialog.Title>
          <Dialog.Content>
            <Paragraph>
              Are you sure you want to delete all records for {selectedDay?.date}?
              This action cannot be undone and will restart tracking from this point.
            </Paragraph>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowDeleteDialog(false)}>Cancel</Button>
            <Button onPress={confirmDeleteDay} mode="contained" buttonColor="#e74c3c">
              Delete
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
  headerCard: {
    margin: 15,
    elevation: 4,
  },
  inputCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    backgroundColor: '#e8f5e8',
  },
  inputTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  inputSubtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 15,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  patientIdInput: {
    flex: 1,
    backgroundColor: 'white',
    fontSize: 16,
    minHeight: 50,
  },
  searchButton: {
    backgroundColor: '#27ae60',
  },
  testButton: {
    borderColor: '#f39c12',
  },
  loadingCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
    backgroundColor: '#f8f9fa',
  },
  loadingContent: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  changingIndicator: {
    fontSize: 14,
    color: '#f39c12',
    fontStyle: 'italic',
    marginTop: 5,
  },
  currentPatientIndicator: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: 'bold',
    marginTop: 5,
    backgroundColor: '#ecf0f1',
    padding: 8,
    borderRadius: 5,
  },
  todayCard: {
    borderColor: '#e74c3c',
    borderWidth: 2,
    backgroundColor: '#fdf2f2',
  },
  todayIndicator: {
    fontSize: 12,
    color: '#e74c3c',
    fontWeight: 'bold',
    backgroundColor: '#ffe6e6',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    marginLeft: 10,
  },
  todayTitle: {
    color: '#e74c3c',
    fontWeight: 'bold',
  },
  debugCard: {
    margin: 15,
    marginTop: 0,
    backgroundColor: '#e8f4fd',
    borderColor: '#3498db',
    borderWidth: 1,
  },
  debugSubtext: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  quickSearchRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 15,
    gap: 10,
  },
  quickButton: {
    minWidth: 150,
    borderColor: '#3498db',
  },
  previousAreaText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    backgroundColor: '#f0f0f0',
    padding: 8,
    borderRadius: 5,
    flex: 1,
    textAlign: 'center',
  },
  areaInfoContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
    gap: 10,
  },
  currentAreaText: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: 'bold',
    backgroundColor: '#e8f4fd',
    padding: 8,
    borderRadius: 5,
    flex: 1,
    textAlign: 'center',
  },
  searchCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  searchbar: {
    elevation: 0,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
    marginBottom: 15,
  },
  statCard: {
    flex: 0.48,
    elevation: 4,
  },
  statTitle: {
    fontSize: 14,
    color: '#7f8c8d',
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginTop: 5,
  },
  historyContainer: {
    paddingHorizontal: 15,
  },
  historyCard: {
    marginBottom: 15,
    elevation: 4,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 15,
  },
  historyInfo: {
    flex: 1,
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  historyDate: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  historyChips: {
    alignItems: 'flex-end',
  },
  chip: {
    alignSelf: 'flex-start',
  },
  chipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  smallChip: {
    alignSelf: 'flex-start',
  },
  smallChipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 10,
  },
  historyDetails: {
    marginBottom: 15,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  detailItem: {
    alignItems: 'center',
    flex: 1,
  },
  detailLabel: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 5,
  },
  detailValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  progressContainer: {
    marginBottom: 15,
  },
  progressLabel: {
    fontSize: 14,
    color: '#2c3e50',
    marginBottom: 5,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 5,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#27ae60',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 12,
    color: '#27ae60',
    fontWeight: '600',
  },
  notesContainer: {
    marginBottom: 15,
  },
  notesLabel: {
    fontSize: 14,
    color: '#2c3e50',
    marginBottom: 5,
  },
  notesText: {
    fontSize: 14,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  historyActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionButton: {
    flex: 0.48,
  },
  emptyCard: {
    margin: 15,
    elevation: 4,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyTitle: {
    fontSize: 20,
    color: '#7f8c8d',
    marginTop: 15,
  },
  emptyText: {
    fontSize: 16,
    color: '#bdc3c7',
    textAlign: 'center',
    marginTop: 10,
    marginBottom: 20,
  },
  emptyButton: {
    backgroundColor: '#667eea',
  },
  patientInfo: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#e3f2fd',
    borderRadius: 8,
  },
  patientDetail: {
    fontSize: 14,
    color: '#2c3e50',
    marginBottom: 5,
  },
  sectionCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#667eea',
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  dayCard: {
    marginBottom: 15,
    elevation: 4,
  },
  dayHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 15,
  },
  dayInfo: {
    flex: 1,
  },
  dayTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  daySubtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  dayChips: {
    alignItems: 'flex-end',
  },
  dayChip: {
    backgroundColor: '#3498db',
  },
  dayActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 15,
  },
  dayActionButton: {
    flex: 0.48,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalScrollView: {
    width: '90%',
    maxHeight: '85%',
  },
  modalScrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 20,
    marginVertical: 20,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 20,
  },
  modalInput: {
    marginBottom: 20,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 20,
  },
  modalButton: {
    flex: 0.48,
  },
  debugText: {
    fontSize: 12,
    color: '#7f8c8d',
    textAlign: 'center',
    marginBottom: 10,
    fontStyle: 'italic',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  headerText: {
    flex: 1,
  },
  refreshButton: {
    marginLeft: 10,
  },
  patientIdHighlight: {
    fontWeight: 'bold',
    color: '#e74c3c',
    backgroundColor: '#fdf2f2',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  dayPatientHeader: {
    backgroundColor: '#f8f9fa',
    padding: 8,
    borderRadius: 4,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#e74c3c',
  },
  dayPatientId: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2c3e50',
    textAlign: 'center',
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginTop: 15,
    marginBottom: 10,
  },
  radioGroup: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 20,
    justifyContent: 'space-between',
  },
  radioItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 10,
    marginBottom: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#f8f9fa',
    borderRadius: 20,
    minWidth: 60,
    justifyContent: 'center',
  },
  radioLabel: {
    marginLeft: 8,
    fontSize: 14,
    color: '#2c3e50',
  },
  radioCircle: {
    width: 18,
    height: 18,
    borderRadius: 9,
    borderWidth: 2,
    borderColor: '#bdc3c7',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 5,
  },
  radioCircleSelected: {
    borderColor: '#3498db',
    backgroundColor: '#e3f2fd',
  },
  radioInner: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#3498db',
  },
});



