import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  Modal,
  Platform,
} from 'react-native';
import {
  Card,
  Title,
  Paragraph,
  Button,
  TextInput,
  List,
  Chip,
  FAB,
  Portal,
  Dialog,
  RadioButton,
  Divider,
} from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { formatDate, generatePatientId } from '../utils/patientUtils';

export default function PatientTrackerScreen({ navigation, route }) {
  const { patientData, analysisData, treatmentPlan } = route.params || {};
  
  const [currentDay, setCurrentDay] = useState(1);
  const [totalDays, setTotalDays] = useState(analysisData?.healingTime || 21);
  const [medicineSchedule, setMedicineSchedule] = useState([]);
  const [woundProgress, setWoundProgress] = useState([]);
  const [showAddMedicine, setShowAddMedicine] = useState(false);
  const [showUpdateWound, setShowUpdateWound] = useState(false);
  const [newMedicine, setNewMedicine] = useState({
    name: '',
    dosage: '',
    frequency: 'daily',
    time: '',
    notes: '',
  });
  const [woundUpdate, setWoundUpdate] = useState({
    area: '',
    painLevel: '1',
    redness: 'none',
    swelling: 'none',
    discharge: 'none',
    notes: '',
  });

  // Initialize medicine schedule based on treatment plan
  useEffect(() => {
    if (treatmentPlan?.medications) {
      const initialMedicines = treatmentPlan.medications.map(med => ({
        id: Date.now() + Math.random(),
        name: med.name || 'Antibiotic',
        dosage: med.dosage || '500mg',
        frequency: med.frequency || 'daily',
        time: med.time || 'Morning',
        notes: med.notes || '',
        days: Array.from({ length: totalDays }, (_, i) => ({
          day: i + 1,
          taken: false,
          time: '',
          notes: '',
        })),
      }));
      setMedicineSchedule(initialMedicines);
    }
  }, [treatmentPlan, totalDays]);

  // Initialize wound progress tracking
  useEffect(() => {
    if (analysisData) {
      const initialProgress = {
        day: 0,
        area: analysisData.area || 5.0,
        painLevel: '3',
        redness: 'moderate',
        swelling: 'moderate',
        discharge: 'none',
        notes: 'Initial assessment',
        timestamp: new Date().toISOString(),
      };
      setWoundProgress([initialProgress]);
    }
  }, [analysisData]);

  const addMedicine = () => {
    if (!newMedicine.name.trim()) {
      Alert.alert('Required Field', 'Please enter medicine name.');
      return;
    }

    const medicine = {
      id: Date.now() + Math.random(),
      ...newMedicine,
      days: Array.from({ length: totalDays }, (_, i) => ({
        day: i + 1,
        taken: false,
        time: '',
        notes: '',
      })),
    };

    setMedicineSchedule([...medicineSchedule, medicine]);
    setNewMedicine({
      name: '',
      dosage: '',
      frequency: 'daily',
      time: '',
      notes: '',
    });
    setShowAddMedicine(false);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const updateMedicineTaken = (medicineId, day, taken, time = '', notes = '') => {
    setMedicineSchedule(prev => 
      prev.map(med => 
        med.id === medicineId 
          ? {
              ...med,
              days: med.days.map(d => 
                d.day === day ? { ...d, taken, time, notes } : d
              )
            }
          : med
      )
    );
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const addWoundUpdate = () => {
    if (!woundUpdate.area.trim()) {
      Alert.alert('Required Field', 'Please enter wound area.');
      return;
    }

    const update = {
      day: currentDay,
      area: parseFloat(woundUpdate.area),
      painLevel: woundUpdate.painLevel,
      redness: woundUpdate.redness,
      swelling: woundUpdate.swelling,
      discharge: woundUpdate.discharge,
      notes: woundUpdate.notes,
      timestamp: new Date().toISOString(),
    };

    setWoundProgress([...woundProgress, update]);
    setWoundUpdate({
      area: '',
      painLevel: '1',
      redness: 'none',
      swelling: 'none',
      discharge: 'none',
      notes: '',
    });
    setShowUpdateWound(false);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const getPainLevelColor = (level) => {
    const colors = {
      '1': '#27ae60', // Green - No pain
      '2': '#f39c12', // Orange - Mild pain
      '3': '#e67e22', // Dark orange - Moderate pain
      '4': '#e74c3c', // Red - Severe pain
      '5': '#8e44ad', // Purple - Extreme pain
    };
    return colors[level] || '#95a5a6';
  };

  const getSeverityColor = (severity) => {
    const colors = {
      'none': '#27ae60',
      'mild': '#f39c12',
      'moderate': '#e67e22',
      'severe': '#e74c3c',
    };
    return colors[severity] || '#95a5a6';
  };

  const calculateHealingProgress = () => {
    if (woundProgress.length < 2) return 0;
    const initial = woundProgress[0].area;
    const latest = woundProgress[woundProgress.length - 1].area;
    return ((initial - latest) / initial) * 100;
  };

  const calculateDayProgress = () => {
    if (woundProgress.length < 2) return { progress: 0, isImprovement: true };
    
    const currentDayData = woundProgress.find(p => p.day === currentDay);
    const previousDayData = woundProgress.find(p => p.day === currentDay - 1);
    
    if (!currentDayData || !previousDayData) {
      return { progress: 0, isImprovement: true };
    }
    
    const previousArea = previousDayData.area;
    const currentArea = currentDayData.area;
    const progress = ((previousArea - currentArea) / previousArea) * 100;
    const isImprovement = currentArea < previousArea;
    
    return { progress: Math.abs(progress), isImprovement };
  };

  const getCurrentDayProgress = () => {
    const dayProgress = woundProgress.find(p => p.day === currentDay);
    return dayProgress || null;
  };

  const getMedicineForDay = (day) => {
    return medicineSchedule.map(med => ({
      ...med,
      dayData: med.days.find(d => d.day === day),
    }));
  };

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <Card style={styles.headerCard}>
          <Card.Content>
            <Title>📅 Day {currentDay} of {totalDays}</Title>
            <Paragraph>
              Patient: {patientData?.name || 'Unknown'} | 
              Wound Type: {(analysisData?.woundType || 'unknown').toUpperCase()}
            </Paragraph>
            <View style={styles.progressContainer}>
              <Text style={styles.progressLabel}>Overall Healing Progress</Text>
              <View style={styles.progressBar}>
                <View 
                  style={[
                    styles.progressFill, 
                    { width: `${Math.min(calculateHealingProgress(), 100)}%` }
                  ]} 
                />
              </View>
              <Text style={styles.progressText}>
                {calculateHealingProgress().toFixed(1)}% improvement
              </Text>
            </View>
          </Card.Content>
        </Card>

        {/* Day Navigation */}
        <Card style={styles.navigationCard}>
          <Card.Content>
            <Title>📅 Day Navigation</Title>
            <View style={styles.dayNavigation}>
              <Button
                mode="outlined"
                onPress={() => setCurrentDay(Math.max(1, currentDay - 1))}
                disabled={currentDay <= 1}
                icon="chevron-left"
              >
                Previous
              </Button>
              <Text style={styles.currentDayText}>Day {currentDay}</Text>
              <Button
                mode="outlined"
                onPress={() => setCurrentDay(Math.min(totalDays, currentDay + 1))}
                disabled={currentDay >= totalDays}
                icon="chevron-right"
              >
                Next
              </Button>
            </View>
          </Card.Content>
        </Card>

        {/* Wound Progress */}
        <Card style={styles.woundCard}>
          <Card.Content>
            <View style={styles.cardHeader}>
              <Title>🩹 Wound Progress</Title>
              <Button
                mode="outlined"
                onPress={() => setShowUpdateWound(true)}
                icon="plus"
                compact
              >
                Update
              </Button>
            </View>
            
            {getCurrentDayProgress() ? (
              <View style={styles.woundDetails}>
                <View style={styles.detailRow}>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Area</Text>
                    <Text style={styles.detailValue}>
                      {getCurrentDayProgress().area} cm²
                    </Text>
                  </View>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Pain Level</Text>
                    <Chip
                      style={[styles.chip, { backgroundColor: getPainLevelColor(getCurrentDayProgress().painLevel) }]}
                      textStyle={styles.chipText}
                    >
                      {getCurrentDayProgress().painLevel}/5
                    </Chip>
                  </View>
                </View>
                
                <View style={styles.detailRow}>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Redness</Text>
                    <Chip
                      style={[styles.chip, { backgroundColor: getSeverityColor(getCurrentDayProgress().redness) }]}
                      textStyle={styles.chipText}
                    >
                      {getCurrentDayProgress().redness}
                    </Chip>
                  </View>
                  <View style={styles.detailItem}>
                    <Text style={styles.detailLabel}>Swelling</Text>
                    <Chip
                      style={[styles.chip, { backgroundColor: getSeverityColor(getCurrentDayProgress().swelling) }]}
                      textStyle={styles.chipText}
                    >
                      {getCurrentDayProgress().swelling}
                    </Chip>
                  </View>
                </View>
                
                {getCurrentDayProgress().notes && (
                  <View style={styles.notesContainer}>
                    <Text style={styles.notesLabel}>Notes</Text>
                    <Text style={styles.notesText}>{getCurrentDayProgress().notes}</Text>
                  </View>
                )}

                {/* Day-to-Day Progress Bar */}
                {currentDay > 1 && (() => {
                  const dayProgress = calculateDayProgress();
                  return (
                    <View style={styles.dayProgressContainer}>
                      <Text style={styles.dayProgressLabel}>
                        Change from Day {currentDay - 1}
                      </Text>
                      <View style={styles.dayProgressBar}>
                        <View 
                          style={[
                            styles.dayProgressFill, 
                            { 
                              width: `${Math.min(dayProgress.progress, 100)}%`,
                              backgroundColor: dayProgress.isImprovement ? '#27ae60' : '#e74c3c'
                            }
                          ]} 
                        />
                      </View>
                      <Text style={[
                        styles.dayProgressText,
                        { color: dayProgress.isImprovement ? '#27ae60' : '#e74c3c' }
                      ]}>
                        {dayProgress.isImprovement ? '↓' : '↑'} {dayProgress.progress.toFixed(1)}% 
                        {dayProgress.isImprovement ? ' improvement' : ' worsening'}
                      </Text>
                    </View>
                  );
                })()}
              </View>
            ) : (
              <View style={styles.emptyState}>
                <Text style={styles.emptyText}>No wound data for Day {currentDay}</Text>
                <Button
                  mode="contained"
                  onPress={() => setShowUpdateWound(true)}
                  icon="plus"
                >
                  Add Wound Update
                </Button>
              </View>
            )}
          </Card.Content>
        </Card>

        {/* Medicine Schedule */}
        <Card style={styles.medicineCard}>
          <Card.Content>
            <View style={styles.cardHeader}>
              <Title>💊 Medicine Schedule</Title>
              <Button
                mode="outlined"
                onPress={() => setShowAddMedicine(true)}
                icon="plus"
                compact
              >
                Add Medicine
              </Button>
            </View>
            
            {getMedicineForDay(currentDay).map((medicine) => (
              <View key={medicine.id} style={styles.medicineItem}>
                <View style={styles.medicineInfo}>
                  <Text style={styles.medicineName}>{medicine.name}</Text>
                  <Text style={styles.medicineDetails}>
                    {medicine.dosage} - {medicine.frequency} - {medicine.time}
                  </Text>
                  {medicine.notes && (
                    <Text style={styles.medicineNotes}>{medicine.notes}</Text>
                  )}
                </View>
                
                <View style={styles.medicineActions}>
                  <Button
                    mode={medicine.dayData?.taken ? "contained" : "outlined"}
                    onPress={() => updateMedicineTaken(
                      medicine.id, 
                      currentDay, 
                      !medicine.dayData?.taken,
                      new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    )}
                    icon={medicine.dayData?.taken ? "check" : "clock"}
                    compact
                  >
                    {medicine.dayData?.taken ? 'Taken' : 'Mark Taken'}
                  </Button>
                </View>
                
                {medicine.dayData?.taken && (
                  <View style={styles.takenInfo}>
                    <Text style={styles.takenText}>
                      Taken at: {medicine.dayData.time}
                    </Text>
                    {medicine.dayData.notes && (
                      <Text style={styles.takenNotes}>{medicine.dayData.notes}</Text>
                    )}
                  </View>
                )}
              </View>
            ))}
            
            {getMedicineForDay(currentDay).length === 0 && (
              <View style={styles.emptyState}>
                <Text style={styles.emptyText}>No medicines scheduled for Day {currentDay}</Text>
                <Button
                  mode="contained"
                  onPress={() => setShowAddMedicine(true)}
                  icon="plus"
                >
                  Add Medicine
                </Button>
              </View>
            )}
          </Card.Content>
        </Card>

        {/* Progress Timeline */}
        <Card style={styles.timelineCard}>
          <Card.Content>
            <Title>📈 Progress Timeline</Title>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.timeline}>
                {woundProgress.map((progress, index) => {
                  const isImprovement = index === 0 || progress.area < woundProgress[index - 1].area;
                  const dotColor = isImprovement ? '#27ae60' : '#e74c3c';
                  
                  return (
                    <View key={index} style={styles.timelineItem}>
                      <View style={[styles.timelineDot, { backgroundColor: dotColor }]} />
                      <Text style={styles.timelineDay}>Day {progress.day}</Text>
                      <Text style={styles.timelineArea}>{progress.area} cm²</Text>
                      <Chip
                        style={[styles.timelineChip, { backgroundColor: getPainLevelColor(progress.painLevel) }]}
                        textStyle={styles.timelineChipText}
                      >
                        Pain: {progress.painLevel}/5
                      </Chip>
                      {index > 0 && (
                        <Text style={[styles.timelineChange, { color: dotColor }]}>
                          {isImprovement ? '↓' : '↑'}
                        </Text>
                      )}
                    </View>
                  );
                })}
              </View>
            </ScrollView>
          </Card.Content>
        </Card>
      </ScrollView>

      {/* Add Medicine Modal */}
      <Portal>
        <Dialog visible={showAddMedicine} onDismiss={() => setShowAddMedicine(false)}>
          <Dialog.Title>Add Medicine</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Medicine Name *"
              value={newMedicine.name}
              onChangeText={(text) => setNewMedicine(prev => ({ ...prev, name: text }))}
              style={styles.input}
              mode="outlined"
            />
            <TextInput
              label="Dosage"
              value={newMedicine.dosage}
              onChangeText={(text) => setNewMedicine(prev => ({ ...prev, dosage: text }))}
              style={styles.input}
              mode="outlined"
              placeholder="e.g., 500mg"
            />
            <TextInput
              label="Frequency"
              value={newMedicine.frequency}
              onChangeText={(text) => setNewMedicine(prev => ({ ...prev, frequency: text }))}
              style={styles.input}
              mode="outlined"
              placeholder="e.g., daily, twice daily"
            />
            <TextInput
              label="Time"
              value={newMedicine.time}
              onChangeText={(text) => setNewMedicine(prev => ({ ...prev, time: text }))}
              style={styles.input}
              mode="outlined"
              placeholder="e.g., Morning, Evening"
            />
            <TextInput
              label="Notes"
              value={newMedicine.notes}
              onChangeText={(text) => setNewMedicine(prev => ({ ...prev, notes: text }))}
              style={styles.input}
              mode="outlined"
              multiline
              numberOfLines={2}
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowAddMedicine(false)}>Cancel</Button>
            <Button onPress={addMedicine}>Add</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>

      {/* Update Wound Modal */}
      <Portal>
        <Dialog visible={showUpdateWound} onDismiss={() => setShowUpdateWound(false)}>
          <Dialog.Title>Update Wound Progress</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Wound Area (cm²) *"
              value={woundUpdate.area}
              onChangeText={(text) => setWoundUpdate(prev => ({ ...prev, area: text }))}
              style={styles.input}
              mode="outlined"
              keyboardType="numeric"
            />
            
            <Text style={styles.inputLabel}>Pain Level (1-5)</Text>
            <View style={styles.radioGroup}>
              {[1, 2, 3, 4, 5].map((level) => (
                <View key={level} style={styles.radioItem}>
                  <RadioButton
                    value={level.toString()}
                    status={woundUpdate.painLevel === level.toString() ? 'checked' : 'unchecked'}
                    onPress={() => setWoundUpdate(prev => ({ ...prev, painLevel: level.toString() }))}
                  />
                  <Text style={styles.radioLabel}>{level}</Text>
                </View>
              ))}
            </View>
            
            <Text style={styles.inputLabel}>Redness</Text>
            <View style={styles.radioGroup}>
              {['none', 'mild', 'moderate', 'severe'].map((level) => (
                <View key={level} style={styles.radioItem}>
                  <RadioButton
                    value={level}
                    status={woundUpdate.redness === level ? 'checked' : 'unchecked'}
                    onPress={() => setWoundUpdate(prev => ({ ...prev, redness: level }))}
                  />
                  <Text style={styles.radioLabel}>{level}</Text>
                </View>
              ))}
            </View>
            
            <Text style={styles.inputLabel}>Swelling</Text>
            <View style={styles.radioGroup}>
              {['none', 'mild', 'moderate', 'severe'].map((level) => (
                <View key={level} style={styles.radioItem}>
                  <RadioButton
                    value={level}
                    status={woundUpdate.swelling === level ? 'checked' : 'unchecked'}
                    onPress={() => setWoundUpdate(prev => ({ ...prev, swelling: level }))}
                  />
                  <Text style={styles.radioLabel}>{level}</Text>
                </View>
              ))}
            </View>
            
            <TextInput
              label="Notes"
              value={woundUpdate.notes}
              onChangeText={(text) => setWoundUpdate(prev => ({ ...prev, notes: text }))}
              style={styles.input}
              mode="outlined"
              multiline
              numberOfLines={3}
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowUpdateWound(false)}>Cancel</Button>
            <Button onPress={addWoundUpdate}>Update</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>

      {/* Floating Action Button */}
      <FAB
        style={styles.fab}
        icon="plus"
        onPress={() => {
          Alert.alert(
            'Quick Action',
            'What would you like to do?',
            [
              { text: 'Add Medicine', onPress: () => setShowAddMedicine(true) },
              { text: 'Update Wound', onPress: () => setShowUpdateWound(true) },
              { text: 'Cancel', style: 'cancel' },
            ]
          );
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  headerCard: {
    margin: 15,
    elevation: 4,
  },
  progressContainer: {
    marginTop: 15,
  },
  progressLabel: {
    fontSize: 14,
    color: '#7f8c8d',
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
  dayProgressContainer: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#3498db',
  },
  dayProgressLabel: {
    fontSize: 14,
    color: '#2c3e50',
    marginBottom: 8,
    fontWeight: '600',
  },
  dayProgressBar: {
    height: 10,
    backgroundColor: '#e9ecef',
    borderRadius: 5,
    overflow: 'hidden',
    marginBottom: 8,
  },
  dayProgressFill: {
    height: '100%',
    borderRadius: 5,
  },
  dayProgressText: {
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  navigationCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  dayNavigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 15,
  },
  currentDayText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  woundCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  medicineCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  timelineCard: {
    margin: 15,
    marginTop: 0,
    elevation: 4,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  woundDetails: {
    marginTop: 10,
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
  chip: {
    alignSelf: 'center',
  },
  chipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
  notesContainer: {
    marginTop: 10,
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
  medicineItem: {
    marginBottom: 15,
    padding: 15,
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
  },
  medicineInfo: {
    marginBottom: 10,
  },
  medicineName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  medicineDetails: {
    fontSize: 14,
    color: '#7f8c8d',
    marginTop: 5,
  },
  medicineNotes: {
    fontSize: 12,
    color: '#95a5a6',
    marginTop: 5,
    fontStyle: 'italic',
  },
  medicineActions: {
    alignItems: 'flex-end',
  },
  takenInfo: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#e8f5e8',
    borderRadius: 5,
  },
  takenText: {
    fontSize: 12,
    color: '#27ae60',
    fontWeight: '600',
  },
  takenNotes: {
    fontSize: 12,
    color: '#27ae60',
    marginTop: 5,
  },
  timeline: {
    flexDirection: 'row',
    paddingVertical: 10,
  },
  timelineItem: {
    alignItems: 'center',
    marginRight: 20,
    minWidth: 80,
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#3498db',
    marginBottom: 5,
  },
  timelineDay: {
    fontSize: 12,
    color: '#7f8c8d',
    marginBottom: 5,
  },
  timelineArea: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  timelineChip: {
    alignSelf: 'center',
  },
  timelineChipText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 10,
  },
  timelineChange: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 2,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  emptyText: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 15,
    textAlign: 'center',
  },
  input: {
    marginBottom: 15,
  },
  inputLabel: {
    fontSize: 16,
    color: '#2c3e50',
    marginBottom: 10,
    marginTop: 10,
  },
  radioGroup: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 15,
  },
  radioItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 20,
    marginBottom: 10,
  },
  radioLabel: {
    fontSize: 14,
    color: '#2c3e50',
    marginLeft: 5,
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: '#667eea',
  },
});
