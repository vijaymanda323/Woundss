import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { Provider as PaperProvider } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import CameraScreen from './src/screens/CameraScreen';
import PhotoUploadScreen from './src/screens/PhotoUploadScreen';
import AnalysisScreen from './src/screens/AnalysisScreen';
import AnalysisResultsScreen from './src/screens/AnalysisResultsScreen';
import TreatmentPlanScreen from './src/screens/TreatmentPlanScreen';
import PatientInfoScreen from './src/screens/PatientInfoScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import ReportsScreen from './src/screens/ReportsScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import DoctorAppointmentScreen from './src/screens/DoctorAppointmentScreen';
import WoundQuestionnaireScreen from './src/screens/WoundQuestionnaireScreen';
import PatientTrackerScreen from './src/screens/PatientTrackerScreen';

// Import theme
import { theme } from './src/theme/theme';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <PaperProvider theme={theme}>
        <NavigationContainer>
          <StatusBar style="light" />
          <Stack.Navigator
            initialRouteName="Home"
            screenOptions={{
              headerStyle: {
                backgroundColor: '#667eea',
              },
              headerTintColor: '#fff',
              headerTitleStyle: {
                fontWeight: 'bold',
              },
            }}
          >
            <Stack.Screen 
              name="Home" 
              component={HomeScreen}
              options={{ title: 'Wound Healing Tracker' }}
            />
            <Stack.Screen 
              name="Camera" 
              component={CameraScreen}
              options={{ title: 'Take Photo' }}
            />
            <Stack.Screen 
              name="PhotoUpload" 
              component={PhotoUploadScreen}
              options={{ title: 'Upload Photo' }}
            />
            <Stack.Screen 
              name="PatientInfo" 
              component={PatientInfoScreen}
              options={{ title: 'Patient Information' }}
            />
            <Stack.Screen 
              name="Analysis" 
              component={AnalysisScreen}
              options={{ title: 'Analyzing Wound' }}
            />
            <Stack.Screen 
              name="AnalysisResults" 
              component={AnalysisResultsScreen}
              options={{ title: 'Analysis Results' }}
            />
            <Stack.Screen 
              name="TreatmentPlan" 
              component={TreatmentPlanScreen}
              options={{ title: 'Treatment Plan' }}
            />
            <Stack.Screen 
              name="Results" 
              component={ResultsScreen}
              options={{ title: 'Analysis Results' }}
            />
            <Stack.Screen 
              name="Reports" 
              component={ReportsScreen}
              options={{ title: 'Reports' }}
            />
            <Stack.Screen 
              name="History" 
              component={HistoryScreen}
              options={{ title: 'Patient History' }}
            />
            <Stack.Screen 
              name="DoctorAppointment" 
              component={DoctorAppointmentScreen}
              options={{ title: 'Book Doctor Appointment' }}
            />
            <Stack.Screen 
              name="WoundQuestionnaire" 
              component={WoundQuestionnaireScreen}
              options={{ title: 'Medical Questionnaire' }}
            />
            <Stack.Screen 
              name="PatientTracker" 
              component={PatientTrackerScreen}
              options={{ title: 'Patient Progress Tracker' }}
            />
          </Stack.Navigator>
        </NavigationContainer>
      </PaperProvider>
    </SafeAreaProvider>
  );
}
