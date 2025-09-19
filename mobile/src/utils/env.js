import Constants from 'expo-constants';
import { Platform } from 'react-native';

/**
 * Environment configuration for API base URL
 * 
 * To change the API URL for different environments:
 * 1. Web: Use http://localhost:5000
 * 2. Android Emulator: Use http://10.0.2.2:5000
 * 3. iOS Simulator: Use http://localhost:5000
 * 4. Real Device: Use http://YOUR_LOCAL_IP:5000
 */

const getApiBaseUrl = () => {
  // Check if we're running in Expo Go or development
  const isDev = __DEV__;
  
  if (!isDev) {
    // Production - replace with your production API URL
    return 'https://your-production-api.com';
  }
  
  // Development URLs based on platform
  if (Platform.OS === 'web') {
    return 'http://localhost:5000';
  } else if (Platform.OS === 'android') {
    // Android emulator - use 10.0.2.2 to access host machine
    return 'http://10.0.2.2:5000';
  } else if (Platform.OS === 'ios') {
    // iOS simulator - localhost works
    return 'http://localhost:5000';
  }
  
  // Fallback
  return 'http://localhost:5000';
};

export const API_BASE_URL = getApiBaseUrl();

// Log the API URL for debugging
if (__DEV__) {
  console.log(`API Base URL: ${API_BASE_URL}`);
  console.log(`Platform: ${Platform.OS}`);
}






