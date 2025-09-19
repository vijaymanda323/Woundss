import React from 'react';
import { StatusBar } from 'expo-status-bar';
import AppNavigator from './src/navigation/AppNavigator';

/**
 * Main App component
 * 
 * This is the root component that sets up:
 * - Navigation container
 * - Status bar configuration
 * - Global app state (if needed)
 */

export default function App() {
  return (
    <>
      <StatusBar style="light" backgroundColor="#6366f1" />
      <AppNavigator />
    </>
  );
}





