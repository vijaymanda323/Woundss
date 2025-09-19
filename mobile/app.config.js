/**
 * Expo app configuration
 * 
 * This file can be used to configure environment-specific settings
 * For API URL configuration, see src/utils/env.js
 */

export default {
  expo: {
    name: "Expo Flask Starter",
    slug: "expo-flask-starter",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "light",
    splash: {
      image: "./assets/splash.png",
      resizeMode: "contain",
      backgroundColor: "#ffffff"
    },
    assetBundlePatterns: [
      "**/*"
    ],
    ios: {
      supportsTablet: true
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#FFFFFF"
      }
    },
    web: {
      favicon: "./assets/favicon.png",
      bundler: "metro"
    },
    plugins: [
      "expo-router"
    ],
    // Environment-specific overrides can be added here
    extra: {
      // Add any environment variables here
      // apiUrl: process.env.API_URL || "http://localhost:5000"
    }
  }
};





