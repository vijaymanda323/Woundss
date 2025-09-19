#!/usr/bin/env python3
"""
React Native Wound Healing App Launcher
======================================

Launches the React Native app and provides instructions for different platforms.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_expo_cli():
    """Check if Expo CLI is installed."""
    try:
        result = subprocess.run(['expo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Expo CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Expo CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Expo CLI not found")
        return False

def check_node():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found")
        return False

def install_dependencies():
    """Install npm dependencies."""
    print("📦 Installing dependencies...")
    try:
        result = subprocess.run(['npm', 'install'], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_expo_server():
    """Start the Expo development server."""
    print("🚀 Starting Expo development server...")
    try:
        # Start Expo server
        process = subprocess.Popen(['expo', 'start'])
        
        print("\n" + "="*60)
        print("🏥 WOUND HEALING TRACKER - REACT NATIVE APP")
        print("="*60)
        print()
        print("📱 PLATFORM OPTIONS:")
        print("   • Web: Press 'w' in the terminal")
        print("   • Android: Press 'a' in the terminal")
        print("   • iOS: Press 'i' in the terminal")
        print("   • Expo Go: Scan QR code with Expo Go app")
        print()
        print("🌐 ACCESS URLs:")
        print("   • Web: http://localhost:19006")
        print("   • Expo DevTools: http://localhost:19002")
        print()
        print("📱 EXPO GO APP:")
        print("   • Android: https://play.google.com/store/apps/details?id=host.exp.exponent")
        print("   • iOS: https://apps.apple.com/app/expo-go/id982107779")
        print()
        print("🔧 DEVELOPMENT:")
        print("   • Hot reloading enabled")
        print("   • Press 'r' to reload")
        print("   • Press 'm' to toggle menu")
        print("   • Press 'd' to open developer menu")
        print("   • Press Ctrl+C to stop")
        print()
        print("="*60)
        
        return process
    except Exception as e:
        print(f"❌ Failed to start Expo server: {e}")
        return None

def show_instructions():
    """Show detailed instructions for different platforms."""
    print("\n📋 DETAILED INSTRUCTIONS:")
    print("-" * 50)
    
    print("\n🌐 WEB DEVELOPMENT:")
    print("1. Press 'w' in the Expo terminal")
    print("2. App will open in your default browser")
    print("3. Full-featured web application")
    print("4. No device required")
    
    print("\n📱 ANDROID DEVELOPMENT:")
    print("1. Install Expo Go app on your Android device")
    print("2. Press 'a' in the Expo terminal")
    print("3. Scan QR code with Expo Go app")
    print("4. Or use Android emulator")
    
    print("\n🍎 iOS DEVELOPMENT:")
    print("1. Install Expo Go app on your iOS device")
    print("2. Press 'i' in the Expo terminal")
    print("3. Scan QR code with Expo Go app")
    print("4. Or use iOS simulator (macOS only)")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• Clear cache: expo start --clear")
    print("• Reset Metro: npx expo start --reset-cache")
    print("• Check Expo CLI: expo --version")
    print("• Check Node.js: node --version")

def main():
    """Main launcher function."""
    print("🏥 React Native Wound Healing App Launcher")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path('package.json').exists():
        print("❌ package.json not found. Please run this script from the project root directory.")
        return
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    if not check_node():
        print("❌ Node.js is required. Please install Node.js from https://nodejs.org/")
        return
    
    if not check_expo_cli():
        print("❌ Expo CLI is required. Install with: npm install -g @expo/cli")
        return
    
    print()
    
    # Install dependencies if needed
    if not Path('node_modules').exists():
        if not install_dependencies():
            return
    else:
        print("✅ Dependencies already installed")
    
    print()
    
    # Start Expo server
    process = start_expo_server()
    if not process:
        return
    
    # Show instructions
    show_instructions()
    
    try:
        # Wait for user to stop
        print("\n⏳ Server is running... Press Ctrl+C to stop")
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()




