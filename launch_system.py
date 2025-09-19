#!/usr/bin/env python3
"""
Wound Healing System Launcher
============================

Launches both the main API server and the enhanced UI server.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def check_port(port):
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_server(script_name, port, description):
    """Start a server script."""
    print(f"🚀 Starting {description} on port {port}...")
    
    if not check_port(port):
        print(f"⚠️  Port {port} is already in use. Skipping {description}.")
        return None
    
    try:
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment to see if it starts successfully
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {description} started successfully on port {port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start {description}:")
            print(f"   stdout: {stdout.decode()}")
            print(f"   stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {description}: {e}")
        return None

def main():
    """Main launcher function."""
    print("🏥 Wound Healing System Launcher")
    print("=" * 50)
    print()
    
    # Check if required files exist
    required_files = [
        'app.py',
        'enhanced_ui_server.py',
        'advanced_wound_ui.html'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print()
        print("Please make sure all files are in the current directory.")
        return
    
    print("📋 Starting servers...")
    print()
    
    # Start the main API server
    api_process = start_server('app.py', 5000, 'Main API Server')
    
    # Start the enhanced UI server
    ui_process = start_server('enhanced_ui_server.py', 5001, 'Enhanced UI Server')
    
    print()
    print("🌐 Access URLs:")
    print("   Main API: http://localhost:5000")
    print("   Enhanced UI: http://localhost:5001")
    print("   Simple UI: Open 'advanced_wound_ui.html' in your browser")
    print()
    
    if api_process or ui_process:
        print("✅ System is running! Press Ctrl+C to stop all servers.")
        print()
        
        try:
            # Wait for user to stop
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print()
            print("🛑 Stopping servers...")
            
            if api_process:
                api_process.terminate()
                print("   ✅ Main API Server stopped")
            
            if ui_process:
                ui_process.terminate()
                print("   ✅ Enhanced UI Server stopped")
            
            print("👋 Goodbye!")
    else:
        print("❌ No servers were started successfully.")
        print("   Please check the error messages above and try again.")

if __name__ == "__main__":
    main()




