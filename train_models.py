#!/usr/bin/env python3
"""
Direct Model Training Script
===========================

Train models directly without needing the Flask server.
"""

import sys
from pathlib import Path
import subprocess

def train_model_direct(dataset_name, epochs=50):
    """Train model directly using the app.py functions."""
    try:
        # Import the training function from app.py
        sys.path.append('.')
        from app import train_model, Path as AppPath
        
        print(f"🚀 Training model for dataset: {dataset_name}")
        print(f"📊 Epochs: {epochs}")
        print("-" * 50)
        
        # Train the model
        dataset_path = AppPath("datasets") / dataset_name
        success = train_model(dataset_path, epochs)
        
        if success:
            print(f"✅ Training completed successfully for {dataset_name}!")
            return True
        else:
            print(f"❌ Training failed for {dataset_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error training {dataset_name}: {e}")
        return False

def main():
    """Train all available datasets."""
    
    # List of your datasets
    datasets = [
        "Abrasions",
        "Bruises", 
        "Burns",
        "Cut",
        "ingrow",
        "Laceration",
        "Stab_wound",
        "test_wounds"
    ]
    
    print("🚀 Wound Model Training")
    print("=" * 50)
    print(f"📊 Found {len(datasets)} datasets to train")
    print()
    
    success_count = 0
    
    for i, dataset_name in enumerate(datasets, 1):
        print(f"📁 Training {i}/{len(datasets)}: {dataset_name}")
        
        # Use fewer epochs for faster training
        epochs = 30 if dataset_name == "test_wounds" else 50
        
        if train_model_direct(dataset_name, epochs):
            success_count += 1
        
        print()
    
    print("📊 Training Summary:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {len(datasets) - success_count}")
    
    if success_count > 0:
        print("\n🎉 Models are ready!")
        print("You can now analyze new wound images using the trained models.")
    
    return success_count > 0

if __name__ == "__main__":
    main()





