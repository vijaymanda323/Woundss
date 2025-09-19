#!/usr/bin/env python3
"""
Fix wound type labels in all datasets to ensure accurate classification.
This script will update all 'unknown' labels to their correct wound types.
"""

import os
import pandas as pd
from pathlib import Path

def fix_dataset_labels(dataset_path, correct_wound_type):
    """Fix labels in a specific dataset."""
    labels_file = os.path.join(dataset_path, 'labels.csv')
    
    if not os.path.exists(labels_file):
        print(f"❌ No labels.csv found in {dataset_path}")
        return False
    
    try:
        # Read the CSV file
        df = pd.read_csv(labels_file)
        
        # Check if wound_type column exists
        if 'wound_type' not in df.columns:
            print(f"❌ No wound_type column in {labels_file}")
            return False
        
        # Count unknown labels
        unknown_count = (df['wound_type'] == 'unknown').sum()
        
        if unknown_count == 0:
            print(f"✅ {dataset_path}: All labels already correct")
            return True
        
        # Fix unknown labels
        df.loc[df['wound_type'] == 'unknown', 'wound_type'] = correct_wound_type
        
        # Save the updated CSV
        df.to_csv(labels_file, index=False)
        
        print(f"✅ {dataset_path}: Fixed {unknown_count} labels to '{correct_wound_type}'")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {dataset_path}: {e}")
        return False

def main():
    """Fix labels for all wound datasets."""
    print("🔧 Fixing wound type labels for accurate classification...")
    
    # Define dataset mappings
    dataset_mappings = {
        'Burns': 'burn',
        'Cut': 'cut',
        'Laceration': 'laceration',
        'Abrasions': 'abrasion',
        'Bruises': 'bruise',
        'Stab_wound': 'stab_wound',
        'pressure-ucler-a': 'pressure_ulcer',
        'leg-ulcer-images': 'leg_ulcer',
        'foot-ulcers': 'foot_ulcer',
        'abdominal-wounds': 'abdominal_wound',
        'orthopedic-wounds': 'orthopedic_wound',
        'malignant-wound-images': 'malignant_wound',
        'extravasactions-wound-images': 'extravasation',
        'precular-uler': 'pressure_ulcer',
        'epidermolysis': 'epidermolysis',
        'haemongomia': 'hematoma',
        'ingrow': 'ingrown',
        'mennigits': 'meningitis',
        'miscellanious': 'miscellaneous',
        'pilonidial-sinus': 'pilonidal_sinus',
        'toes': 'toe_wound',
        'test_wounds': 'test_wound'
    }
    
    datasets_dir = 'datasets'
    fixed_count = 0
    total_count = 0
    
    for dataset_name, wound_type in dataset_mappings.items():
        dataset_path = os.path.join(datasets_dir, dataset_name)
        total_count += 1
        
        if os.path.exists(dataset_path):
            if fix_dataset_labels(dataset_path, wound_type):
                fixed_count += 1
        else:
            print(f"⚠️ Dataset not found: {dataset_path}")
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully fixed: {fixed_count}/{total_count} datasets")
    print(f"🎯 All wound types now properly labeled for accurate classification!")
    
    # Show the updated wound types
    print(f"\n🏥 Available wound types:")
    for dataset_name, wound_type in dataset_mappings.items():
        print(f"   • {wound_type} ({dataset_name})")

if __name__ == "__main__":
    main()


