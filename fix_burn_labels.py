#!/usr/bin/env python3
"""
Fix burn labels to match actual image filenames.
"""

import pandas as pd
from pathlib import Path

def fix_burn_labels():
    """Fix burn labels to match actual image filenames."""
    
    print("🔧 Fixing burn labels...")
    
    # Get all burn images
    burn_images_dir = Path('datasets/Burns/images')
    if not burn_images_dir.exists():
        print("❌ Burn images directory not found!")
        return False
    
    # Get all image files
    image_files = list(burn_images_dir.glob('*.jpg')) + list(burn_images_dir.glob('*.png'))
    print(f"📸 Found {len(image_files)} burn images")
    
    # Create new labels data
    labels_data = []
    
    for image_file in image_files:
        filename = image_file.name
        labels_data.append({
            'filename': filename,
            'wound_type': 'burn',
            'healing_time_category': 'moderate_healing',
            'days_to_cure': 30
        })
    
    # Create DataFrame
    df = pd.DataFrame(labels_data)
    
    # Save to CSV
    labels_file = Path('datasets/Burns/labels.csv')
    df.to_csv(labels_file, index=False)
    
    print(f"✅ Updated {labels_file} with {len(labels_data)} burn labels")
    print(f"📋 Sample labels:")
    print(df.head())
    
    return True

if __name__ == "__main__":
    fix_burn_labels()


