#!/usr/bin/env python3
"""
Dataset Upload Helper Script
===========================

This script helps you upload wound datasets from your local folder
to the proper structure for training the ML models.

Usage:
    python upload_dataset.py --source_folder /path/to/your/images --dataset_name my_wounds
"""

import os
import shutil
import argparse
import csv
from pathlib import Path
import json

def create_dataset_structure(dataset_name: str):
    """Create the proper dataset directory structure."""
    dataset_path = Path("datasets") / dataset_name
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created dataset structure: {dataset_path}")
    return dataset_path, images_dir, masks_dir

def upload_images(source_folder: str, images_dir: Path, masks_dir: Path, 
                 dataset_name: str, create_labels: bool = True):
    """Upload images and create labels file."""
    source_path = Path(source_folder)
    
    if not source_path.exists():
        print(f"❌ Source folder not found: {source_folder}")
        return False
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No image files found in {source_folder}")
        return False
    
    print(f"📁 Found {len(image_files)} images in source folder")
    
    # Copy images and create labels
    labels_data = []
    copied_count = 0
    
    for i, img_file in enumerate(image_files):
        try:
            # Copy image
            dest_img_path = images_dir / img_file.name
            shutil.copy2(img_file, dest_img_path)
            
            # Look for corresponding mask
            mask_name = f"{img_file.stem}_mask{img_file.suffix}"
            mask_path = source_path / mask_name
            
            if not mask_path.exists():
                # Try alternative naming
                mask_path = source_path / f"{img_file.stem}{img_file.suffix}"
            
            if mask_path.exists():
                # Copy mask
                dest_mask_path = masks_dir / mask_name
                shutil.copy2(mask_path, dest_mask_path)
                print(f"✅ Copied {img_file.name} with mask")
            else:
                print(f"⚠️  No mask found for {img_file.name}")
                # Create placeholder mask (you can manually create masks later)
                dest_mask_path = masks_dir / mask_name
                shutil.copy2(img_file, dest_mask_path)  # Copy image as placeholder
            
            # Add to labels (you can edit this later)
            labels_data.append({
                'filename': img_file.name,
                'wound_type': 'unknown',  # Edit this manually
                'healing_time_category': 'moderate_healing',  # Edit this manually
                'days_to_cure': 30  # Edit this manually
            })
            
            copied_count += 1
            
        except Exception as e:
            print(f"❌ Error copying {img_file.name}: {e}")
    
    # Create labels CSV
    if create_labels and labels_data:
        labels_file = Path("datasets") / dataset_name / "labels.csv"
        with open(labels_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
            writer.writeheader()
            writer.writerows(labels_data)
        
        print(f"📝 Created labels file: {labels_file}")
        print("⚠️  Please edit the labels.csv file to specify correct wound types and healing times")
    
    print(f"✅ Successfully uploaded {copied_count} images to dataset '{dataset_name}'")
    return True

def interactive_label_editor(dataset_name: str):
    """Interactive tool to edit labels."""
    labels_file = Path("datasets") / dataset_name / "labels.csv"
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        return
    
    print(f"\n📝 Editing labels for dataset: {dataset_name}")
    print("Wound types: chronic, surgical, burn, diabetic, pressure_ulcer, trauma")
    print("Healing categories: fast_healing, moderate_healing, slow_healing, chronic_non_healing")
    print("Press Enter to skip editing a field")
    
    # Read current labels
    labels = []
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        labels = list(reader)
    
    # Edit each label
    for i, label in enumerate(labels):
        print(f"\n--- Image {i+1}/{len(labels)}: {label['filename']} ---")
        
        # Edit wound type
        new_type = input(f"Wound type [{label['wound_type']}]: ").strip()
        if new_type:
            label['wound_type'] = new_type
        
        # Edit healing category
        new_category = input(f"Healing category [{label['healing_time_category']}]: ").strip()
        if new_category:
            label['healing_time_category'] = new_category
        
        # Edit days to cure
        new_days = input(f"Days to cure [{label['days_to_cure']}]: ").strip()
        if new_days.isdigit():
            label['days_to_cure'] = int(new_days)
    
    # Save updated labels
    with open(labels_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
        writer.writeheader()
        writer.writerows(labels)
    
    print(f"✅ Updated labels saved to {labels_file}")

def main():
    parser = argparse.ArgumentParser(description='Upload wound dataset')
    parser.add_argument('--source_folder', required=True, help='Path to folder containing wound images')
    parser.add_argument('--dataset_name', required=True, help='Name for the dataset')
    parser.add_argument('--edit_labels', action='store_true', help='Edit labels interactively')
    parser.add_argument('--no_labels', action='store_true', help='Skip creating labels file')
    
    args = parser.parse_args()
    
    print("🚀 Wound Dataset Upload Tool")
    print("=" * 40)
    
    # Create dataset structure
    dataset_path, images_dir, masks_dir = create_dataset_structure(args.dataset_name)
    
    # Upload images
    success = upload_images(
        args.source_folder, 
        images_dir, 
        masks_dir, 
        args.dataset_name,
        create_labels=not args.no_labels
    )
    
    if not success:
        return
    
    # Edit labels if requested
    if args.edit_labels:
        interactive_label_editor(args.dataset_name)
    
    print(f"\n🎉 Dataset '{args.dataset_name}' is ready!")
    print(f"📁 Dataset location: {dataset_path}")
    print(f"📝 Labels file: {dataset_path}/labels.csv")
    print(f"🖼️  Images: {images_dir}")
    print(f"🎭 Masks: {masks_dir}")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Edit {dataset_path}/labels.csv to specify correct wound types")
    print(f"2. Create proper masks for each image (if not done)")
    print(f"3. Train the model: curl -X POST -F 'dataset_name={args.dataset_name}' http://localhost:5000/train")

if __name__ == "__main__":
    main()





