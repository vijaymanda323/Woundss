#!/usr/bin/env python3
"""
Multiple Dataset Upload Tool
===========================

Upload multiple wound dataset types from different folders.
Supports batch uploading of chronic, surgical, burn, diabetic, etc. wound types.

Usage:
    python upload_multiple_datasets.py --config config.json
    python upload_multiple_datasets.py --interactive
"""

import os
import shutil
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List

class MultipleDatasetUploader:
    def __init__(self):
        self.datasets_config = {}
        self.uploaded_datasets = []
    
    def create_config_template(self):
        """Create a template configuration file."""
        config_template = {
            "datasets": [
                {
                    "name": "chronic_wounds",
                    "source_folder": "C:/path/to/chronic/wounds",
                    "wound_type": "chronic",
                    "healing_category": "slow_healing",
                    "default_days_to_cure": 60,
                    "description": "Chronic non-healing wounds"
                },
                {
                    "name": "surgical_wounds", 
                    "source_folder": "C:/path/to/surgical/wounds",
                    "wound_type": "surgical",
                    "healing_category": "fast_healing",
                    "default_days_to_cure": 7,
                    "description": "Post-surgical incisions"
                },
                {
                    "name": "burn_wounds",
                    "source_folder": "C:/path/to/burn/wounds", 
                    "wound_type": "burn",
                    "healing_category": "moderate_healing",
                    "default_days_to_cure": 21,
                    "description": "Thermal burn injuries"
                },
                {
                    "name": "diabetic_wounds",
                    "source_folder": "C:/path/to/diabetic/wounds",
                    "wound_type": "diabetic", 
                    "healing_category": "chronic_non_healing",
                    "default_days_to_cure": 90,
                    "description": "Diabetic foot ulcers"
                },
                {
                    "name": "pressure_ulcers",
                    "source_folder": "C:/path/to/pressure/wounds",
                    "wound_type": "pressure_ulcer",
                    "healing_category": "slow_healing", 
                    "default_days_to_cure": 45,
                    "description": "Pressure sores/bedsores"
                },
                {
                    "name": "trauma_wounds",
                    "source_folder": "C:/path/to/trauma/wounds",
                    "wound_type": "trauma",
                    "healing_category": "moderate_healing",
                    "default_days_to_cure": 14,
                    "description": "Accidental trauma wounds"
                }
            ],
            "settings": {
                "create_combined_dataset": True,
                "combined_dataset_name": "mixed_wounds",
                "validation_split": 0.2,
                "augment_data": True
            }
        }
        
        with open("datasets_config.json", "w") as f:
            json.dump(config_template, f, indent=2)
        
        print("✅ Created config template: datasets_config.json")
        print("📝 Please edit this file with your actual folder paths")
        return config_template
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                self.datasets_config = json.load(f)
            print(f"✅ Loaded configuration from {config_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    
    def interactive_config(self):
        """Create configuration interactively."""
        print("🚀 Interactive Dataset Configuration")
        print("=" * 50)
        
        datasets = []
        
        while True:
            print(f"\n📁 Dataset {len(datasets) + 1}:")
            
            name = input("Dataset name (e.g., chronic_wounds): ").strip()
            if not name:
                break
            
            source_folder = input("Source folder path: ").strip()
            if not source_folder:
                print("❌ Source folder is required")
                continue
            
            print("Wound types: chronic, surgical, burn, diabetic, pressure_ulcer, trauma")
            wound_type = input("Wound type: ").strip() or "unknown"
            
            print("Healing categories: fast_healing, moderate_healing, slow_healing, chronic_non_healing")
            healing_category = input("Healing category: ").strip() or "moderate_healing"
            
            days_input = input("Default days to cure: ").strip()
            days_to_cure = int(days_input) if days_input.isdigit() else 30
            
            description = input("Description (optional): ").strip()
            
            datasets.append({
                "name": name,
                "source_folder": source_folder,
                "wound_type": wound_type,
                "healing_category": healing_category,
                "default_days_to_cure": days_to_cure,
                "description": description
            })
            
            continue_upload = input("Add another dataset? (y/n): ").strip().lower()
            if continue_upload != 'y':
                break
        
        self.datasets_config = {
            "datasets": datasets,
            "settings": {
                "create_combined_dataset": True,
                "combined_dataset_name": "mixed_wounds",
                "validation_split": 0.2,
                "augment_data": True
            }
        }
        
        # Save config
        with open("datasets_config.json", "w") as f:
            json.dump(self.datasets_config, f, indent=2)
        
        print(f"✅ Configuration saved to datasets_config.json")
        return True
    
    def upload_single_dataset(self, dataset_config: Dict) -> bool:
        """Upload a single dataset."""
        name = dataset_config["name"]
        source_folder = dataset_config["source_folder"]
        wound_type = dataset_config["wound_type"]
        healing_category = dataset_config["healing_category"]
        default_days = dataset_config["default_days_to_cure"]
        
        print(f"\n📤 Uploading dataset: {name}")
        print(f"   Source: {source_folder}")
        print(f"   Type: {wound_type}")
        print(f"   Healing: {healing_category}")
        
        # Create dataset structure
        dataset_path = Path("datasets") / name
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Check source folder
        source_path = Path(source_folder)
        if not source_path.exists():
            print(f"❌ Source folder not found: {source_folder}")
            return False
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {source_folder}")
            return False
        
        print(f"   Found {len(image_files)} images")
        
        # Copy images and create labels
        labels_data = []
        copied_count = 0
        
        for img_file in image_files:
            try:
                # Copy image
                dest_img_path = images_dir / img_file.name
                shutil.copy2(img_file, dest_img_path)
                
                # Look for mask
                mask_name = f"{img_file.stem}_mask{img_file.suffix}"
                mask_path = source_path / mask_name
                
                if not mask_path.exists():
                    mask_path = source_path / f"{img_file.stem}{img_file.suffix}"
                
                if mask_path.exists():
                    dest_mask_path = masks_dir / mask_name
                    shutil.copy2(mask_path, dest_mask_path)
                else:
                    # Copy image as placeholder mask
                    dest_mask_path = masks_dir / mask_name
                    shutil.copy2(img_file, dest_mask_path)
                
                # Add to labels with dataset-specific defaults
                labels_data.append({
                    'filename': img_file.name,
                    'wound_type': wound_type,
                    'healing_time_category': healing_category,
                    'days_to_cure': default_days
                })
                
                copied_count += 1
                
            except Exception as e:
                print(f"   ❌ Error copying {img_file.name}: {e}")
        
        # Create labels CSV
        if labels_data:
            labels_file = dataset_path / "labels.csv"
            with open(labels_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
                writer.writeheader()
                writer.writerows(labels_data)
            
            print(f"   📝 Created labels: {labels_file}")
        
        print(f"   ✅ Uploaded {copied_count} images")
        self.uploaded_datasets.append(name)
        return True
    
    def create_combined_dataset(self):
        """Create a combined dataset from all uploaded datasets."""
        if not self.datasets_config.get("settings", {}).get("create_combined_dataset", False):
            return
        
        combined_name = self.datasets_config["settings"].get("combined_dataset_name", "mixed_wounds")
        print(f"\n🔄 Creating combined dataset: {combined_name}")
        
        # Create combined dataset structure
        combined_path = Path("datasets") / combined_name
        combined_images_dir = combined_path / "images"
        combined_masks_dir = combined_path / "masks"
        
        combined_images_dir.mkdir(parents=True, exist_ok=True)
        combined_masks_dir.mkdir(parents=True, exist_ok=True)
        
        all_labels = []
        
        # Copy from all uploaded datasets
        for dataset_name in self.uploaded_datasets:
            source_dataset_path = Path("datasets") / dataset_name
            source_images_dir = source_dataset_path / "images"
            source_masks_dir = source_dataset_path / "masks"
            source_labels_file = source_dataset_path / "labels.csv"
            
            if not source_images_dir.exists():
                continue
            
            print(f"   📁 Adding {dataset_name} to combined dataset")
            
            # Copy images and masks
            for img_file in source_images_dir.glob("*"):
                if img_file.is_file():
                    # Copy image
                    dest_img_path = combined_images_dir / f"{dataset_name}_{img_file.name}"
                    shutil.copy2(img_file, dest_img_path)
                    
                    # Copy corresponding mask
                    mask_file = source_masks_dir / img_file.name
                    if mask_file.exists():
                        dest_mask_path = combined_masks_dir / f"{dataset_name}_{img_file.name}"
                        shutil.copy2(mask_file, dest_mask_path)
            
            # Load and update labels
            if source_labels_file.exists():
                with open(source_labels_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row['filename'] = f"{dataset_name}_{row['filename']}"
                        all_labels.append(row)
        
        # Save combined labels
        if all_labels:
            combined_labels_file = combined_path / "labels.csv"
            with open(combined_labels_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
                writer.writeheader()
                writer.writerows(all_labels)
            
            print(f"   📝 Created combined labels: {combined_labels_file}")
            print(f"   📊 Total samples: {len(all_labels)}")
        
        print(f"   ✅ Combined dataset created: {combined_path}")
    
    def upload_all_datasets(self):
        """Upload all datasets from configuration."""
        if not self.datasets_config.get("datasets"):
            print("❌ No datasets configured")
            return False
        
        print(f"🚀 Starting upload of {len(self.datasets_config['datasets'])} datasets")
        print("=" * 60)
        
        success_count = 0
        
        for dataset_config in self.datasets_config["datasets"]:
            if self.upload_single_dataset(dataset_config):
                success_count += 1
        
        print(f"\n📊 Upload Summary:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {len(self.datasets_config['datasets']) - success_count}")
        
        if success_count > 0:
            self.create_combined_dataset()
            
            print(f"\n🎉 Upload completed!")
            print(f"📁 Uploaded datasets: {', '.join(self.uploaded_datasets)}")
            
            print(f"\n🚀 Next steps:")
            print(f"1. Review and edit labels.csv files if needed")
            print(f"2. Train individual models:")
            for dataset_name in self.uploaded_datasets:
                print(f"   curl -X POST -F 'dataset_name={dataset_name}' http://localhost:5000/train")
            
            if self.datasets_config.get("settings", {}).get("create_combined_dataset"):
                combined_name = self.datasets_config["settings"].get("combined_dataset_name", "mixed_wounds")
                print(f"3. Train combined model:")
                print(f"   curl -X POST -F 'dataset_name={combined_name}' http://localhost:5000/train")
        
        return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Upload multiple wound datasets')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--interactive', action='store_true', help='Interactive configuration')
    parser.add_argument('--create_template', action='store_true', help='Create configuration template')
    
    args = parser.parse_args()
    
    uploader = MultipleDatasetUploader()
    
    if args.create_template:
        uploader.create_config_template()
        return
    
    if args.interactive:
        if uploader.interactive_config():
            uploader.upload_all_datasets()
    elif args.config:
        if uploader.load_config(args.config):
            uploader.upload_all_datasets()
    else:
        print("❌ Please specify --config, --interactive, or --create_template")
        print("💡 Use --create_template to create a configuration template")

if __name__ == "__main__":
    main()





