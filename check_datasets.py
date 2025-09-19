#!/usr/bin/env python3
"""
Dataset Validation Tool
======================

Check if your uploaded datasets are ready for model training.
Validates folder structure, image files, masks, and labels.

Usage:
    python check_datasets.py
    python check_datasets.py --dataset_name chronic_wounds
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

class DatasetValidator:
    def __init__(self):
        self.datasets_dir = Path("datasets")
        self.validation_results = {}
    
    def check_dataset_structure(self, dataset_name: str) -> Dict:
        """Check if dataset has proper structure."""
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            return {
                "valid": False,
                "error": f"Dataset folder not found: {dataset_path}"
            }
        
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        labels_file = dataset_path / "labels.csv"
        
        result = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "valid": True,
            "issues": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check images directory
        if not images_dir.exists():
            result["valid"] = False
            result["issues"].append("Missing images/ directory")
        else:
            image_files = list(images_dir.glob("*"))
            image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
            result["stats"]["image_count"] = len(image_files)
            
            if len(image_files) == 0:
                result["valid"] = False
                result["issues"].append("No image files found in images/ directory")
        
        # Check masks directory
        if not masks_dir.exists():
            result["valid"] = False
            result["issues"].append("Missing masks/ directory")
        else:
            mask_files = list(masks_dir.glob("*"))
            mask_files = [f for f in mask_files if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
            result["stats"]["mask_count"] = len(mask_files)
            
            if len(mask_files) == 0:
                result["valid"] = False
                result["issues"].append("No mask files found in masks/ directory")
        
        # Check labels file
        if not labels_file.exists():
            result["warnings"].append("No labels.csv file found - will use default labels")
        else:
            try:
                with open(labels_file, 'r') as f:
                    reader = csv.DictReader(f)
                    labels = list(reader)
                    result["stats"]["label_count"] = len(labels)
                    
                    # Check required columns
                    required_columns = ['filename', 'wound_type', 'healing_time_category', 'days_to_cure']
                    if not all(col in reader.fieldnames for col in required_columns):
                        result["issues"].append(f"Missing required columns in labels.csv: {required_columns}")
                    
                    # Check wound types
                    wound_types = set(row['wound_type'] for row in labels)
                    valid_wound_types = {'chronic', 'surgical', 'burn', 'diabetic', 'pressure_ulcer', 'trauma', 'unknown'}
                    invalid_types = wound_types - valid_wound_types
                    if invalid_types:
                        result["warnings"].append(f"Unknown wound types found: {invalid_types}")
                    
                    # Check healing categories
                    healing_categories = set(row['healing_time_category'] for row in labels)
                    valid_categories = {'fast_healing', 'moderate_healing', 'slow_healing', 'chronic_non_healing'}
                    invalid_categories = healing_categories - valid_categories
                    if invalid_categories:
                        result["warnings"].append(f"Unknown healing categories found: {invalid_categories}")
                    
                    result["stats"]["unique_wound_types"] = list(wound_types)
                    result["stats"]["unique_healing_categories"] = list(healing_categories)
                    
            except Exception as e:
                result["issues"].append(f"Error reading labels.csv: {e}")
        
        # Check image-mask correspondence
        if images_dir.exists() and masks_dir.exists():
            image_files = [f.name for f in images_dir.glob("*") if f.is_file()]
            mask_files = [f.name for f in masks_dir.glob("*") if f.is_file()]
            
            missing_masks = []
            for img_file in image_files:
                mask_name = f"{Path(img_file).stem}_mask{Path(img_file).suffix}"
                if mask_name not in mask_files and img_file not in mask_files:
                    missing_masks.append(img_file)
            
            if missing_masks:
                result["warnings"].append(f"Missing masks for {len(missing_masks)} images")
                if len(missing_masks) <= 5:  # Show details for small numbers
                    result["warnings"].append(f"Missing masks: {missing_masks}")
        
        return result
    
    def check_all_datasets(self) -> Dict:
        """Check all datasets in the datasets directory."""
        if not self.datasets_dir.exists():
            return {
                "valid": False,
                "error": "Datasets directory not found"
            }
        
        dataset_folders = [d for d in self.datasets_dir.iterdir() if d.is_dir()]
        
        if not dataset_folders:
            return {
                "valid": False,
                "error": "No datasets found in datasets/ directory"
            }
        
        results = {
            "total_datasets": len(dataset_folders),
            "valid_datasets": 0,
            "invalid_datasets": 0,
            "datasets": {}
        }
        
        for dataset_folder in dataset_folders:
            dataset_name = dataset_folder.name
            result = self.check_dataset_structure(dataset_name)
            results["datasets"][dataset_name] = result
            
            if result["valid"]:
                results["valid_datasets"] += 1
            else:
                results["invalid_datasets"] += 1
        
        return results
    
    def print_validation_report(self, results: Dict):
        """Print a detailed validation report."""
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        print("🔍 Dataset Validation Report")
        print("=" * 50)
        print(f"📊 Total datasets: {results['total_datasets']}")
        print(f"✅ Valid datasets: {results['valid_datasets']}")
        print(f"❌ Invalid datasets: {results['invalid_datasets']}")
        print()
        
        for dataset_name, result in results["datasets"].items():
            print(f"📁 Dataset: {dataset_name}")
            print(f"   Path: {result['dataset_path']}")
            
            if result["valid"]:
                print("   Status: ✅ READY FOR TRAINING")
            else:
                print("   Status: ❌ NOT READY")
            
            # Print stats
            if "stats" in result:
                stats = result["stats"]
                if "image_count" in stats:
                    print(f"   Images: {stats['image_count']}")
                if "mask_count" in stats:
                    print(f"   Masks: {stats['mask_count']}")
                if "label_count" in stats:
                    print(f"   Labels: {stats['label_count']}")
                if "unique_wound_types" in stats:
                    print(f"   Wound types: {', '.join(stats['unique_wound_types'])}")
                if "unique_healing_categories" in stats:
                    print(f"   Healing categories: {', '.join(stats['unique_healing_categories'])}")
            
            # Print issues
            if result["issues"]:
                print("   ❌ Issues:")
                for issue in result["issues"]:
                    print(f"      • {issue}")
            
            # Print warnings
            if result["warnings"]:
                print("   ⚠️  Warnings:")
                for warning in result["warnings"]:
                    print(f"      • {warning}")
            
            print()
        
        # Training recommendations
        valid_datasets = [name for name, result in results["datasets"].items() if result["valid"]]
        
        if valid_datasets:
            print("🚀 Training Commands:")
            print("-" * 30)
            for dataset_name in valid_datasets:
                print(f"curl -X POST -F 'dataset_name={dataset_name}' \\")
                print(f"     -F 'epochs=100' \\")
                print(f"     http://localhost:5000/train")
                print()
        
        if results["invalid_datasets"] > 0:
            print("🔧 Fix Issues:")
            print("-" * 20)
            print("1. Ensure images/ and masks/ directories exist")
            print("2. Add image files (.jpg, .png, .bmp, .tiff)")
            print("3. Add corresponding mask files")
            print("4. Create labels.csv with proper format")
            print("5. Use upload tools to fix structure")

def main():
    parser = argparse.ArgumentParser(description='Validate wound datasets')
    parser.add_argument('--dataset_name', help='Check specific dataset')
    parser.add_argument('--list', action='store_true', help='List all datasets')
    
    args = parser.parse_args()
    
    validator = DatasetValidator()
    
    if args.list:
        # List all datasets
        if not validator.datasets_dir.exists():
            print("❌ Datasets directory not found")
            return
        
        dataset_folders = [d.name for d in validator.datasets_dir.iterdir() if d.is_dir()]
        
        if not dataset_folders:
            print("❌ No datasets found")
            return
        
        print("📁 Available datasets:")
        for dataset_name in dataset_folders:
            print(f"   • {dataset_name}")
        
        return
    
    if args.dataset_name:
        # Check specific dataset
        result = validator.check_dataset_structure(args.dataset_name)
        validator.print_validation_report({"datasets": {args.dataset_name: result}})
    else:
        # Check all datasets
        results = validator.check_all_datasets()
        validator.print_validation_report(results)

if __name__ == "__main__":
    main()





