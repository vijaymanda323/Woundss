#!/usr/bin/env python3
"""
Create Test Dataset
==================

Create sample wound images and masks for testing the ML system.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_test_wound_image(size=(512, 512), wound_type="chronic"):
    """Create a test wound image."""
    # Create base image
    img = Image.new('RGB', size, color=(240, 220, 200))  # Skin-like color
    
    # Add some texture
    noise = np.random.randint(-20, 20, size + (3,))
    img_array = np.array(img) + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Draw wound
    draw = ImageDraw.Draw(img)
    
    if wound_type == "chronic":
        # Irregular chronic wound
        wound_points = [
            (200, 200), (250, 180), (300, 200), (320, 250), 
            (300, 300), (250, 320), (200, 300), (180, 250)
        ]
        draw.polygon(wound_points, fill=(120, 80, 60))  # Dark wound color
        draw.polygon(wound_points, outline=(100, 60, 40), width=2)
        
    elif wound_type == "surgical":
        # Linear surgical wound
        draw.line([(200, 200), (300, 300)], fill=(120, 80, 60), width=8)
        draw.line([(200, 200), (300, 300)], fill=(100, 60, 40), width=2)
        
    elif wound_type == "burn":
        # Burn wound
        draw.ellipse([(180, 180), (320, 320)], fill=(150, 100, 80))
        draw.ellipse([(180, 180), (320, 320)], outline=(120, 80, 60), width=3)
    
    return img

def create_test_mask(size=(512, 512), wound_type="chronic"):
    """Create a test mask for the wound."""
    mask = Image.new('L', size, color=0)  # Black background
    draw = ImageDraw.Draw(mask)
    
    if wound_type == "chronic":
        wound_points = [
            (200, 200), (250, 180), (300, 200), (320, 250), 
            (300, 300), (250, 320), (200, 300), (180, 250)
        ]
        draw.polygon(wound_points, fill=255)  # White wound area
        
    elif wound_type == "surgical":
        draw.line([(200, 200), (300, 300)], fill=255, width=8)
        
    elif wound_type == "burn":
        draw.ellipse([(180, 180), (320, 320)], fill=255)
    
    return mask

def create_test_dataset():
    """Create a complete test dataset."""
    print("🧪 Creating test dataset...")
    
    # Create dataset structure
    dataset_path = Path("datasets/test_wounds")
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test images
    wound_types = ["chronic", "surgical", "burn"]
    
    for i, wound_type in enumerate(wound_types):
        for j in range(3):  # 3 images per type
            # Create image
            img = create_test_wound_image(wound_type=wound_type)
            img_filename = f"{wound_type}_wound_{j+1:02d}.jpg"
            img_path = images_dir / img_filename
            img.save(img_path, "JPEG")
            
            # Create mask
            mask = create_test_mask(wound_type=wound_type)
            mask_filename = f"{wound_type}_wound_{j+1:02d}_mask.jpg"
            mask_path = masks_dir / mask_filename
            mask.save(mask_path, "JPEG")
            
            print(f"   ✅ Created {img_filename} with mask")
    
    # Create labels file
    labels_data = []
    for wound_type in wound_types:
        for j in range(3):
            filename = f"{wound_type}_wound_{j+1:02d}.jpg"
            
            # Set healing time based on wound type
            healing_mapping = {
                "chronic": ("slow_healing", 60),
                "surgical": ("fast_healing", 7),
                "burn": ("moderate_healing", 21)
            }
            
            healing_category, days = healing_mapping[wound_type]
            
            labels_data.append({
                "filename": filename,
                "wound_type": wound_type,
                "healing_time_category": healing_category,
                "days_to_cure": days
            })
    
    # Write labels CSV
    import csv
    labels_file = dataset_path / "labels.csv"
    with open(labels_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
        writer.writeheader()
        writer.writerows(labels_data)
    
    print(f"📝 Created labels file: {labels_file}")
    print(f"🎉 Test dataset created: {dataset_path}")
    print(f"   📊 Total images: {len(labels_data)}")
    print(f"   🏷️  Wound types: {', '.join(wound_types)}")
    
    return dataset_path

if __name__ == "__main__":
    create_test_dataset()





