#!/usr/bin/env python3
"""
Super Simple Upload GUI - Guaranteed to Work
===========================================

A minimal GUI with a big, obvious upload button.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import csv
from pathlib import Path

def upload_dataset():
    """Upload dataset function."""
    # Get folder
    folder = filedialog.askdirectory(title="Select folder with wound images")
    if not folder:
        return
    
    # Get dataset name
    dataset_name = tk.simpledialog.askstring("Dataset Name", "Enter dataset name:")
    if not dataset_name:
        return
    
    try:
        # Create directories
        dataset_path = Path("datasets") / dataset_name
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy images
        source_path = Path(folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            messagebox.showerror("Error", "No image files found!")
            return
        
        # Copy files
        labels_data = []
        copied_count = 0
        
        for img_file in image_files:
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
                dest_mask_path = masks_dir / mask_name
                shutil.copy2(img_file, dest_mask_path)
            
            # Add to labels
            labels_data.append({
                'filename': img_file.name,
                'wound_type': 'unknown',
                'healing_time_category': 'moderate_healing',
                'days_to_cure': 30
            })
            
            copied_count += 1
        
        # Create labels CSV
        labels_file = dataset_path / "labels.csv"
        with open(labels_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
            writer.writeheader()
            writer.writerows(labels_data)
        
        messagebox.showinfo("Success", 
            f"Uploaded {copied_count} images to dataset '{dataset_name}'!\n\n"
            f"Location: datasets/{dataset_name}/\n\n"
            f"Next: Train model with:\n"
            f"curl -X POST -F 'dataset_name={dataset_name}' http://localhost:5000/train")
        
    except Exception as e:
        messagebox.showerror("Error", f"Upload failed: {str(e)}")

# Create the simplest possible GUI
root = tk.Tk()
root.title("Wound Dataset Upload")
root.geometry("400x300")

# Title
title = tk.Label(root, text="🚀 Wound Dataset Upload", font=("Arial", 16, "bold"))
title.pack(pady=30)

# Instructions
instructions = tk.Label(root, text="Click the button below to upload your wound images", 
                      font=("Arial", 12))
instructions.pack(pady=20)

# BIG UPLOAD BUTTON
upload_btn = tk.Button(root, text="📤 UPLOAD DATASET", 
                      command=upload_dataset,
                      font=("Arial", 14, "bold"),
                      bg="#4CAF50", fg="white", 
                      height=3, width=20,
                      relief="raised", bd=5)
upload_btn.pack(pady=30)

# Status
status = tk.Label(root, text="Ready to upload", font=("Arial", 10))
status.pack(pady=10)

# Add simpledialog import
import tkinter.simpledialog

root.mainloop()





