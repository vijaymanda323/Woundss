#!/usr/bin/env python3
"""
Simple GUI Dataset Upload Tool
=============================

A simple GUI to upload wound datasets from your local folder.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv
from pathlib import Path

class DatasetUploader:
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Dataset Upload Tool")
        self.root.geometry("600x500")
        
        # Variables
        self.source_folder = tk.StringVar()
        self.dataset_name = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="🚀 Wound Dataset Upload Tool", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Source folder selection
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(folder_frame, text="Source Folder:", font=("Arial", 12)).pack(anchor="w")
        
        folder_select_frame = tk.Frame(folder_frame)
        folder_select_frame.pack(fill="x", pady=5)
        
        tk.Entry(folder_select_frame, textvariable=self.source_folder, 
                font=("Arial", 10), width=50).pack(side="left", fill="x", expand=True)
        
        tk.Button(folder_select_frame, text="Browse", 
                 command=self.browse_folder, font=("Arial", 10)).pack(side="right", padx=(10, 0))
        
        # Dataset name
        name_frame = tk.Frame(self.root)
        name_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(name_frame, text="Dataset Name:", font=("Arial", 12)).pack(anchor="w")
        tk.Entry(name_frame, textvariable=self.dataset_name, 
                font=("Arial", 10), width=50).pack(fill="x", pady=5)
        
        # Instructions
        instructions = """
📋 Instructions:
1. Select the folder containing your wound images
2. Enter a name for your dataset
3. Click 'Upload Dataset' to copy files
4. Edit the labels.csv file to specify wound types and healing times
5. Train your model using the API

📁 Expected file structure:
- wound_001.jpg (wound image)
- wound_001_mask.jpg (corresponding mask)
- wound_002.jpg, wound_002_mask.jpg, etc.

🏷️ Wound Types: chronic, surgical, burn, diabetic, pressure_ulcer, trauma
⏱️ Healing Categories: fast_healing, moderate_healing, slow_healing, chronic_non_healing
        """
        
        instructions_label = tk.Label(self.root, text=instructions, 
                                    font=("Arial", 9), justify="left")
        instructions_label.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Upload button
        upload_button = tk.Button(self.root, text="📤 Upload Dataset", 
                                command=self.upload_dataset, 
                                font=("Arial", 12, "bold"),
                                bg="#4CAF50", fg="white", height=2)
        upload_button.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill="x")
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to upload", 
                                   font=("Arial", 10))
        self.status_label.pack(pady=5)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing wound images")
        if folder:
            self.source_folder.set(folder)
    
    def upload_dataset(self):
        if not self.source_folder.get():
            messagebox.showerror("Error", "Please select a source folder")
            return
        
        if not self.dataset_name.get():
            messagebox.showerror("Error", "Please enter a dataset name")
            return
        
        try:
            self.progress.start()
            self.status_label.config(text="Uploading dataset...")
            self.root.update()
            
            success = self.upload_images()
            
            self.progress.stop()
            
            if success:
                self.status_label.config(text="✅ Dataset uploaded successfully!")
                messagebox.showinfo("Success", 
                    f"Dataset '{self.dataset_name.get()}' uploaded successfully!\n\n"
                    f"Location: datasets/{self.dataset_name.get()}/\n\n"
                    f"Next steps:\n"
                    f"1. Edit labels.csv to specify wound types\n"
                    f"2. Train model: curl -X POST -F 'dataset_name={self.dataset_name.get()}' http://localhost:5000/train")
            else:
                self.status_label.config(text="❌ Upload failed")
                
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="❌ Upload failed")
            messagebox.showerror("Error", f"Upload failed: {str(e)}")
    
    def upload_images(self):
        source_path = Path(self.source_folder.get())
        dataset_name = self.dataset_name.get()
        
        # Create dataset structure
        dataset_path = Path("datasets") / dataset_name
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            messagebox.showerror("Error", f"No image files found in {source_path}")
            return False
        
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
                
                # Add to labels
                labels_data.append({
                    'filename': img_file.name,
                    'wound_type': 'unknown',
                    'healing_time_category': 'moderate_healing',
                    'days_to_cure': 30
                })
                
                copied_count += 1
                
            except Exception as e:
                print(f"Error copying {img_file.name}: {e}")
        
        # Create labels CSV
        if labels_data:
            labels_file = dataset_path / "labels.csv"
            with open(labels_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
                writer.writeheader()
                writer.writerows(labels_data)
        
        return True

def main():
    root = tk.Tk()
    app = DatasetUploader(root)
    root.mainloop()

if __name__ == "__main__":
    main()





