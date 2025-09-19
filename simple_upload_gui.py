#!/usr/bin/env python3
"""
Simple Upload GUI - Fixed Version
=================================

A simple, reliable GUI for uploading wound datasets.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv
from pathlib import Path

class SimpleUploadGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Wound Dataset Upload")
        self.root.geometry("500x400")
        
        # Variables
        self.source_folder = tk.StringVar()
        self.dataset_name = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="🚀 Simple Wound Dataset Upload", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Instructions
        instructions = """
📋 Instructions:
1. Select folder containing your wound images
2. Enter a name for your dataset
3. Click 'UPLOAD DATASET' button
4. Check the results and train your model
        """
        
        instructions_label = tk.Label(self.root, text=instructions, 
                                    font=("Arial", 10), justify="left")
        instructions_label.pack(pady=10, padx=20)
        
        # Source folder selection
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(folder_frame, text="📁 Source Folder:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        folder_select_frame = tk.Frame(folder_frame)
        folder_select_frame.pack(fill="x", pady=5)
        
        tk.Entry(folder_select_frame, textvariable=self.source_folder, 
                font=("Arial", 10), width=50).pack(side="left", fill="x", expand=True)
        
        tk.Button(folder_select_frame, text="Browse", 
                 command=self.browse_folder, font=("Arial", 10), bg="#2196F3", fg="white").pack(side="right", padx=(10, 0))
        
        # Dataset name
        name_frame = tk.Frame(self.root)
        name_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(name_frame, text="📝 Dataset Name:", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Entry(name_frame, textvariable=self.dataset_name, 
                font=("Arial", 12), width=30).pack(fill="x", pady=5)
        
        # Upload button - BIG AND CLEAR
        upload_button = tk.Button(self.root, text="📤 UPLOAD DATASET", 
                                command=self.upload_dataset,
                                font=("Arial", 14, "bold"),
                                bg="#4CAF50", fg="white", height=3,
                                relief="raised", bd=5)
        upload_button.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill="x")
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to upload", 
                                   font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Results text area
        self.results_text = tk.Text(self.root, height=8, width=60, font=("Arial", 9))
        self.results_text.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing wound images")
        if folder:
            self.source_folder.set(folder)
            self.log_message(f"Selected folder: {folder}")
    
    def log_message(self, message):
        """Add message to results text area."""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
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
            self.log_message("🚀 Starting upload...")
            self.root.update()
            
            success = self.upload_images()
            
            self.progress.stop()
            
            if success:
                self.status_label.config(text="✅ Dataset uploaded successfully!")
                self.log_message("✅ Upload completed successfully!")
                
                messagebox.showinfo("Success", 
                    f"Dataset '{self.dataset_name.get()}' uploaded successfully!\n\n"
                    f"Location: datasets/{self.dataset_name.get()}/\n\n"
                    f"Next steps:\n"
                    f"1. Edit labels.csv to specify wound types\n"
                    f"2. Train model: curl -X POST -F 'dataset_name={self.dataset_name.get()}' http://localhost:5000/train")
            else:
                self.status_label.config(text="❌ Upload failed")
                self.log_message("❌ Upload failed!")
                
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="❌ Upload failed")
            self.log_message(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Upload failed: {str(e)}")
    
    def upload_images(self):
        source_path = Path(self.source_folder.get())
        dataset_name = self.dataset_name.get()
        
        self.log_message(f"📁 Source folder: {source_path}")
        self.log_message(f"📝 Dataset name: {dataset_name}")
        
        # Create dataset structure
        dataset_path = Path("datasets") / dataset_name
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_message(f"📂 Created directories: {dataset_path}")
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        self.log_message(f"🔍 Found {len(image_files)} image files")
        
        if not image_files:
            self.log_message("❌ No image files found!")
            return False
        
        # Copy images and create labels
        labels_data = []
        copied_count = 0
        
        for img_file in image_files:
            try:
                self.log_message(f"📷 Processing: {img_file.name}")
                
                # Copy image
                dest_img_path = images_dir / img_file.name
                shutil.copy2(img_file, dest_img_path)
                
                # Look for mask
                mask_name = f"{img_file.stem}_mask{img_file.suffix}"
                mask_path = source_path / mask_name
                
                if not mask_path.exists():
                    # Try alternative naming
                    mask_path = source_path / f"{img_file.stem}{img_file.suffix}"
                
                if mask_path.exists():
                    dest_mask_path = masks_dir / mask_name
                    shutil.copy2(mask_path, dest_mask_path)
                    self.log_message(f"   ✅ Found mask: {mask_path.name}")
                else:
                    # Copy image as placeholder mask
                    dest_mask_path = masks_dir / mask_name
                    shutil.copy2(img_file, dest_mask_path)
                    self.log_message(f"   ⚠️  No mask found, using image as placeholder")
                
                # Add to labels
                labels_data.append({
                    'filename': img_file.name,
                    'wound_type': 'unknown',
                    'healing_time_category': 'moderate_healing',
                    'days_to_cure': 30
                })
                
                copied_count += 1
                
            except Exception as e:
                self.log_message(f"   ❌ Error copying {img_file.name}: {e}")
        
        # Create labels CSV
        if labels_data:
            labels_file = dataset_path / "labels.csv"
            with open(labels_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'wound_type', 'healing_time_category', 'days_to_cure'])
                writer.writeheader()
                writer.writerows(labels_data)
            
            self.log_message(f"📝 Created labels file: {labels_file}")
        
        self.log_message(f"✅ Successfully uploaded {copied_count} images")
        return copied_count > 0

def main():
    root = tk.Tk()
    app = SimpleUploadGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()





