#!/usr/bin/env python3
"""
Multiple Dataset Upload GUI Tool
================================

GUI for uploading multiple wound dataset types from different folders.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import csv
import json
from pathlib import Path

class MultipleDatasetUploaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multiple Wound Dataset Upload Tool")
        self.root.geometry("800x700")
        
        self.datasets = []
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="🚀 Multiple Wound Dataset Upload Tool", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Instructions
        instructions = """
📋 Upload multiple wound types from different folders:
• Chronic wounds, surgical wounds, burn wounds, diabetic wounds, etc.
• Each dataset will be organized separately
• Optionally create a combined mixed dataset
• Automatic label generation with wound type classification
        """
        
        instructions_label = tk.Label(self.root, text=instructions, 
                                    font=("Arial", 10), justify="left")
        instructions_label.pack(pady=10, padx=20)
        
        # Dataset list frame
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        tk.Label(list_frame, text="📁 Configured Datasets:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # Listbox for datasets
        self.dataset_listbox = tk.Listbox(list_frame, height=8, font=("Arial", 10))
        self.dataset_listbox.pack(fill="both", expand=True, pady=5)
        
        # Scrollbar for listbox
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.dataset_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.dataset_listbox.yview)
        
        # Dataset configuration frame
        config_frame = tk.Frame(self.root)
        config_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(config_frame, text="➕ Add Dataset:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # Dataset name
        name_frame = tk.Frame(config_frame)
        name_frame.pack(fill="x", pady=5)
        
        tk.Label(name_frame, text="Dataset Name:", font=("Arial", 10)).pack(side="left")
        self.name_var = tk.StringVar()
        tk.Entry(name_frame, textvariable=self.name_var, font=("Arial", 10), width=20).pack(side="left", padx=(10, 0))
        
        # Source folder
        folder_frame = tk.Frame(config_frame)
        folder_frame.pack(fill="x", pady=5)
        
        tk.Label(folder_frame, text="Source Folder:", font=("Arial", 10)).pack(side="left")
        self.folder_var = tk.StringVar()
        tk.Entry(folder_frame, textvariable=self.folder_var, font=("Arial", 10), width=40).pack(side="left", padx=(10, 0))
        tk.Button(folder_frame, text="Browse", command=self.browse_folder, font=("Arial", 9)).pack(side="left", padx=(10, 0))
        
        # Wound type
        type_frame = tk.Frame(config_frame)
        type_frame.pack(fill="x", pady=5)
        
        tk.Label(type_frame, text="Wound Type:", font=("Arial", 10)).pack(side="left")
        self.type_var = tk.StringVar(value="chronic")
        type_combo = ttk.Combobox(type_frame, textvariable=self.type_var, 
                                 values=["chronic", "surgical", "burn", "diabetic", "pressure_ulcer", "trauma"],
                                 font=("Arial", 10), width=15)
        type_combo.pack(side="left", padx=(10, 0))
        
        # Healing category
        healing_frame = tk.Frame(config_frame)
        healing_frame.pack(fill="x", pady=5)
        
        tk.Label(healing_frame, text="Healing Category:", font=("Arial", 10)).pack(side="left")
        self.healing_var = tk.StringVar(value="moderate_healing")
        healing_combo = ttk.Combobox(healing_frame, textvariable=self.healing_var,
                                   values=["fast_healing", "moderate_healing", "slow_healing", "chronic_non_healing"],
                                   font=("Arial", 10), width=15)
        healing_combo.pack(side="left", padx=(10, 0))
        
        # Days to cure
        days_frame = tk.Frame(config_frame)
        days_frame.pack(fill="x", pady=5)
        
        tk.Label(days_frame, text="Days to Cure:", font=("Arial", 10)).pack(side="left")
        self.days_var = tk.StringVar(value="30")
        tk.Entry(days_frame, textvariable=self.days_var, font=("Arial", 10), width=10).pack(side="left", padx=(10, 0))
        
        # Add dataset button
        add_button = tk.Button(config_frame, text="➕ Add Dataset", command=self.add_dataset,
                              font=("Arial", 10), bg="#2196F3", fg="white")
        add_button.pack(pady=10)
        
        # Remove dataset button
        remove_button = tk.Button(config_frame, text="🗑️ Remove Selected", command=self.remove_dataset,
                                font=("Arial", 10), bg="#f44336", fg="white")
        remove_button.pack(pady=5)
        
        # Combined dataset option
        combined_frame = tk.Frame(self.root)
        combined_frame.pack(pady=10, padx=20, fill="x")
        
        self.create_combined_var = tk.BooleanVar(value=True)
        tk.Checkbutton(combined_frame, text="Create combined mixed dataset", 
                      variable=self.create_combined_var, font=("Arial", 10)).pack(anchor="w")
        
        combined_name_frame = tk.Frame(combined_frame)
        combined_name_frame.pack(fill="x", pady=5)
        
        tk.Label(combined_name_frame, text="Combined Dataset Name:", font=("Arial", 10)).pack(side="left")
        self.combined_name_var = tk.StringVar(value="mixed_wounds")
        tk.Entry(combined_name_frame, textvariable=self.combined_name_var, 
                font=("Arial", 10), width=20).pack(side="left", padx=(10, 0))
        
        # Upload button
        upload_button = tk.Button(self.root, text="📤 Upload All Datasets", 
                                command=self.upload_all_datasets,
                                font=("Arial", 14, "bold"),
                                bg="#4CAF50", fg="white", height=2)
        upload_button.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill="x")
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to upload datasets", 
                                   font=("Arial", 10))
        self.status_label.pack(pady=5)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing wound images")
        if folder:
            self.folder_var.set(folder)
    
    def add_dataset(self):
        name = self.name_var.get().strip()
        folder = self.folder_var.get().strip()
        wound_type = self.type_var.get()
        healing_category = self.healing_var.get()
        days = self.days_var.get().strip()
        
        if not name or not folder:
            messagebox.showerror("Error", "Please enter dataset name and select source folder")
            return
        
        if not days.isdigit():
            messagebox.showerror("Error", "Days to cure must be a number")
            return
        
        dataset_info = {
            "name": name,
            "source_folder": folder,
            "wound_type": wound_type,
            "healing_category": healing_category,
            "days_to_cure": int(days)
        }
        
        self.datasets.append(dataset_info)
        self.update_dataset_list()
        
        # Clear form
        self.name_var.set("")
        self.folder_var.set("")
        self.days_var.set("30")
        
        messagebox.showinfo("Success", f"Added dataset: {name}")
    
    def remove_dataset(self):
        selection = self.dataset_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to remove")
            return
        
        index = selection[0]
        dataset_name = self.datasets[index]["name"]
        
        if messagebox.askyesno("Confirm", f"Remove dataset '{dataset_name}'?"):
            del self.datasets[index]
            self.update_dataset_list()
    
    def update_dataset_list(self):
        self.dataset_listbox.delete(0, tk.END)
        for dataset in self.datasets:
            info = f"{dataset['name']} | {dataset['wound_type']} | {dataset['healing_category']} | {dataset['days_to_cure']} days"
            self.dataset_listbox.insert(tk.END, info)
    
    def upload_all_datasets(self):
        if not self.datasets:
            messagebox.showerror("Error", "Please add at least one dataset")
            return
        
        try:
            self.progress.start()
            self.status_label.config(text="Uploading datasets...")
            self.root.update()
            
            success_count = 0
            uploaded_datasets = []
            
            for dataset in self.datasets:
                if self.upload_single_dataset(dataset):
                    success_count += 1
                    uploaded_datasets.append(dataset["name"])
            
            self.progress.stop()
            
            if success_count > 0:
                # Create combined dataset if requested
                if self.create_combined_var.get():
                    self.create_combined_dataset(uploaded_datasets)
                
                self.status_label.config(text=f"✅ Uploaded {success_count} datasets successfully!")
                
                message = f"Successfully uploaded {success_count} datasets:\n\n"
                for name in uploaded_datasets:
                    message += f"• {name}\n"
                
                if self.create_combined_var.get():
                    message += f"\n• Combined dataset: {self.combined_name_var.get()}"
                
                message += f"\n\nNext steps:\n"
                message += f"1. Review labels.csv files\n"
                message += f"2. Train models:\n"
                for name in uploaded_datasets:
                    message += f"   curl -X POST -F 'dataset_name={name}' http://localhost:5000/train\n"
                
                messagebox.showinfo("Success", message)
            else:
                self.status_label.config(text="❌ Upload failed")
                messagebox.showerror("Error", "Failed to upload any datasets")
                
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="❌ Upload failed")
            messagebox.showerror("Error", f"Upload failed: {str(e)}")
    
    def upload_single_dataset(self, dataset_config):
        name = dataset_config["name"]
        source_folder = dataset_config["source_folder"]
        wound_type = dataset_config["wound_type"]
        healing_category = dataset_config["healing_category"]
        default_days = dataset_config["days_to_cure"]
        
        # Create dataset structure
        dataset_path = Path("datasets") / name
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Check source folder
        source_path = Path(source_folder)
        if not source_path.exists():
            return False
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
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
                    'wound_type': wound_type,
                    'healing_time_category': healing_category,
                    'days_to_cure': default_days
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
        
        return copied_count > 0
    
    def create_combined_dataset(self, uploaded_datasets):
        combined_name = self.combined_name_var.get()
        
        # Create combined dataset structure
        combined_path = Path("datasets") / combined_name
        combined_images_dir = combined_path / "images"
        combined_masks_dir = combined_path / "masks"
        
        combined_images_dir.mkdir(parents=True, exist_ok=True)
        combined_masks_dir.mkdir(parents=True, exist_ok=True)
        
        all_labels = []
        
        # Copy from all uploaded datasets
        for dataset_name in uploaded_datasets:
            source_dataset_path = Path("datasets") / dataset_name
            source_images_dir = source_dataset_path / "images"
            source_masks_dir = source_dataset_path / "masks"
            source_labels_file = source_dataset_path / "labels.csv"
            
            if not source_images_dir.exists():
                continue
            
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

def main():
    root = tk.Tk()
    app = MultipleDatasetUploaderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()





