#!/usr/bin/env python3
"""
Wound Healing Progress Tracker API
=================================

This is a prototype for hackathon/demo only. Not for clinical diagnosis.

A Flask-based backend API for tracking wound healing progress using computer vision
and machine learning. Provides robust segmentation with ML model fallback to OpenCV.

Author: ML Engineer
License: MIT
"""

import os
import sqlite3
import logging
import torch
import torch.nn as nn
import pickle
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import base64
import io

def load_classification_model():
    """Load the wound classification model."""
    try:
        model_path = os.path.join(MODEL_PATH, 'wound_classification_model.pth')
        encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            logger.warning("Classification model files not found")
            return None, None, None
        
        # Load model data with weights_only=False for compatibility
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        num_classes = model_data['num_classes']
        model = SimpleWoundClassifier(num_classes)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info(f"Classification model loaded with {num_classes} classes")
        return model, label_encoder, model_data['classes']
        
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        return None, None, None

class SimpleWoundClassifier(nn.Module):
    """Simple CNN for wound classification."""
    
    def __init__(self, num_classes):
        super(SimpleWoundClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

import base64
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths and directories
UPLOAD_DIR = Path("uploads")
MODEL_PATH = Path("models/wound_segmentation_model.pth")
DATASET_DIR = Path("datasets")
TRAINING_DIR = Path("training_data")
DB_PATH = Path("wound_tracker.db")
IMAGE_SIZE = (512, 512)  # Target size for model input
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Model and processing parameters
SEGMENTATION_THRESHOLD = 0.5
MORPH_KERNEL_SIZE = 5
MIN_AREA_PIXELS = 100  # Minimum area to consider valid wound
CONFIDENCE_THRESHOLD = 0.3

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Wound classification parameters
WOUND_TYPES = ['chronic', 'surgical', 'burn', 'diabetic', 'pressure_ulcer', 'trauma']
HEALING_TIME_CATEGORIES = ['fast_healing', 'moderate_healing', 'slow_healing', 'chronic_non_healing']

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create patients table with comprehensive patient details
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            date_of_birth TEXT,
            gender TEXT,
            contact TEXT,
            address TEXT,
            clinician TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create images table for storing analysis records
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            filename TEXT NOT NULL,
            mask_filename TEXT,
            timestamp TEXT NOT NULL,
            area_pixels INTEGER,
            area_cm2 REAL,
            model_version TEXT,
            model_confidence REAL,
            healing_pct REAL,
            days_to_heal INTEGER,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def insert_image_record(record_data: Dict[str, Any]) -> int:
    """Insert image analysis record into database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO images (
            patient_id, filename, mask_filename, timestamp,
            area_pixels, area_cm2, model_version, model_confidence,
            healing_pct, days_to_heal, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record_data.get('patient_id'),
        record_data['filename'],
        record_data.get('mask_filename'),
        record_data['timestamp'],
        record_data['area_pixels'],
        record_data.get('area_cm2'),
        record_data.get('model_version'),
        record_data.get('model_confidence'),
        record_data.get('healing_pct'),
        record_data.get('days_to_heal'),
        record_data.get('notes')
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Inserted image record with ID: {record_id}")
    return record_id

def get_patient_history(patient_id: str) -> List[Dict[str, Any]]:
    """Get analysis history for a patient."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM images 
        WHERE patient_id = ? 
        ORDER BY timestamp DESC
    ''', (patient_id,))
    
    columns = [description[0] for description in cursor.description]
    records = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return records

def get_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Get specific record by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM images WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    
    if row:
        columns = [description[0] for description in cursor.description]
        record = dict(zip(columns, row))
    else:
        record = None
    
    conn.close()
    return record

def upsert_patient_data(patient_data: Dict[str, Any]) -> bool:
    """Insert or update patient data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO patients (
                id, name, date_of_birth, gender, contact, address, 
                clinician, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data.get('id'),
            patient_data.get('name'),
            patient_data.get('date_of_birth'),
            patient_data.get('gender'),
            patient_data.get('contact'),
            patient_data.get('address'),
            patient_data.get('clinician'),
            patient_data.get('notes')
        ))
        
        conn.commit()
        logger.info(f"Patient data upserted for ID: {patient_data.get('id')}")
        return True
        
    except Exception as e:
        logger.error(f"Error upserting patient data: {e}")
        return False
    finally:
        conn.close()

def get_patient_details(patient_id: str) -> Optional[Dict[str, Any]]:
    """Get patient details by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = cursor.fetchone()
    
    if row:
        columns = [description[0] for description in cursor.description]
        patient = dict(zip(columns, row))
    else:
        patient = None
    
    conn.close()
    return patient

def get_all_patients_with_details() -> List[Dict[str, Any]]:
    """Get all patients with their details and analysis counts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT p.*, 
               COUNT(i.id) as total_records,
               MIN(i.timestamp) as first_analysis,
               MAX(i.timestamp) as latest_analysis
        FROM patients p
        LEFT JOIN images i ON p.id = i.patient_id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    ''')
    
    columns = [description[0] for description in cursor.description]
    patients = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return patients

# =============================================================================
# MODEL LOADING AND SEGMENTATION
# =============================================================================

def create_unet_model():
    """Create a U-Net model architecture for wound segmentation."""
    try:
        import torch
        import torch.nn as nn
        
        class UNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1):
                super(UNet, self).__init__()
                
                # Encoder
                self.enc1 = self.conv_block(in_channels, 64)
                self.enc2 = self.conv_block(64, 128)
                self.enc3 = self.conv_block(128, 256)
                self.enc4 = self.conv_block(256, 512)
                
                # Bottleneck
                self.bottleneck = self.conv_block(512, 1024)
                
                # Decoder
                self.dec4 = self.conv_block(1024 + 512, 512)
                self.dec3 = self.conv_block(512 + 256, 256)
                self.dec2 = self.conv_block(256 + 128, 128)
                self.dec1 = self.conv_block(128 + 64, 64)
                
                # Output
                self.final = nn.Conv2d(64, out_channels, kernel_size=1)
                
                # Pooling and upsampling
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                
            def conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                
                # Bottleneck
                b = self.bottleneck(self.pool(e4))
                
                # Decoder
                d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                
                return self.final(d1)
        
        return UNet()
        
    except ImportError:
        logger.warning("PyTorch not available for model creation")
        return None

def create_classification_model():
    """Create a wound classification model."""
    try:
        import torch
        import torch.nn as nn
        
        class WoundClassifier(nn.Module):
            def __init__(self, num_classes=len(WOUND_TYPES)):
                super(WoundClassifier, self).__init__()
                
                # Feature extractor (ResNet-like)
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Block 1
                    self._make_layer(64, 64, 2),
                    # Block 2
                    self._make_layer(64, 128, 2, stride=2),
                    # Block 3
                    self._make_layer(128, 256, 2, stride=2),
                    # Block 4
                    self._make_layer(256, 512, 2, stride=2),
                )
                
                # Global average pooling
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
                # Classifier heads
                self.wound_type_classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
                
                self.healing_time_classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, len(HEALING_TIME_CATEGORIES))
                )
                
            def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                layers = []
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                
                for _ in range(1, blocks):
                    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                    layers.append(nn.BatchNorm2d(out_channels))
                    layers.append(nn.ReLU(inplace=True))
                
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                
                wound_type = self.wound_type_classifier(x)
                healing_time = self.healing_time_classifier(x)
                
                return wound_type, healing_time
        
        return WoundClassifier()
        
    except ImportError:
        logger.warning("PyTorch not available for classification model creation")
        return None

def load_model():
    """Load PyTorch segmentation model or return None for fallback."""
    try:
        import torch
        import torchvision.transforms as transforms
        
        # Check for segmentation model first
        segmentation_model_path = MODEL_PATH / 'wound_segmentation_model.pth'
        if segmentation_model_path.exists():
            # Load segmentation model
            model_data = torch.load(segmentation_model_path, map_location='cpu', weights_only=False)
            if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                # Create model from state dict
                from fix_model_issues import SimpleUNet
                model = SimpleUNet()
                model.load_state_dict(model_data['model_state_dict'])
                model.eval()
                logger.info(f"Loaded PyTorch segmentation model from {segmentation_model_path}")
                return model
            else:
                # Direct model object
                model = model_data
                model.eval()
                logger.info(f"Loaded PyTorch segmentation model from {segmentation_model_path}")
                return model
        elif MODEL_PATH.exists():
            # Load existing model (legacy)
            model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            if hasattr(model, 'eval'):
                model.eval()
                logger.info(f"Loaded PyTorch model from {MODEL_PATH}")
                return model
            else:
                logger.warning("Loaded object is not a PyTorch model")
                return None
        else:
            # Create new model if none exists
            logger.info("No existing model found, creating new U-Net model")
            model = create_unet_model()
            if model:
                model.eval()
                # Save the new model
                MODEL_PATH.parent.mkdir(exist_ok=True)
                torch.save(model, MODEL_PATH)
                logger.info(f"Created and saved new model to {MODEL_PATH}")
            return model
            
    except ImportError:
        logger.warning("PyTorch not available, using OpenCV fallback")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}, using OpenCV fallback")
        return None

def preprocess_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess image for model input."""
    # Resize image
    img_resized = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to RGB if needed
    if len(img_normalized.shape) == 3:
        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
    
    return img_normalized

def opencv_segmentation(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Robust OpenCV-based wound segmentation fallback.
    
    Algorithm:
    1. Convert to grayscale
    2. Apply CLAHE for contrast enhancement
    3. Apply Otsu thresholding
    4. Morphological closing to fill gaps
    5. Keep largest connected component
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Otsu thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Closing to fill gaps
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Opening to remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    if num_labels > 1:
        # Get largest component (excluding background)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_component).astype(np.uint8) * 255
    else:
        mask = opened
    
    # Calculate confidence as ratio of wound pixels to total image
    confidence = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    
    return mask, confidence

def model_segmentation(img: np.ndarray, model) -> Tuple[np.ndarray, float]:
    """Perform segmentation using PyTorch model."""
    try:
        import torch
        
        # Preprocess image
        img_processed = preprocess_image(img, IMAGE_SIZE)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_processed).permute(2, 0, 1).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            
        # Apply sigmoid to get probabilities
        prob_map = torch.sigmoid(output).squeeze().numpy()
        
        # Calculate confidence as mean probability inside predicted mask
        mask = (prob_map > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
        confidence = np.mean(prob_map[mask > 0]) if np.sum(mask > 0) > 0 else 0.0
        
        return mask, confidence
        
    except Exception as e:
        logger.error(f"Model segmentation failed: {e}, falling back to OpenCV")
        return opencv_segmentation(img)

def postprocess_mask(prob_map: np.ndarray, threshold: float = SEGMENTATION_THRESHOLD) -> np.ndarray:
    """
    Postprocess probability map to create final mask.
    
    Steps:
    1. Apply threshold
    2. Morphological opening/closing
    3. Remove small objects
    4. Fill holes
    5. Extract largest contour
    """
    # Apply threshold
    mask = (prob_map > threshold).astype(np.uint8) * 255
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Opening to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Closing to fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA_PIXELS:
            mask[labels == i] = 0
    
    # Fill holes
    mask_filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create filled mask from largest contour
        mask_filled = np.zeros_like(mask)
        cv2.fillPoly(mask_filled, [largest_contour], 255)
    
    return mask_filled

# =============================================================================
# METRICS AND ANALYSIS
# =============================================================================

def calculate_metrics(mask: np.ndarray, pixel_per_cm: Optional[float] = None) -> Dict[str, Any]:
    """Calculate wound metrics from segmentation mask."""
    # Area in pixels
    area_pixels = np.sum(mask > 0)
    
    # Area in cm² if pixel_per_cm provided
    area_cm2 = None
    if pixel_per_cm:
        area_cm2 = area_pixels / (pixel_per_cm ** 2)
    
    # Bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        
        # Perimeter
        perimeter = cv2.arcLength(largest_contour, True)
    else:
        bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
        perimeter = 0
    
    return {
        "area_pixels": int(area_pixels),
        "area_cm2": area_cm2,
        "bbox": bbox,
        "perimeter": float(perimeter)
    }

def calculate_healing_metrics(current_area: int, patient_id: str, current_timestamp: str, 
                            wound_type: str = "unknown", image: np.ndarray = None, 
                            mask: np.ndarray = None) -> Dict[str, Any]:
    """Calculate healing progress metrics with dynamic prediction."""
    history = get_patient_history(patient_id)
    
    if not history:
        return {"healing_pct": None, "days_to_heal": None, "healing_category": "moderate_healing"}
    
    # Get most recent previous record (excluding current one)
    previous_record = None
    for record in history:
        if record['timestamp'] < current_timestamp:
            previous_record = record
            break
    
    if not previous_record:
        return {"healing_pct": None, "days_to_heal": None, "healing_category": "moderate_healing"}
    
    previous_area = previous_record['area_pixels']
    
    # Calculate healing percentage
    if previous_area > 0:
        healing_pct = max(0, (previous_area - current_area) / previous_area * 100)
    else:
        healing_pct = 0
    
    # Calculate days between measurements
    try:
        current_date = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
        previous_date = datetime.fromisoformat(previous_record['timestamp'].replace('Z', '+00:00'))
        days_between = (current_date - previous_date).days
    except:
        days_between = 0
    
    # Dynamic healing time prediction
    days_to_heal = None
    healing_category = "moderate_healing"
    
    if image is not None and mask is not None:
        # Use improved dynamic prediction
        try:
            from improved_healing_predictor import DynamicHealingPredictor
            predictor = DynamicHealingPredictor()
            
            # Analyze wound characteristics
            wound_chars = predictor.analyze_wound_characteristics(image, mask)
            
            # Calculate healing progress
            healing_progress = predictor.calculate_healing_progress(
                wound_chars['area_cm2'], history
            )
            
            # Predict healing time
            prediction = predictor.predict_healing_time(
                wound_type=wound_type,
                wound_characteristics=wound_chars,
                healing_progress=healing_progress,
                patient_age=None,  # Could be added to patient data
                wound_location='limbs'  # Could be added to patient data
            )
            
            days_to_heal = prediction['estimated_days_to_cure']
            healing_category = prediction['healing_time_category']
            
        except Exception as e:
            logger.warning(f"Dynamic prediction failed, using fallback: {e}")
            # Fallback to simple calculation
            if days_between > 0 and current_area > 0:
                rate_per_day = (previous_area - current_area) / days_between
                if rate_per_day > 0:
                    days_to_heal = int(current_area / rate_per_day)
    else:
        # Simple calculation fallback
        if days_between > 0 and current_area > 0:
            rate_per_day = (previous_area - current_area) / days_between
            if rate_per_day > 0:
                days_to_heal = int(current_area / rate_per_day)
    
    return {
        "healing_pct": float(healing_pct) if healing_pct is not None else None,
        "days_to_heal": days_to_heal,
        "healing_category": healing_category
    }

def calculate_dice_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Dict[str, float]:
    """Calculate Dice coefficient and IoU between predicted and ground truth masks."""
    # Ensure masks are binary
    mask_pred = (mask_pred > 0).astype(np.uint8)
    mask_gt = (mask_gt > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (mask_pred.sum() + mask_gt.sum()) if (mask_pred.sum() + mask_gt.sum()) > 0 else 0.0
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return {"dice": float(dice), "iou": float(iou)}

# =============================================================================
# IMAGE PROCESSING AND VISUALIZATION
# =============================================================================

def create_overlay_image(original_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create overlay visualization of wound segmentation."""
    # Convert original image to RGB if needed
    if len(original_img.shape) == 3:
        overlay = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask (red for wound area)
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color
    
    # Blend original image with colored mask
    alpha = 0.3  # Transparency factor
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay

def image_to_base64(img: np.ndarray, format: str = 'PNG') -> str:
    """Convert numpy image array to base64 string."""
    # Convert to PIL Image
    if len(img.shape) == 3:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_str}"

def save_image(img: np.ndarray, filepath: Path) -> None:
    """Save numpy image array to file."""
    if len(img.shape) == 3:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img, mode='L')
    
    pil_img.save(filepath)

# =============================================================================
# DATASET LOADING AND TRAINING
# =============================================================================

def load_dataset(dataset_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str], List[int]]:
    """Load wound dataset with images, masks, wound types, healing times, and days to cure."""
    images = []
    masks = []
    wound_types = []
    healing_times = []
    days_to_cure = []
    
    # Expected structure: dataset_path/images/, dataset_path/masks/, dataset_path/labels.csv
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    labels_file = dataset_path / "labels.csv"
    
    if not images_dir.exists() or not masks_dir.exists():
        logger.error(f"Dataset structure not found. Expected: {images_dir} and {masks_dir}")
        return [], [], [], [], []
    
    # Load labels if available
    labels_data = {}
    if labels_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(labels_file)
            for _, row in df.iterrows():
                labels_data[row['filename']] = {
                    'wound_type': row.get('wound_type', 'unknown'),
                    'healing_time_category': row.get('healing_time_category', 'moderate_healing'),
                    'days_to_cure': row.get('days_to_cure', 30)
                }
            logger.info(f"Loaded labels for {len(labels_data)} images")
        except Exception as e:
            logger.warning(f"Could not load labels file: {e}")
    
    # Get all image files
    image_files = []
    for ext in ALLOWED_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*.{ext}"))
        image_files.extend(images_dir.glob(f"*.{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} images in dataset")
    
    for img_file in image_files:
        try:
            # Load image
            img = np.array(Image.open(img_file))
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Find corresponding mask
            mask_file = masks_dir / f"{img_file.stem}_mask{img_file.suffix}"
            if not mask_file.exists():
                # Try alternative naming
                mask_file = masks_dir / f"{img_file.stem}{img_file.suffix}"
            
            if mask_file.exists():
                mask = np.array(Image.open(mask_file))
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                
                # Ensure mask is binary
                mask = (mask > 128).astype(np.uint8) * 255
                
                # Get labels for this image
                filename = img_file.name
                if filename in labels_data:
                    wound_type = labels_data[filename]['wound_type']
                    healing_time = labels_data[filename]['healing_time_category']
                    days = labels_data[filename]['days_to_cure']
                else:
                    # Default values if no labels
                    wound_type = 'unknown'
                    healing_time = 'moderate_healing'
                    days = 30
                
                images.append(img)
                masks.append(mask)
                wound_types.append(wound_type)
                healing_times.append(healing_time)
                days_to_cure.append(days)
            else:
                logger.warning(f"No mask found for {img_file.name}")
                
        except Exception as e:
            logger.error(f"Error loading {img_file}: {e}")
    
    logger.info(f"Successfully loaded {len(images)} image-mask pairs with labels")
    return images, masks, wound_types, healing_times, days_to_cure

def augment_data(images: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Apply data augmentation to increase dataset diversity."""
    augmented_images = []
    augmented_masks = []
    
    for img, mask in zip(images, masks):
        # Original
        augmented_images.append(img)
        augmented_masks.append(mask)
        
        # Horizontal flip
        img_flip = cv2.flip(img, 1)
        mask_flip = cv2.flip(mask, 1)
        augmented_images.append(img_flip)
        augmented_masks.append(mask_flip)
        
        # Vertical flip
        img_flip_v = cv2.flip(img, 0)
        mask_flip_v = cv2.flip(mask, 0)
        augmented_images.append(img_flip_v)
        augmented_masks.append(mask_flip_v)
        
        # Rotation (90 degrees)
        img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask_rot = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(img_rot)
        augmented_masks.append(mask_rot)
        
        # Brightness adjustment
        img_bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        augmented_images.append(img_bright)
        augmented_masks.append(mask)
        
        # Contrast adjustment
        img_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        augmented_images.append(img_contrast)
        augmented_masks.append(mask)
    
    logger.info(f"Augmented dataset from {len(images)} to {len(augmented_images)} samples")
    return augmented_images, augmented_masks

def train_model(dataset_path: Path, epochs: int = EPOCHS, train_classification: bool = True) -> bool:
    """Train both segmentation and classification models on provided dataset."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        logger.info(f"Starting model training with dataset: {dataset_path}")
        
        # Load dataset with labels
        images, masks, wound_types, healing_times, days_to_cure = load_dataset(dataset_path)
        if len(images) == 0:
            logger.error("No valid data found in dataset")
            return False
        
        # Augment data
        images, masks = augment_data(images, masks)
        
        # Encode labels
        wound_type_encoder = LabelEncoder()
        healing_time_encoder = LabelEncoder()
        
        wound_type_labels = wound_type_encoder.fit_transform(wound_types)
        healing_time_labels = healing_time_encoder.fit_transform(healing_times)
        
        logger.info(f"Wound types: {wound_type_encoder.classes_}")
        logger.info(f"Healing time categories: {healing_time_encoder.classes_}")
        
        # Preprocess images
        processed_images = []
        processed_masks = []
        
        for img, mask in zip(images, masks):
            # Resize to model input size
            img_resized = cv2.resize(img, IMAGE_SIZE)
            mask_resized = cv2.resize(mask, IMAGE_SIZE)
            
            # Normalize image
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Convert to RGB
            if len(img_normalized.shape) == 3:
                img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
            
            # Normalize mask
            mask_normalized = (mask_resized > 128).astype(np.float32)
            
            processed_images.append(img_normalized)
            processed_masks.append(mask_normalized)
        
        # Convert to tensors
        X = torch.tensor(np.array(processed_images)).permute(0, 3, 1, 2)
        y_seg = torch.tensor(np.array(processed_masks)).unsqueeze(1)
        y_wound_type = torch.tensor(wound_type_labels)
        y_healing_time = torch.tensor(healing_time_labels)
        
        # Split data
        X_train, X_val, y_seg_train, y_seg_val, y_wound_train, y_wound_val, y_heal_train, y_heal_val = train_test_split(
            X, y_seg, y_wound_type, y_healing_time, test_size=VALIDATION_SPLIT, random_state=42
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_seg_train, y_wound_train, y_heal_train)
        val_dataset = TensorDataset(X_val, y_seg_val, y_wound_val, y_heal_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Create models
        seg_model = create_unet_model()
        if seg_model is None:
            logger.error("Failed to create segmentation model")
            return False
        
        # Train segmentation model
        logger.info("Training segmentation model...")
        seg_criterion = nn.BCEWithLogitsLoss()
        seg_optimizer = optim.Adam(seg_model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            seg_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y_seg, _, _ in train_loader:
                seg_optimizer.zero_grad()
                outputs = seg_model(batch_X)
                loss = seg_criterion(outputs, batch_y_seg)
                loss.backward()
                seg_optimizer.step()
                train_loss += loss.item()
            
            # Validation
            seg_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y_seg, _, _ in val_loader:
                    outputs = seg_model(batch_X)
                    loss = seg_criterion(outputs, batch_y_seg)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Segmentation Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(seg_model, MODEL_PATH)
                logger.info(f"New best segmentation model saved with validation loss: {best_val_loss:.4f}")
        
        # Train classification model if requested
        if train_classification:
            logger.info("Training classification model...")
            cls_model = create_classification_model()
            if cls_model is None:
                logger.error("Failed to create classification model")
                return False
            
            cls_criterion = nn.CrossEntropyLoss()
            cls_optimizer = optim.Adam(cls_model.parameters(), lr=LEARNING_RATE)
            
            best_cls_val_loss = float('inf')
            
            for epoch in range(epochs // 2):  # Fewer epochs for classification
                # Training
                cls_model.train()
                train_loss = 0.0
                
                for batch_X, _, batch_y_wound, batch_y_heal in train_loader:
                    cls_optimizer.zero_grad()
                    wound_pred, heal_pred = cls_model(batch_X)
                    loss = cls_criterion(wound_pred, batch_y_wound) + cls_criterion(heal_pred, batch_y_heal)
                    loss.backward()
                    cls_optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                cls_model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, _, batch_y_wound, batch_y_heal in val_loader:
                        wound_pred, heal_pred = cls_model(batch_X)
                        loss = cls_criterion(wound_pred, batch_y_wound) + cls_criterion(heal_pred, batch_y_heal)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                logger.info(f"Classification Epoch {epoch+1}/{epochs//2} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_cls_val_loss:
                    best_cls_val_loss = avg_val_loss
                    torch.save(cls_model, MODEL_PATH.parent / "classification_model.pth")
                    logger.info(f"New best classification model saved with validation loss: {best_cls_val_loss:.4f}")
        
        # Save encoders
        import pickle
        with open(MODEL_PATH.parent / "wound_type_encoder.pkl", 'wb') as f:
            pickle.dump(wound_type_encoder, f)
        with open(MODEL_PATH.parent / "healing_time_encoder.pkl", 'wb') as f:
            pickle.dump(healing_time_encoder, f)
        
        logger.info("Training completed successfully")
        return True
        
    except ImportError:
        logger.error("PyTorch not available for training")
        return False
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def predict_wound_classification(img: np.ndarray, mask: np.ndarray = None, 
                                patient_id: str = None, timestamp: str = None) -> Dict[str, Any]:
    """Predict wound type and healing time category."""
    try:
        import torch
        import pickle
        
        # Load classification model
        cls_model_path = MODEL_PATH.parent / "classification_model.pth"
        if not cls_model_path.exists():
            return {
                "wound_type": "unknown",
                "wound_type_confidence": 0.0,
                "healing_time_category": "moderate_healing",
                "healing_time_confidence": 0.0,
                "estimated_days_to_cure": 30,
                "model_available": False
            }
        
        cls_model = torch.load(cls_model_path, map_location='cpu')
        cls_model.eval()
        
        # Load encoders
        wound_type_encoder_path = MODEL_PATH.parent / "wound_type_encoder.pkl"
        healing_time_encoder_path = MODEL_PATH.parent / "healing_time_encoder.pkl"
        
        if wound_type_encoder_path.exists() and healing_time_encoder_path.exists():
            with open(wound_type_encoder_path, 'rb') as f:
                wound_type_encoder = pickle.load(f)
            with open(healing_time_encoder_path, 'rb') as f:
                healing_time_encoder = pickle.load(f)
        else:
            return {
                "wound_type": "unknown",
                "wound_type_confidence": 0.0,
                "healing_time_category": "moderate_healing",
                "healing_time_confidence": 0.0,
                "estimated_days_to_cure": 30,
                "model_available": False
            }
        
        # Preprocess image
        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        if len(img_normalized.shape) == 3:
            img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            wound_pred, healing_pred = cls_model(img_tensor)
            
            # Get predictions with confidence
            wound_probs = torch.softmax(wound_pred, dim=1)
            healing_probs = torch.softmax(healing_pred, dim=1)
            
            wound_type_idx = torch.argmax(wound_probs, dim=1).item()
            healing_time_idx = torch.argmax(healing_probs, dim=1).item()
            
            wound_type = wound_type_encoder.inverse_transform([wound_type_idx])[0]
            healing_time_category = healing_time_encoder.inverse_transform([healing_time_idx])[0]
            
            wound_confidence = wound_probs[0][wound_type_idx].item()
            healing_confidence = healing_probs[0][healing_time_idx].item()
        
        # Use dynamic healing prediction if mask and patient data available
        estimated_days = 30  # Default
        if mask is not None and patient_id is not None and timestamp is not None:
            try:
                from improved_healing_predictor import DynamicHealingPredictor
                predictor = DynamicHealingPredictor()
                
                # Analyze wound characteristics
                wound_chars = predictor.analyze_wound_characteristics(img, mask)
                
                # Get patient history for healing progress
                history = get_patient_history(patient_id)
                healing_progress = predictor.calculate_healing_progress(
                    wound_chars['area_cm2'], history
                )
                
                # Predict healing time
                prediction = predictor.predict_healing_time(
                    wound_type=wound_type,
                    wound_characteristics=wound_chars,
                    healing_progress=healing_progress,
                    patient_age=None,
                    wound_location='limbs'
                )
                
                estimated_days = prediction['estimated_days_to_cure']
                healing_time_category = prediction['healing_time_category']
                healing_confidence = prediction['confidence']
                
            except Exception as e:
                logger.warning(f"Dynamic prediction failed, using static: {e}")
                # Fallback to static mapping
                days_mapping = {
                    'fast_healing': 7,
                    'moderate_healing': 21,
                    'slow_healing': 45,
                    'chronic_non_healing': 90
                }
                estimated_days = days_mapping.get(healing_time_category, 30)
        else:
            # Static mapping fallback
            days_mapping = {
                'fast_healing': 7,
                'moderate_healing': 21,
                'slow_healing': 45,
                'chronic_non_healing': 90
            }
            estimated_days = days_mapping.get(healing_time_category, 30)
        
        return {
            "wound_type": wound_type,
            "wound_type_confidence": round(wound_confidence, 3),
            "healing_time_category": healing_time_category,
            "healing_time_confidence": round(healing_confidence, 3),
            "estimated_days_to_cure": estimated_days,
            "model_available": True
        }
        
    except Exception as e:
        logger.error(f"Wound classification failed: {e}")
        return {
            "wound_type": "unknown",
            "wound_type_confidence": 0.0,
            "healing_time_category": "moderate_healing",
            "healing_time_confidence": 0.0,
            "estimated_days_to_cure": 30,
            "model_available": False,
            "error": str(e)
        }

def evaluate_model_accuracy(dataset_path: Path) -> Dict[str, float]:
    """Evaluate model accuracy on test dataset."""
    try:
        images, masks, _, _, _ = load_dataset(dataset_path)
        if len(images) == 0:
            return {"error": "No test data available"}
        
        model = load_model()
        if model is None:
            return {"error": "Model not available"}
        
        total_dice = 0.0
        total_iou = 0.0
        total_samples = 0
        
        for img, gt_mask in zip(images, masks):
            try:
                # Run inference
                mask_pred, _ = model_segmentation(img, model)
                
                # Calculate metrics
                metrics = calculate_dice_iou(mask_pred, gt_mask)
                
                total_dice += metrics['dice']
                total_iou += metrics['iou']
                total_samples += 1
                
            except Exception as e:
                logger.error(f"Error evaluating sample: {e}")
        
        if total_samples > 0:
            avg_dice = total_dice / total_samples
            avg_iou = total_iou / total_samples
            
            logger.info(f"Model evaluation - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
            
            return {
                "dice_score": avg_dice,
                "iou_score": avg_iou,
                "samples_evaluated": total_samples
            }
        else:
            return {"error": "No samples could be evaluated"}
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat(),
        "model_available": MODEL_PATH.exists()
    })

@app.route('/analyze', methods=['POST'])
def analyze_wound():
    """Analyze wound image and return segmentation results."""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Check file extension
        if not allowed_file(image_file.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
        
        # Get optional parameters
        patient_id = request.form.get('patient_id')
        timestamp = request.form.get('timestamp', datetime.now().isoformat())
        pixel_per_cm = request.form.get('pixel_per_cm', type=float)
        baseline_image_id = request.form.get('baseline_image_id', type=int)
        notes = request.form.get('notes', '')
        
        # Load and process image
        image_data = image_file.read()
        img = np.array(Image.open(io.BytesIO(image_data)))
        
        # Generate unique filenames
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wound_{timestamp_str}_{image_file.filename}"
        mask_filename = f"mask_{timestamp_str}_{image_file.filename}"
        
        # Save original image
        original_path = UPLOAD_DIR / filename
        save_image(img, original_path)
        
        # Load model or use fallback
        model = load_model()
        
        # Perform segmentation
        if model:
            mask, confidence = model_segmentation(img, model)
            model_version = "pytorch_v1.0"
        else:
            mask, confidence = opencv_segmentation(img)
            model_version = "opencv_fallback"
        
        # Postprocess mask
        mask_processed = postprocess_mask(mask)
        
        # Save mask
        mask_path = UPLOAD_DIR / mask_filename
        save_image(mask_processed, mask_path)
        
        # Calculate metrics
        metrics = calculate_metrics(mask_processed, pixel_per_cm)
        
        # Calculate healing metrics if patient_id provided
        healing_metrics = {}
        wound_type = "unknown"  # Default wound type
        if patient_id:
            healing_metrics = calculate_healing_metrics(
                metrics['area_pixels'], patient_id, timestamp, wound_type, img, mask_processed
            )
        
        # Handle ground truth mask if provided
        validation_metrics = {}
        if 'gt_mask' in request.files:
            gt_file = request.files['gt_mask']
            gt_data = gt_file.read()
            gt_img = np.array(Image.open(io.BytesIO(gt_data)))
            
            if len(gt_img.shape) == 3:
                gt_mask = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
            else:
                gt_mask = gt_img
            
            validation_metrics = calculate_dice_iou(mask_processed, gt_mask)
        
        # Create overlay
        overlay = create_overlay_image(img, mask_processed)
        
        # Prepare record data
        record_data = {
            'patient_id': patient_id,
            'filename': filename,
            'mask_filename': mask_filename,
            'timestamp': timestamp,
            'area_pixels': metrics['area_pixels'],
            'area_cm2': metrics['area_cm2'],
            'model_version': model_version,
            'model_confidence': confidence,
            'healing_pct': healing_metrics.get('healing_pct'),
            'days_to_heal': healing_metrics.get('days_to_heal'),
            'notes': notes
        }
        
        # Insert record into database
        record_id = insert_image_record(record_data)
        
        # Load classification model and predict wound type/healing time
        wound_classification = predict_wound_classification(img, mask_processed, patient_id, timestamp)
        
        # Prepare response
        response = {
            "status": "success",
            "record_id": record_id,
            "patient_id": patient_id,
            "area_pixels": metrics['area_pixels'],
            "area_cm2": metrics['area_cm2'],
            "healing_pct": healing_metrics.get('healing_pct'),
            "days_to_heal": healing_metrics.get('days_to_heal'),
            "model_version": model_version,
            "model_confidence": confidence,
            "bbox": metrics['bbox'],
            "perimeter": metrics['perimeter'],
            "mask_base64": image_to_base64(mask_processed),
            "overlay_base64": image_to_base64(overlay),
            "metrics": validation_metrics,
            "wound_classification": wound_classification
        }
        
        logger.info(f"Analysis completed for record {record_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_wound: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/history/<patient_id>', methods=['GET'])
def get_history(patient_id: str):
    """Get analysis history for a patient."""
    try:
        history = get_patient_history(patient_id)
        return jsonify({"patient_id": patient_id, "history": history})
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({"error": f"Failed to get history: {str(e)}"}), 500

@app.route('/mask/<int:record_id>', methods=['GET'])
def get_mask(record_id: int):
    """Get mask image for a record."""
    try:
        record = get_record_by_id(record_id)
        if not record:
            return jsonify({"error": "Record not found"}), 404
        
        mask_path = UPLOAD_DIR / record['mask_filename']
        if not mask_path.exists():
            return jsonify({"error": "Mask file not found"}), 404
        
        return send_file(mask_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error getting mask: {e}")
        return jsonify({"error": f"Failed to get mask: {str(e)}"}), 500

@app.route('/overlay/<int:record_id>', methods=['GET'])
def get_overlay(record_id: int):
    """Get overlay image for a record."""
    try:
        record = get_record_by_id(record_id)
        if not record:
            return jsonify({"error": "Record not found"}), 404
        
        # Load original image and mask
        original_path = UPLOAD_DIR / record['filename']
        mask_path = UPLOAD_DIR / record['mask_filename']
        
        if not original_path.exists() or not mask_path.exists():
            return jsonify({"error": "Image files not found"}), 404
        
        # Load images
        original_img = np.array(Image.open(original_path))
        mask_img = np.array(Image.open(mask_path))
        
        # Create overlay
        overlay = create_overlay_image(original_img, mask_img)
        
        # Convert to PIL and return
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error getting overlay: {e}")
        return jsonify({"error": f"Failed to get overlay: {str(e)}"}), 500

@app.route('/record/<int:record_id>', methods=['GET'])
def get_record(record_id: int):
    """Get record data by ID."""
    try:
        record = get_record_by_id(record_id)
        if not record:
            return jsonify({"error": "Record not found"}), 404
        
        return jsonify(record)
    except Exception as e:
        logger.error(f"Error getting record: {e}")
        return jsonify({"error": f"Failed to get record: {str(e)}"}), 500

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """Train model on uploaded dataset."""
    try:
        # Get training parameters
        epochs = request.form.get('epochs', EPOCHS, type=int)
        dataset_name = request.form.get('dataset_name', 'default')
        
        # Check if dataset exists
        dataset_path = DATASET_DIR / dataset_name
        if not dataset_path.exists():
            return jsonify({"error": f"Dataset '{dataset_name}' not found at {dataset_path}"}), 404
        
        # Start training
        logger.info(f"Starting training with dataset: {dataset_name}, epochs: {epochs}")
        success = train_model(dataset_path, epochs)
        
        if success:
            # Evaluate model after training
            accuracy_metrics = evaluate_model_accuracy(dataset_path)
            
            return jsonify({
                "status": "success",
                "message": "Model training completed",
                "dataset": dataset_name,
                "epochs": epochs,
                "accuracy_metrics": accuracy_metrics
            })
        else:
            return jsonify({"error": "Training failed"}), 500
            
    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_model_endpoint():
    """Evaluate model accuracy on test dataset."""
    try:
        dataset_name = request.form.get('dataset_name', 'default')
        dataset_path = DATASET_DIR / dataset_name
        
        if not dataset_path.exists():
            return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404
        
        accuracy_metrics = evaluate_model_accuracy(dataset_path)
        
        if "error" in accuracy_metrics:
            return jsonify(accuracy_metrics), 500
        
        return jsonify({
            "status": "success",
            "dataset": dataset_name,
            "accuracy_metrics": accuracy_metrics
        })
        
    except Exception as e:
        logger.error(f"Evaluation endpoint error: {e}")
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

@app.route('/datasets', methods=['GET'])
def list_datasets():
    """List available datasets."""
    try:
        if not DATASET_DIR.exists():
            return jsonify({"datasets": []})
        
        datasets = []
        for dataset_path in DATASET_DIR.iterdir():
            if dataset_path.is_dir():
                images_dir = dataset_path / "images"
                masks_dir = dataset_path / "masks"
                
                if images_dir.exists() and masks_dir.exists():
                    # Count images
                    image_count = len(list(images_dir.glob("*")))
                    mask_count = len(list(masks_dir.glob("*")))
                    
                    datasets.append({
                        "name": dataset_path.name,
                        "path": str(dataset_path),
                        "image_count": image_count,
                        "mask_count": mask_count,
                        "ready_for_training": image_count > 0 and mask_count > 0
                    })
        
        return jsonify({"datasets": datasets})
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return jsonify({"error": f"Failed to list datasets: {str(e)}"}), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get model status and information."""
    try:
        model_info = {
            "model_exists": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "model_size_mb": 0,
            "last_trained": None
        }
        
        if MODEL_PATH.exists():
            # Get file size
            model_info["model_size_mb"] = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2)
            
            # Get modification time
            model_info["last_trained"] = datetime.fromtimestamp(
                MODEL_PATH.stat().st_mtime
            ).isoformat()
        
        return jsonify(model_info)
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({"error": f"Failed to get model status: {str(e)}"}), 500

@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    """Batch prediction on multiple images."""
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        images = request.files.getlist('images')
        if not images:
            return jsonify({"error": "No images provided"}), 400
        
        pixel_per_cm = request.form.get('pixel_per_cm', type=float)
        patient_id = request.form.get('patient_id', 'batch')
        
        results = []
        
        for i, image_file in enumerate(images):
            try:
                # Process each image
                image_data = image_file.read()
                img = np.array(Image.open(io.BytesIO(image_data)))
                
                # Generate unique filename
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_{timestamp_str}_{i}_{image_file.filename}"
                
                # Save original image
                original_path = UPLOAD_DIR / filename
                save_image(img, original_path)
                
                # Load model
                model = load_model()
                
                # Perform segmentation
                if model:
                    mask, confidence = model_segmentation(img, model)
                    model_version = "pytorch_v1.0"
                else:
                    mask, confidence = opencv_segmentation(img)
                    model_version = "opencv_fallback"
                
                # Postprocess mask
                mask_processed = postprocess_mask(mask)
                
                # Calculate metrics
                metrics = calculate_metrics(mask_processed, pixel_per_cm)
                
                # Create overlay
                overlay = create_overlay_image(img, mask_processed)
                
                results.append({
                    "image_index": i,
                    "filename": filename,
                    "area_pixels": metrics['area_pixels'],
                    "area_cm2": metrics['area_cm2'],
                    "model_version": model_version,
                    "model_confidence": confidence,
                    "bbox": metrics['bbox'],
                    "perimeter": metrics['perimeter'],
                    "mask_base64": image_to_base64(mask_processed),
                    "overlay_base64": image_to_base64(overlay)
                })
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                results.append({
                    "image_index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "status": "success",
            "patient_id": patient_id,
            "total_images": len(images),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/generate-report/<patient_id>', methods=['GET'])
def generate_patient_report(patient_id: str):
    """Generate comprehensive report for a patient with all stored data."""
    try:
        # Get patient history from database
        patient_history = get_patient_history(patient_id)
        
        if not patient_history:
            return jsonify({"error": f"No data found for patient ID: {patient_id}"}), 404
        
        # Get patient information from the most recent record
        latest_record = patient_history[0]  # Most recent record
        
        # Calculate healing progress over time
        healing_progress = []
        for i, record in enumerate(patient_history):
            if i > 0:  # Skip first record (no previous to compare)
                previous_record = patient_history[i-1]
                if previous_record['area_pixels'] > 0:
                    progress_pct = ((previous_record['area_pixels'] - record['area_pixels']) / previous_record['area_pixels']) * 100
                    healing_progress.append({
                        "date": record['timestamp'],
                        "progress_percentage": max(0, progress_pct),
                        "area_change": previous_record['area_pixels'] - record['area_pixels'],
                        "days_between": (datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00')) - 
                                       datetime.fromisoformat(previous_record['timestamp'].replace('Z', '+00:00'))).days
                    })
        
        # Calculate overall statistics
        total_records = len(patient_history)
        initial_area = patient_history[-1]['area_pixels'] if patient_history else 0
        current_area = patient_history[0]['area_pixels'] if patient_history else 0
        total_healing_pct = ((initial_area - current_area) / initial_area * 100) if initial_area > 0 else 0
        
        # Calculate time span
        if len(patient_history) > 1:
            start_date = datetime.fromisoformat(patient_history[-1]['timestamp'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(patient_history[0]['timestamp'].replace('Z', '+00:00'))
            total_days = (end_date - start_date).days
        else:
            total_days = 0
        
        # Generate wound classification for latest record
        latest_wound_classification = predict_wound_classification(
            None, None, patient_id, latest_record['timestamp']
        )
        
        # Prepare comprehensive report data
        report_data = {
            "patient_id": patient_id,
            "report_generated": datetime.now().isoformat(),
            "total_records": total_records,
            "time_span_days": total_days,
            "overall_healing_percentage": max(0, total_healing_pct),
            "initial_area_pixels": initial_area,
            "current_area_pixels": current_area,
            "latest_analysis": {
                "timestamp": latest_record['timestamp'],
                "area_pixels": latest_record['area_pixels'],
                "area_cm2": latest_record.get('area_cm2'),
                "model_version": latest_record.get('model_version'),
                "model_confidence": latest_record.get('model_confidence'),
                "healing_pct": latest_record.get('healing_pct'),
                "days_to_heal": latest_record.get('days_to_heal'),
                "notes": latest_record.get('notes')
            },
            "wound_classification": latest_wound_classification,
            "healing_progress": healing_progress,
            "all_records": patient_history,
            "statistics": {
                "average_area": sum(r['area_pixels'] for r in patient_history) / len(patient_history),
                "min_area": min(r['area_pixels'] for r in patient_history),
                "max_area": max(r['area_pixels'] for r in patient_history),
                "healing_rate_per_day": total_healing_pct / total_days if total_days > 0 else 0
            }
        }
        
        logger.info(f"Generated comprehensive report for patient {patient_id} with {total_records} records")
        return jsonify({
            "status": "success",
            "report_data": report_data
        })
        
    except Exception as e:
        logger.error(f"Error generating patient report: {e}")
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

@app.route('/patients', methods=['GET'])
def list_patients():
    """List all patients with their data summary."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all unique patient IDs with their latest record info
        cursor.execute('''
            SELECT patient_id, COUNT(*) as record_count, 
                   MIN(timestamp) as first_record,
                   MAX(timestamp) as latest_record,
                   MAX(area_pixels) as max_area,
                   MIN(area_pixels) as min_area,
                   AVG(area_pixels) as avg_area
            FROM images 
            WHERE patient_id IS NOT NULL 
            GROUP BY patient_id
            ORDER BY latest_record DESC
        ''')
        
        patients = []
        for row in cursor.fetchall():
            patient_id, record_count, first_record, latest_record, max_area, min_area, avg_area = row
            
            # Calculate healing progress
            healing_pct = ((max_area - min_area) / max_area * 100) if max_area > 0 else 0
            
            patients.append({
                "patient_id": patient_id,
                "record_count": record_count,
                "first_record_date": first_record,
                "latest_record_date": latest_record,
                "healing_percentage": max(0, healing_pct),
                "max_area_pixels": max_area,
                "min_area_pixels": min_area,
                "avg_area_pixels": round(avg_area, 2)
            })
        
        conn.close()
        
        return jsonify({
            "status": "success",
            "patients": patients,
            "total_patients": len(patients)
        })
        
    except Exception as e:
        logger.error(f"Error listing patients: {e}")
        return jsonify({"error": f"Failed to list patients: {str(e)}"}), 500

@app.route('/patients/details', methods=['GET'])
def list_patients_with_details():
    """List all patients with their complete details."""
    try:
        patients = get_all_patients_with_details()
        
        return jsonify({
            "status": "success",
            "patients": patients,
            "total_patients": len(patients)
        })
        
    except Exception as e:
        logger.error(f"Error listing patients with details: {e}")
        return jsonify({"error": f"Failed to list patients: {str(e)}"}), 500

@app.route('/patient/<patient_id>/details', methods=['GET'])
def get_patient_details_endpoint(patient_id: str):
    """Get patient details by ID."""
    try:
        patient = get_patient_details(patient_id)
        
        if not patient:
            return jsonify({"error": f"Patient {patient_id} not found"}), 404
        
        return jsonify({
            "status": "success",
            "patient": patient
        })
        
    except Exception as e:
        logger.error(f"Error getting patient details: {e}")
        return jsonify({"error": f"Failed to get patient details: {str(e)}"}), 500

@app.route('/patient/<patient_id>/save', methods=['POST'])
def save_patient_details(patient_id: str):
    """Save or update patient details."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        patient_data = {
            'id': patient_id,
            'name': data.get('name', ''),
            'date_of_birth': data.get('date_of_birth', ''),
            'gender': data.get('gender', ''),
            'contact': data.get('contact', ''),
            'address': data.get('address', ''),
            'clinician': data.get('clinician', ''),
            'notes': data.get('notes', '')
        }
        
        success = upsert_patient_data(patient_data)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Patient {patient_id} details saved successfully"
            })
        else:
            return jsonify({"error": "Failed to save patient details"}), 500
            
    except Exception as e:
        logger.error(f"Error saving patient details: {e}")
        return jsonify({"error": f"Failed to save patient details: {str(e)}"}), 500

@app.route('/patient/<patient_id>/history/days', methods=['GET'])
def get_patient_history_by_days(patient_id: str):
    """Get patient history organized by days."""
    try:
        history = get_patient_history(patient_id)
        
        if not history:
            return jsonify({"error": f"No history found for patient {patient_id}"}), 404
        
        # Group records by day
        days_data = {}
        for record in history:
            # Extract date from timestamp
            date_str = record['timestamp'][:10]  # Get YYYY-MM-DD part
            
            if date_str not in days_data:
                days_data[date_str] = {
                    'date': date_str,
                    'records': [],
                    'total_records': 0,
                    'avg_area': 0,
                    'min_area': float('inf'),
                    'max_area': 0
                }
            
            days_data[date_str]['records'].append(record)
            days_data[date_str]['total_records'] += 1
            
            area = record['area_pixels']
            days_data[date_str]['min_area'] = min(days_data[date_str]['min_area'], area)
            days_data[date_str]['max_area'] = max(days_data[date_str]['max_area'], area)
        
        # Calculate averages and convert to list
        days_list = []
        for date_str, day_data in days_data.items():
            day_data['avg_area'] = sum(r['area_pixels'] for r in day_data['records']) / day_data['total_records']
            day_data['min_area'] = day_data['min_area'] if day_data['min_area'] != float('inf') else 0
            days_list.append(day_data)
        
        # Sort by date (newest first)
        days_list.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            "status": "success",
            "patient_id": patient_id,
            "days": days_list,
            "total_days": len(days_list)
        })
        
    except Exception as e:
        logger.error(f"Error getting patient history by days: {e}")
        return jsonify({"error": f"Failed to get patient history: {str(e)}"}), 500

@app.route('/patient/<patient_id>/day/<date>/delete', methods=['DELETE'])
def delete_patient_day(patient_id: str, date: str):
    """Delete all records for a specific patient on a specific day."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete records for the specific patient and date
        cursor.execute('''
            DELETE FROM images 
            WHERE patient_id = ? AND DATE(timestamp) = ?
        ''', (patient_id, date))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} records for patient {patient_id} on {date}")
            return jsonify({
                "status": "success",
                "message": f"Deleted {deleted_count} records for {date}",
                "deleted_count": deleted_count
            })
        else:
            return jsonify({"error": f"No records found for patient {patient_id} on {date}"}), 404
            
    except Exception as e:
        logger.error(f"Error deleting patient day: {e}")
        return jsonify({"error": f"Failed to delete day: {str(e)}"}), 500

@app.route('/patient/<patient_id>/analysis', methods=['POST'])
def save_patient_analysis(patient_id: str):
    """Save analysis data for a patient without requiring image upload."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Prepare record data
        record_data = {
            'patient_id': patient_id,
            'filename': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            'mask_filename': None,
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'area_pixels': data.get('area_pixels', 0),
            'area_cm2': data.get('area_cm2', 0),
            'model_version': data.get('model_version', 'report_generated'),
            'model_confidence': data.get('model_confidence', 0.85),
            'healing_pct': data.get('healing_pct', 0),
            'days_to_heal': data.get('days_to_heal', 21),
            'notes': data.get('notes', 'Report generated analysis')
        }
        
        # Insert record into database
        record_id = insert_image_record(record_data)
        
        logger.info(f"Analysis data saved for patient {patient_id}, record {record_id}")
        
        return jsonify({
            "status": "success",
            "message": f"Analysis data saved for patient {patient_id}",
            "record_id": record_id
        })
        
    except Exception as e:
        logger.error(f"Error saving patient analysis: {e}")
        return jsonify({"error": f"Failed to save analysis data: {str(e)}"}), 500

@app.route('/patient/<patient_id>/day/<date>/update', methods=['POST'])
def update_patient_day(patient_id: str, date: str):
    """Update wound data for a specific patient on a specific day."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Update records for the specific patient and date
        cursor.execute('''
            UPDATE images 
            SET area_pixels = ?, area_cm2 = ?, notes = ?, 
                healing_pct = ?, days_to_heal = ?
            WHERE patient_id = ? AND DATE(timestamp) = ?
        ''', (
            data.get('area_pixels'),
            data.get('area_cm2'),
            data.get('notes', ''),
            data.get('healing_pct'),
            data.get('days_to_heal'),
            patient_id,
            date
        ))
        
        updated_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if updated_count > 0:
            logger.info(f"Updated {updated_count} records for patient {patient_id} on {date}")
            return jsonify({
                "status": "success",
                "message": f"Updated {updated_count} records for {date}",
                "updated_count": updated_count
            })
        else:
            return jsonify({"error": f"No records found for patient {patient_id} on {date}"}), 404
            
    except Exception as e:
        logger.error(f"Error updating patient day: {e}")
        return jsonify({"error": f"Failed to update day: {str(e)}"}), 500

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =============================================================================
# INITIALIZATION
# =============================================================================

def setup_directories():
    """Create necessary directories."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)
    TRAINING_DIR.mkdir(exist_ok=True)
    logger.info(f"Created directories: {UPLOAD_DIR}, {MODEL_PATH.parent}, {DATASET_DIR}, {TRAINING_DIR}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Setup
    setup_directories()
    init_database()
    
    # Load model (optional)
    model = load_model()
    
    logger.info("Wound Healing Progress Tracker API started")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Model available: {MODEL_PATH.exists()}")
    
    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)

"""
SETUP INSTRUCTIONS
==================

1. Install dependencies:
   pip install flask flask-cors pillow opencv-python numpy scikit-learn

2. For PyTorch model training:
   pip install torch torchvision

3. Prepare your dataset:
   mkdir -p datasets/your_dataset_name/images
   mkdir -p datasets/your_dataset_name/masks
   # Place wound images in images/ folder
   # Place corresponding masks in masks/ folder
   # Mask files should be named: image_name_mask.png or image_name.png

4. Run the application:
   python app.py

5. Train your model:
   curl -X POST -F "dataset_name=your_dataset_name" \
        -F "epochs=50" \
        http://localhost:5000/train

6. Evaluate model accuracy:
   curl -X POST -F "dataset_name=your_dataset_name" \
        http://localhost:5000/evaluate

7. Test single image analysis:
   curl -X POST -F "image=@wound_image.jpg" \
        -F "patient_id=patient_001" \
        -F "pixel_per_cm=50.0" \
        -F "timestamp=2024-01-15T10:30:00Z" \
        http://localhost:5000/analyze

8. Batch prediction:
   curl -X POST -F "images=@image1.jpg" -F "images=@image2.jpg" \
        -F "patient_id=batch_test" \
        -F "pixel_per_cm=50.0" \
        http://localhost:5000/predict/batch

9. Check available datasets:
   curl http://localhost:5000/datasets

10. Check model status:
    curl http://localhost:5000/model/status

DATASET STRUCTURE
=================

Your dataset should be organized as follows:
datasets/
└── your_dataset_name/
    ├── images/
    │   ├── wound_001.jpg
    │   ├── wound_002.png
    │   └── ...
    └── masks/
        ├── wound_001_mask.jpg  (or wound_001.jpg)
        ├── wound_002_mask.png  (or wound_002.png)
        └── ...

TRAINING FEATURES
=================

- Automatic data augmentation (flips, rotations, brightness/contrast)
- U-Net architecture with skip connections
- Validation split and early stopping
- Dice and IoU metrics for evaluation
- Robust OpenCV fallback when PyTorch unavailable
- Batch processing for multiple images

ACCURACY IMPROVEMENTS
=====================

1. **Data Quality**: Ensure high-quality, diverse wound images
2. **Mask Quality**: Accurate ground truth masks are crucial
3. **Data Augmentation**: Automatically applied during training
4. **Model Architecture**: U-Net with skip connections for precise segmentation
5. **Validation**: Regular evaluation on test set
6. **Incremental Learning**: Retrain with new data to improve accuracy

For production deployment:
- Use gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app:app
- Mount persistent storage for uploads, datasets, and models
- Set up proper logging and monitoring
- Configure reverse proxy (nginx) for SSL termination
- Regular model retraining with new data
"""
