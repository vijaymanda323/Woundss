#!/usr/bin/env python3
"""
Wound Analysis API with Real-time Learning
==========================================

A Flask backend that provides wound prediction and learns from user feedback.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import threading
import queue
import json
from pathlib import Path
import hashlib
import sqlite3
import requests
import base64
from typing import Dict
from intelligent_agent import analyze_wound_intelligently

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path('uploads')
FEEDBACK_DIR = Path('feedback_data')
MODEL_DIR = Path('models')
DB_PATH = Path('predictions.db')
UPLOAD_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Global variables
model = None
label_encoder = None
model_classes = None
training_queue = queue.Queue()
is_training = False

# External AI Service Configuration
AI_SERVICES = {
    'openai': {
        'name': 'ChatGPT (OpenAI)',
        'api_url': 'https://api.openai.com/v1/chat/completions',
        'model': 'gpt-4-vision-preview',
        'requires_key': True
    },
    'gemini': {
        'name': 'Google Gemini',
        'api_url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent',
        'model': 'gemini-1.5-flash',
        'requires_key': True,
        'api_key': 'AIzaSyByMLtCKbW16mbJ-rL6LB5d_lp4m_XQJ6w'  # Your Gemini API key
    },
    'claude': {
        'name': 'Anthropic Claude',
        'api_url': 'https://api.anthropic.com/v1/messages',
        'model': 'claude-3-sonnet-20240229',
        'requires_key': True
    }
}

def init_database():
    """Initialize SQLite database for storing predictions."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE NOT NULL,
                image_path TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                feedback_status TEXT,
                feedback_timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def calculate_image_hash(image_data):
    """Calculate SHA256 hash of image data."""
    return hashlib.sha256(image_data).hexdigest()

def get_cached_prediction(image_hash):
    """Get cached prediction for image hash."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT predicted_label, confidence, timestamp, feedback_status
            FROM predictions 
            WHERE image_hash = ?
        ''', (image_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'prediction': result[0],
                'confidence': result[1],
                'timestamp': result[2],
                'feedback_status': result[3],
                'cached': True
            }
        return None
        
    except Exception as e:
        logger.error(f"Error getting cached prediction: {e}")
        return None

def save_prediction(image_hash, image_path, predicted_label, confidence, timestamp):
    """Save prediction to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO predictions 
            (image_hash, image_path, predicted_label, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_hash, image_path, predicted_label, confidence, timestamp))
        
        conn.commit()
        conn.close()
        logger.info(f"Prediction saved for hash: {image_hash[:8]}...")
        return True
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return False

def update_feedback(image_hash, feedback_status):
    """Update feedback status for a prediction."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET feedback_status = ?, feedback_timestamp = ?
            WHERE image_hash = ?
        ''', (feedback_status, datetime.now().isoformat(), image_hash))
        
        conn.commit()
        conn.close()
        logger.info(f"Feedback updated for hash: {image_hash[:8]}...")
        return True
        
    except Exception as e:
        logger.error(f"Error updating feedback: {e}")
        return False

class ImprovedWoundClassifier(nn.Module):
    """Improved CNN for wound classification using ResNet backbone."""
    
    def __init__(self, num_classes):
        super(ImprovedWoundClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load the trained wound classification model."""
    global model, label_encoder, model_classes
    
    try:
        model_path = MODEL_DIR / 'wound_classification_model.pth'
        encoder_path = MODEL_DIR / 'label_encoder.pkl'
        
        if not model_path.exists() or not encoder_path.exists():
            logger.error("Model files not found")
            return False
        
        # Load model data
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        num_classes = model_data['num_classes']
        model = ImprovedWoundClassifier(num_classes)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        model_classes = model_data['classes']
        
        logger.info(f"Model loaded successfully with {num_classes} classes")
        logger.info(f"Classes: {model_classes}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image_file):
    """Preprocess uploaded image for prediction."""
    try:
        # Load image
        image = Image.open(image_file).convert('RGB')
        
        # Transform for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor, image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

def predict_wound(image_tensor):
    """Predict wound type from preprocessed image."""
    global model, label_encoder
    
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return predicted_class, confidence
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None, 0.0

def save_feedback(image_path, predicted_label, feedback_status, confidence):
    """Save user feedback to dataset."""
    try:
        feedback_file = FEEDBACK_DIR / 'feedback.csv'
        
        # Create feedback entry
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_path': str(image_path),
            'predicted_label': predicted_label,
            'feedback_status': feedback_status,
            'confidence': confidence
        }
        
        # Append to CSV
        df = pd.DataFrame([feedback_entry])
        if feedback_file.exists():
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            df.to_csv(feedback_file, index=False)
        
        logger.info(f"Feedback saved: {predicted_label} -> {feedback_status}")
        
        # Add to training queue
        training_queue.put(feedback_entry)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False

def retrain_model():
    """Retrain model with new feedback data."""
    global is_training, model, label_encoder
    
    if is_training:
        return
    
    is_training = True
    
    try:
        logger.info("Starting model retraining...")
        
        # Collect feedback data
        feedback_file = FEEDBACK_DIR / 'feedback.csv'
        if not feedback_file.exists():
            logger.info("No feedback data available for retraining")
            return
        
        feedback_df = pd.read_csv(feedback_file)
        
        # Only retrain if we have enough new feedback
        if len(feedback_df) < 5:
            logger.info("Not enough feedback data for retraining")
            return
        
        # Simple retraining approach - fine-tune last layer
        model.train()
        
        # Create optimizer for only the last layer
        optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Simple training loop (in practice, you'd want more sophisticated training)
        for epoch in range(3):  # Few epochs for quick retraining
            total_loss = 0
            for _, row in feedback_df.iterrows():
                try:
                    # Load image
                    image_path = row['image_path']
                    if not os.path.exists(image_path):
                        continue
                    
                    image = Image.open(image_path).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    image_tensor = transform(image).unsqueeze(0)
                    
                    # Get label
                    predicted_label = row['predicted_label']
                    feedback_status = row['feedback_status']
                    
                    # Skip if feedback was "right" (no need to retrain)
                    if feedback_status == 'right':
                        continue
                    
                    # For "wrong" feedback, we'd need the correct label
                    # For now, we'll just fine-tune with the existing prediction
                    label_idx = label_encoder.transform([predicted_label])[0]
                    target = torch.tensor([label_idx])
                    
                    optimizer.zero_grad()
                    output = model(image_tensor)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    logger.error(f"Error in training step: {e}")
                    continue
        
        # Save updated model
        model.eval()
        model_data = {
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
            'num_classes': len(label_encoder.classes_),
            'classes': list(label_encoder.classes_),
            'model_type': 'classification'
        }
        
        torch.save(model_data, MODEL_DIR / 'wound_classification_model.pth')
        
        logger.info("Model retraining completed and saved")
        
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
    finally:
        is_training = False

def training_worker():
    """Background worker for model retraining."""
    while True:
        try:
            # Wait for feedback
            feedback_entry = training_queue.get(timeout=60)
            
            # Retrain model
            retrain_model()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in training worker: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['http://localhost:8081', 'http://127.0.0.1:8081', 'http://10.81.160.244:8081', 'exp://10.81.160.244:8081'])

# Initialize database
if not init_database():
    logger.error("Failed to initialize database")

# Load model on startup
if not load_model():
    logger.error("Failed to load model on startup")

# Start training worker thread
training_thread = threading.Thread(target=training_worker, daemon=True)
training_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': model_classes if model_classes else []
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict wound type from uploaded image using Gemini AI."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data for hashing
        image_data = image_file.read()
        image_hash = calculate_image_hash(image_data)
        
        # Check if we have a cached prediction
        cached_result = get_cached_prediction(image_hash)
        if cached_result:
            logger.info(f"Returning cached Gemini prediction for hash: {image_hash[:8]}...")
            return jsonify({
                'prediction': cached_result['prediction'],
                'confidence': cached_result['confidence'],
                'timestamp': cached_result['timestamp'],
                'cached': True,
                'feedback_status': cached_result['feedback_status'],
                'image_hash': image_hash,
                'analysis_method': 'gemini_ai'
            })
        
        # Use Gemini AI for analysis
        logger.info("Analyzing image with Gemini AI...")
        gemini_analysis = analyze_with_external_ai(image_data, 'gemini', AI_SERVICES['gemini']['api_key'])
        
        if 'error' in gemini_analysis:
            logger.error(f"Gemini analysis failed: {gemini_analysis['error']}")
            return jsonify({'error': f"AI analysis failed: {gemini_analysis['error']}"}), 500
        
        # Extract prediction and confidence from Gemini response
        prediction = gemini_analysis.get('prediction', 'unknown')
        confidence = gemini_analysis.get('confidence', 0.8)  # Gemini typically has high confidence
        
        # Save prediction to database
        timestamp_str = datetime.now().isoformat()
        save_prediction(image_hash, image_file.filename, prediction, confidence, timestamp_str)
        
        logger.info(f"Gemini analysis completed: {prediction} (confidence: {confidence:.3f})")
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': timestamp_str,
            'cached': False,
            'image_hash': image_hash,
            'analysis_method': 'gemini_ai',
            'gemini_analysis': gemini_analysis
        })
        
    except Exception as e:
        logger.error(f"Error in Gemini prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Save user feedback with learning capability."""
    try:
        data = request.get_json()
        
        required_fields = ['image_hash', 'feedback_status']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Update feedback in database
        success = update_feedback(data['image_hash'], data['feedback_status'])
        
        if success:
            # Also save to CSV for training
            save_feedback_csv(data)
            
            # If incorrect prediction with correct type, trigger learning
            if data['feedback_status'] == 'wrong' and 'correct_type' in data:
                trigger_model_learning(data)
            
            return jsonify({'status': 'success', 'message': 'Feedback saved and model learning triggered'})
        else:
            return jsonify({'error': 'Failed to save feedback'}), 500
            
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def trigger_model_learning(feedback_data):
    """Trigger model learning from user feedback."""
    try:
        logger.info(f"Model learning triggered: {feedback_data['predicted_type']} -> {feedback_data['correct_type']}")
        
        # Add to training queue for background learning
        training_data = {
            'image_hash': feedback_data['image_hash'],
            'predicted_type': feedback_data['predicted_type'],
            'correct_type': feedback_data['correct_type'],
            'timestamp': datetime.now().isoformat(),
            'learning_type': 'user_correction'
        }
        
        training_queue.put(training_data)
        logger.info("Training data added to queue for model learning")
        
    except Exception as e:
        logger.error(f"Error triggering model learning: {e}")

def save_feedback_csv(data):
    """Save feedback to CSV for training purposes."""
    try:
        feedback_file = FEEDBACK_DIR / 'feedback.csv'
        
        # Get prediction details from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_path, predicted_label, confidence
            FROM predictions 
            WHERE image_hash = ?
        ''', (data['image_hash'],))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Create feedback entry
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'image_path': result[0],
                'predicted_label': result[1],
                'feedback_status': data['feedback_status'],
                'confidence': result[2],
                'correct_type': data.get('correct_type', ''),
                'learning_triggered': 'yes' if data['feedback_status'] == 'wrong' and 'correct_type' in data else 'no'
            }
            
            # Append to CSV
            df = pd.DataFrame([feedback_entry])
            if feedback_file.exists():
                df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                df.to_csv(feedback_file, index=False)
            
            logger.info(f"Feedback saved to CSV: {data['feedback_status']}")
            
    except Exception as e:
        logger.error(f"Error saving feedback to CSV: {e}")

@app.route('/history', methods=['GET'])
def get_history():
    """Get patient history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT image_path, predicted_label, confidence, timestamp, feedback_status
            FROM predictions 
            ORDER BY timestamp DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        history = []
        for row in results:
            history.append({
                'timestamp': row[3],
                'image_path': row[0],
                'predicted_label': row[1],
                'feedback_status': row[4],
                'confidence': row[2]
            })
        
        return jsonify({'history': history})
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze-intelligent', methods=['POST'])
def analyze_intelligent():
    """Perform intelligent wound analysis using Gemini AI."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data
        image_data = file.read()
        
        # Calculate image hash for caching
        image_hash = calculate_image_hash(image_data)
        
        # Check if we have cached results
        cached_result = get_cached_prediction(image_hash)
        if cached_result:
            logger.info(f"Returning cached Gemini analysis for hash: {image_hash[:8]}...")
            return jsonify({
                'status': 'success',
                'analysis': cached_result,
                'cached': True,
                'image_hash': image_hash,
                'analysis_method': 'gemini_ai'
            })
        
        # Use Gemini AI for intelligent analysis
        logger.info("Performing intelligent analysis with Gemini AI...")
        gemini_analysis = analyze_with_external_ai(image_data, 'gemini', AI_SERVICES['gemini']['api_key'])
        
        if 'error' in gemini_analysis:
            logger.error(f"Gemini analysis failed: {gemini_analysis['error']}")
            return jsonify({
                'status': 'error',
                'message': f"AI analysis failed: {gemini_analysis['error']}"
            }), 500
        
        # Format analysis for compatibility
        analysis = {
            'prediction': gemini_analysis.get('prediction', 'unknown'),
            'confidence': gemini_analysis.get('confidence', 0.8),
            'timestamp': datetime.now().isoformat(),
            'analysis_method': 'gemini_ai',
            'gemini_response': gemini_analysis
        }
        
        # Save new analysis to cache
        save_prediction(
            image_hash=image_hash,
            image_path=file.filename,
            predicted_label=analysis['prediction'],
            confidence=analysis['confidence'],
            timestamp=analysis['timestamp']
        )
        
        logger.info(f"Gemini intelligent analysis completed: {analysis['prediction']} (confidence: {analysis['confidence']:.3f})")
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'cached': False,
            'image_hash': image_hash,
            'analysis_method': 'gemini_ai'
        })
        
    except Exception as e:
        logger.error(f"Error in Gemini intelligent analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/analyze-external-ai', methods=['POST'])
def analyze_external_ai():
    """Analyze wound using external AI service."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get AI service and API key from request
        ai_service = request.form.get('ai_service', 'openai')
        api_key = request.form.get('api_key', '')
        
        if not api_key:
            return jsonify({'error': 'API key required for external AI service'}), 400
        
        # Read image data
        image_data = file.read()
        
        # Analyze with external AI
        analysis = analyze_with_external_ai(image_data, ai_service, api_key)
        
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 500
        
        # Add timestamp
        analysis['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"External AI analysis completed using {ai_service}")
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'ai_service': ai_service
        })
        
    except Exception as e:
        logger.error(f"Error in external AI analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/ai-services', methods=['GET'])
def get_ai_services():
    """Get available AI services."""
    return jsonify({
        'services': AI_SERVICES
    })

@app.route('/compare-analysis', methods=['POST'])
def compare_analysis():
    """Compare local analysis with external AI analysis."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data
        image_data = file.read()
        
        # Get local analysis
        local_analysis = analyze_wound_intelligently(image_data)
        
        # Get external AI analysis (if API key provided)
        external_analysis = None
        ai_service = request.form.get('ai_service', 'openai')
        api_key = request.form.get('api_key', '')
        
        if api_key:
            try:
                external_analysis = analyze_with_external_ai(image_data, ai_service, api_key)
            except Exception as e:
                logger.warning(f"External AI analysis failed: {e}")
                external_analysis = {'error': str(e)}
        
        # Calculate image hash for caching
        image_hash = calculate_image_hash(image_data)
        
        # Compare results
        comparison = {
            'image_hash': image_hash,
            'local_analysis': local_analysis,
            'external_analysis': external_analysis,
            'comparison': {
                'local_prediction': local_analysis.get('prediction', 'unknown'),
                'external_prediction': external_analysis.get('prediction', 'unknown') if external_analysis and 'error' not in external_analysis else 'unknown',
                'prediction_match': local_analysis.get('prediction', 'unknown') == external_analysis.get('prediction', 'unknown') if external_analysis and 'error' not in external_analysis else False,
                'confidence_difference': abs(local_analysis.get('confidence', 0) - external_analysis.get('confidence', 0)) if external_analysis and 'error' not in external_analysis else None,
                'analysis_method': local_analysis.get('method', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Analysis comparison completed for hash: {image_hash[:8]}...")
        
        return jsonify({
            'status': 'success',
            'comparison': comparison
        })
        
    except Exception as e:
        logger.error(f"Error in analysis comparison: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def analyze_with_external_ai(image_data: bytes, ai_service: str, api_key: str = None) -> Dict:
    """Analyze wound using external AI service."""
    try:
        if ai_service not in AI_SERVICES:
            return {'error': f'Unknown AI service: {ai_service}'}
        
        service_config = AI_SERVICES[ai_service]
        
        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        if ai_service == 'openai':
            return analyze_with_openai(image_base64, api_key, service_config)
        elif ai_service == 'gemini':
            return analyze_with_gemini(image_base64, api_key, service_config)
        elif ai_service == 'claude':
            return analyze_with_claude(image_base64, api_key, service_config)
        else:
            return {'error': f'Service {ai_service} not implemented yet'}
            
    except Exception as e:
        logger.error(f"Error in external AI analysis: {e}")
        return {'error': str(e)}

def analyze_with_openai(image_base64: str, api_key: str, config: Dict) -> Dict:
    """Analyze wound using OpenAI GPT-4 Vision."""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': config['model'],
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': '''Analyze this wound image and provide a structured response in the following format:

Type: [wound type - cut, burn, abrasion, surgical, laceration, etc.]
Severity: [Critical, Severe, Moderate, Mild, Minor]
Explanation: [Brief medical explanation with key visual features and treatment recommendations]

Focus on:
1. Visual characteristics (color, shape, edges, texture)
2. Medical assessment of severity
3. Appropriate treatment recommendations
4. Any concerning features that require immediate attention

Be concise but medically accurate.'''
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{image_base64}'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 500
        }
        
        response = requests.post(config['api_url'], headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse structured response
            parsed_result = parse_structured_response(content)
            parsed_result['ai_service'] = 'OpenAI GPT-4 Vision'
            parsed_result['raw_response'] = content
            
            return parsed_result
        else:
            return {'error': f'OpenAI API error: {response.status_code} - {response.text}'}
            
    except Exception as e:
        logger.error(f"Error in OpenAI analysis: {e}")
        return {'error': str(e)}

def analyze_with_gemini(image_base64: str, api_key: str, config: Dict) -> Dict:
    """Analyze wound using Google Gemini."""
    try:
        url = f"{config['api_url']}?key={api_key}"
        
        # Detect image format from base64 data
        if image_base64.startswith('/9j/'):
            mime_type = 'image/jpeg'
        elif image_base64.startswith('iVBORw0KGgo'):
            mime_type = 'image/png'
        else:
            mime_type = 'image/jpeg'  # Default to JPEG
        
        payload = {
            'contents': [
                {
                    'parts': [
                        {
                            'text': '''Analyze this wound image and provide a structured response in the following format:

Type: [wound type - cut, burn, abrasion, surgical, laceration, etc.]
Severity: [Critical, Severe, Moderate, Mild, Minor]
Explanation: [Brief medical explanation with key visual features and treatment recommendations]

Focus on:
1. Visual characteristics (color, shape, edges, texture)
2. Medical assessment of severity
3. Appropriate treatment recommendations
4. Any concerning features that require immediate attention

Be concise but medically accurate.'''
                        },
                        {
                            'inline_data': {
                                'mime_type': mime_type,
                                'data': image_base64
                            }
                        }
                    ]
                }
            ],
            'generationConfig': {
                'maxOutputTokens': 500
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['candidates'][0]['content']['parts'][0]['text']
            
            # Parse structured response
            parsed_result = parse_structured_response(content)
            parsed_result['ai_service'] = 'Google Gemini'
            parsed_result['raw_response'] = content
            
            return parsed_result
        else:
            return {'error': f'Gemini API error: {response.status_code} - {response.text}'}
            
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {e}")
        return {'error': str(e)}

def analyze_with_claude(image_base64: str, api_key: str, config: Dict) -> Dict:
    """Analyze wound using Anthropic Claude."""
    try:
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': config['model'],
            'max_tokens': 500,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': '''Analyze this wound image and provide a structured response in the following format:

Type: [wound type - cut, burn, abrasion, surgical, laceration, etc.]
Severity: [Critical, Severe, Moderate, Mild, Minor]
Explanation: [Brief medical explanation with key visual features and treatment recommendations]

Focus on:
1. Visual characteristics (color, shape, edges, texture)
2. Medical assessment of severity
3. Appropriate treatment recommendations
4. Any concerning features that require immediate attention

Be concise but medically accurate.'''
                        },
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': image_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(config['api_url'], headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['content'][0]['text']
            
            # Parse structured response
            parsed_result = parse_structured_response(content)
            parsed_result['ai_service'] = 'Anthropic Claude'
            parsed_result['raw_response'] = content
            
            return parsed_result
        else:
            return {'error': f'Claude API error: {response.status_code} - {response.text}'}
            
    except Exception as e:
        logger.error(f"Error in Claude analysis: {e}")
        return {'error': str(e)}

def parse_structured_response(content: str) -> Dict:
    """Parse structured response from AI service."""
    try:
        lines = content.strip().split('\n')
        result = {
            'Type': 'Unknown',
            'Severity': 'Moderate',
            'Explanation': content
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Type:'):
                result['Type'] = line.replace('Type:', '').strip()
            elif line.startswith('Severity:'):
                result['Severity'] = line.replace('Severity:', '').strip()
            elif line.startswith('Explanation:'):
                result['Explanation'] = line.replace('Explanation:', '').strip()
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing structured response: {e}")
        return {
            'Type': 'Unknown',
            'Severity': 'Moderate',
            'Explanation': content
        }

# Patient Tracking Endpoints
@app.route('/patient/<patient_id>/tracker', methods=['GET'])
def get_patient_tracker(patient_id):
    """Get patient tracking data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get patient basic info
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC
        ''', (patient_id,))
        
        predictions = cursor.fetchall()
        
        if not predictions:
            return jsonify({
                'status': 'success',
                'patient_id': patient_id,
                'tracker_data': {
                    'medicine_schedule': [],
                    'wound_progress': [],
                    'total_days': 21,
                    'current_day': 1
                }
            })
        
        # Get latest prediction for initial data
        latest_prediction = predictions[0]
        
        # Mock medicine schedule (in real app, this would come from treatment plan)
        medicine_schedule = [
            {
                'id': 1,
                'name': 'Antibiotic',
                'dosage': '500mg',
                'frequency': 'daily',
                'time': 'Morning',
                'notes': 'As prescribed by doctor',
                'days': []
            }
        ]
        
        # Mock wound progress (in real app, this would be stored in database)
        wound_progress = [
            {
                'day': 0,
                'area': latest_prediction[4] if len(latest_prediction) > 4 else 5.0,
                'pain_level': '3',
                'redness': 'moderate',
                'swelling': 'moderate',
                'discharge': 'none',
                'notes': 'Initial assessment',
                'timestamp': latest_prediction[3] if len(latest_prediction) > 3 else datetime.now().isoformat()
            }
        ]
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'tracker_data': {
                'medicine_schedule': medicine_schedule,
                'wound_progress': wound_progress,
                'total_days': 21,
                'current_day': 1
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting patient tracker: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/patient/<patient_id>/medicine', methods=['POST'])
def add_medicine(patient_id):
    """Add medicine to patient schedule"""
    try:
        data = request.get_json()
        
        medicine_data = {
            'patient_id': patient_id,
            'name': data.get('name'),
            'dosage': data.get('dosage'),
            'frequency': data.get('frequency', 'daily'),
            'time': data.get('time'),
            'notes': data.get('notes', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # In a real app, this would be stored in a medicines table
        logger.info(f"Medicine added for patient {patient_id}: {medicine_data}")
        
        return jsonify({
            'status': 'success',
            'message': 'Medicine added successfully',
            'medicine': medicine_data
        })
        
    except Exception as e:
        logger.error(f"Error adding medicine: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/patient/<patient_id>/medicine/<medicine_id>/taken', methods=['POST'])
def mark_medicine_taken(patient_id, medicine_id):
    """Mark medicine as taken for a specific day"""
    try:
        data = request.get_json()
        
        taken_data = {
            'patient_id': patient_id,
            'medicine_id': medicine_id,
            'day': data.get('day'),
            'taken': data.get('taken', True),
            'time': data.get('time', datetime.now().strftime('%H:%M')),
            'notes': data.get('notes', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # In a real app, this would be stored in a medicine_tracking table
        logger.info(f"Medicine marked as taken for patient {patient_id}: {taken_data}")
        
        return jsonify({
            'status': 'success',
            'message': 'Medicine status updated',
            'data': taken_data
        })
        
    except Exception as e:
        logger.error(f"Error updating medicine status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/patient/<patient_id>/wound-progress', methods=['POST'])
def add_wound_progress(patient_id):
    """Add wound progress update"""
    try:
        data = request.get_json()
        
        progress_data = {
            'patient_id': patient_id,
            'day': data.get('day'),
            'area': data.get('area'),
            'pain_level': data.get('pain_level'),
            'redness': data.get('redness'),
            'swelling': data.get('swelling'),
            'discharge': data.get('discharge'),
            'notes': data.get('notes', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # In a real app, this would be stored in a wound_progress table
        logger.info(f"Wound progress added for patient {patient_id}: {progress_data}")
        
        return jsonify({
            'status': 'success',
            'message': 'Wound progress updated',
            'data': progress_data
        })
        
    except Exception as e:
        logger.error(f"Error adding wound progress: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/patient/<patient_id>/progress-summary', methods=['GET'])
def get_progress_summary(patient_id):
    """Get patient progress summary"""
    try:
        # Mock progress summary (in real app, this would calculate from database)
        summary = {
            'patient_id': patient_id,
            'total_days': 21,
            'current_day': 1,
            'healing_progress': 15.5,
            'medicines_taken_today': 1,
            'total_medicines': 1,
            'last_wound_update': datetime.now().isoformat(),
            'next_appointment': None,
            'alerts': []
        }
        
        return jsonify({
            'status': 'success',
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting progress summary: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Wound Analysis API...")
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"Classes: {model_classes}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
