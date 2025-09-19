#!/usr/bin/env python3
"""
Improved Healing Time Predictor
===============================

Dynamic healing time prediction based on wound state and progress.
"""

import numpy as np
import cv2
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DynamicHealingPredictor:
    """Predicts healing time based on wound characteristics and progress."""
    
    def __init__(self):
        # Base healing times by wound type (in days)
        self.base_healing_times = {
            'abrasion': 7,
            'cut': 7,
            'surgical': 7,
            'laceration': 14,
            'bruise': 14,
            'ingrown_nail': 14,
            'stab_wound': 14,
            'burn': 21,
            'chronic': 60,
            'diabetic': 90,
            'unknown': 30
        }
        
        # Healing rate factors based on wound characteristics
        self.healing_factors = {
            'size_factor': {
                'small': 0.8,    # < 5 cm²
                'medium': 1.0,   # 5-20 cm²
                'large': 1.3,     # > 20 cm²
                'very_large': 1.6 # > 50 cm²
            },
            'severity_factor': {
                'mild': 0.7,
                'moderate': 1.0,
                'severe': 1.4
            },
            'location_factor': {
                'face': 0.8,      # Faster healing
                'limbs': 1.0,     # Normal
                'torso': 1.1,     # Slightly slower
                'feet': 1.3       # Slower healing
            }
        }
    
    def analyze_wound_characteristics(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze wound characteristics from image and mask."""
        
        # Calculate basic metrics
        area_pixels = np.sum(mask > 0)
        area_cm2 = area_pixels / (50**2)  # Assuming 50 pixels per cm
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'area_cm2': area_cm2,
                'size_category': 'small',
                'severity': 'mild',
                'shape_regularity': 1.0,
                'edge_clarity': 1.0
            }
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape regularity (closer to circle = more regular)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area_pixels / (perimeter ** 2)
        else:
            circularity = 0
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        
        # Determine size category
        if area_cm2 < 5:
            size_category = 'small'
        elif area_cm2 < 20:
            size_category = 'medium'
        elif area_cm2 < 50:
            size_category = 'large'
        else:
            size_category = 'very_large'
        
        # Determine severity based on shape irregularity and size
        if circularity > 0.7 and area_cm2 < 10:
            severity = 'mild'
        elif circularity > 0.4 and area_cm2 < 30:
            severity = 'moderate'
        else:
            severity = 'severe'
        
        # Analyze edge clarity (sharp edges = better healing)
        edge_clarity = self._analyze_edge_clarity(image, mask)
        
        return {
            'area_cm2': area_cm2,
            'size_category': size_category,
            'severity': severity,
            'shape_regularity': circularity,
            'edge_clarity': edge_clarity,
            'aspect_ratio': aspect_ratio
        }
    
    def _analyze_edge_clarity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Analyze edge clarity of the wound."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Get gradient values at wound edges
            edge_pixels = cv2.Canny(mask, 50, 150)
            edge_gradients = gradient_magnitude[edge_pixels > 0]
            
            if len(edge_gradients) > 0:
                # Higher gradient = sharper edges = better healing
                clarity_score = np.mean(edge_gradients) / 100.0
                return min(1.0, max(0.1, clarity_score))
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Error analyzing edge clarity: {e}")
            return 0.5
    
    def calculate_healing_progress(self, current_area: float, previous_records: list) -> Dict[str, Any]:
        """Calculate healing progress from previous records."""
        
        if not previous_records or len(previous_records) < 2:
            return {
                'healing_rate': 0,
                'progress_percentage': 0,
                'days_since_injury': 0,
                'trend': 'unknown'
            }
        
        # Sort records by timestamp (newest first)
        sorted_records = sorted(previous_records, key=lambda x: x['timestamp'], reverse=True)
        
        # Get most recent previous record
        previous_record = sorted_records[1] if len(sorted_records) > 1 else sorted_records[0]
        
        # Calculate healing rate (area reduction per day)
        try:
            current_date = datetime.fromisoformat(sorted_records[0]['timestamp'].replace('Z', '+00:00'))
            previous_date = datetime.fromisoformat(previous_record['timestamp'].replace('Z', '+00:00'))
            days_between = (current_date - previous_date).days
            
            if days_between > 0:
                area_reduction = previous_record['area_cm2'] - current_area
                healing_rate = area_reduction / days_between  # cm² per day
                
                # Calculate progress percentage
                if previous_record['area_cm2'] > 0:
                    progress_percentage = (area_reduction / previous_record['area_cm2']) * 100
                else:
                    progress_percentage = 0
                
                # Determine trend
                if healing_rate > 0.5:
                    trend = 'healing_well'
                elif healing_rate > 0:
                    trend = 'healing_slowly'
                elif healing_rate > -0.2:
                    trend = 'stable'
                else:
                    trend = 'worsening'
            else:
                healing_rate = 0
                progress_percentage = 0
                trend = 'unknown'
                
        except Exception as e:
            logger.warning(f"Error calculating healing progress: {e}")
            healing_rate = 0
            progress_percentage = 0
            trend = 'unknown'
        
        # Calculate days since injury (from oldest record)
        try:
            oldest_record = sorted_records[-1]
            injury_date = datetime.fromisoformat(oldest_record['timestamp'].replace('Z', '+00:00'))
            current_date = datetime.fromisoformat(sorted_records[0]['timestamp'].replace('Z', '+00:00'))
            days_since_injury = (current_date - injury_date).days
        except:
            days_since_injury = 0
        
        return {
            'healing_rate': healing_rate,
            'progress_percentage': progress_percentage,
            'days_since_injury': days_since_injury,
            'trend': trend
        }
    
    def predict_healing_time(self, 
                           wound_type: str, 
                           wound_characteristics: Dict[str, Any],
                           healing_progress: Dict[str, Any],
                           patient_age: Optional[int] = None,
                           wound_location: str = 'limbs') -> Dict[str, Any]:
        """Predict dynamic healing time based on multiple factors."""
        
        # Start with base healing time
        base_days = self.base_healing_times.get(wound_type, 30)
        
        # Apply size factor
        size_factor = self.healing_factors['size_factor'].get(
            wound_characteristics['size_category'], 1.0
        )
        
        # Apply severity factor
        severity_factor = self.healing_factors['severity_factor'].get(
            wound_characteristics['severity'], 1.0
        )
        
        # Apply location factor
        location_factor = self.healing_factors['location_factor'].get(wound_location, 1.0)
        
        # Apply shape regularity factor (more regular = faster healing)
        shape_factor = 0.8 + (wound_characteristics['shape_regularity'] * 0.4)
        
        # Apply edge clarity factor (sharper edges = faster healing)
        edge_factor = 0.7 + (wound_characteristics['edge_clarity'] * 0.6)
        
        # Calculate adjusted healing time
        adjusted_days = base_days * size_factor * severity_factor * location_factor * shape_factor * edge_factor
        
        # Adjust based on healing progress
        if healing_progress['days_since_injury'] > 0:
            # If wound has been healing, adjust prediction
            if healing_progress['trend'] == 'healing_well':
                # Reduce remaining time by 20%
                adjusted_days *= 0.8
            elif healing_progress['trend'] == 'healing_slowly':
                # Increase remaining time by 20%
                adjusted_days *= 1.2
            elif healing_progress['trend'] == 'worsening':
                # Increase remaining time by 50%
                adjusted_days *= 1.5
            
            # If significant progress has been made, reduce remaining time
            if healing_progress['progress_percentage'] > 50:
                progress_factor = 1.0 - (healing_progress['progress_percentage'] / 200.0)
                adjusted_days *= max(0.3, progress_factor)
        
        # Apply age factor (older patients heal slower)
        if patient_age:
            if patient_age > 65:
                age_factor = 1.3
            elif patient_age > 50:
                age_factor = 1.1
            else:
                age_factor = 1.0
            adjusted_days *= age_factor
        
        # Ensure minimum healing time
        adjusted_days = max(1, int(adjusted_days))
        
        # Determine confidence based on available data
        confidence = 0.7  # Base confidence
        if healing_progress['days_since_injury'] > 0:
            confidence += 0.2  # Higher confidence with progress data
        if wound_characteristics['shape_regularity'] > 0.5:
            confidence += 0.1  # Higher confidence with regular shapes
        
        confidence = min(0.95, confidence)
        
        # Determine healing category
        if adjusted_days <= 7:
            healing_category = 'fast_healing'
        elif adjusted_days <= 21:
            healing_category = 'moderate_healing'
        elif adjusted_days <= 60:
            healing_category = 'slow_healing'
        else:
            healing_category = 'chronic_non_healing'
        
        return {
            'estimated_days_to_cure': adjusted_days,
            'healing_time_category': healing_category,
            'confidence': confidence,
            'base_days': base_days,
            'adjustment_factors': {
                'size_factor': size_factor,
                'severity_factor': severity_factor,
                'location_factor': location_factor,
                'shape_factor': shape_factor,
                'edge_factor': edge_factor
            },
            'healing_progress': healing_progress,
            'wound_characteristics': wound_characteristics
        }

def test_healing_predictor():
    """Test the healing predictor with sample data."""
    
    predictor = DynamicHealingPredictor()
    
    # Simulate wound characteristics
    wound_chars = {
        'area_cm2': 15.5,
        'size_category': 'medium',
        'severity': 'moderate',
        'shape_regularity': 0.6,
        'edge_clarity': 0.7,
        'aspect_ratio': 1.2
    }
    
    # Simulate healing progress
    healing_progress = {
        'healing_rate': 0.8,  # cm² per day
        'progress_percentage': 25,
        'days_since_injury': 5,
        'trend': 'healing_well'
    }
    
    # Test different wound types
    wound_types = ['burn', 'cut', 'chronic', 'abrasion']
    
    print("🔍 Dynamic Healing Time Predictor Test")
    print("=" * 50)
    
    for wound_type in wound_types:
        result = predictor.predict_healing_time(
            wound_type=wound_type,
            wound_characteristics=wound_chars,
            healing_progress=healing_progress,
            patient_age=45,
            wound_location='limbs'
        )
        
        print(f"\n🏥 {wound_type.title()} Wound:")
        print(f"   Base healing time: {result['base_days']} days")
        print(f"   Predicted healing time: {result['estimated_days_to_cure']} days")
        print(f"   Category: {result['healing_time_category']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Progress: {healing_progress['progress_percentage']:.1f}% in {healing_progress['days_since_injury']} days")

if __name__ == "__main__":
    test_healing_predictor()




