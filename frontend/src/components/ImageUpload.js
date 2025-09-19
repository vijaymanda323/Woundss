import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './ImageUpload.css';

const API_BASE_URL = 'http://localhost:5000';

const ImageUpload = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [showCorrectTypeSelector, setShowCorrectTypeSelector] = useState(false);
  const [selectedCorrectType, setSelectedCorrectType] = useState('');

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
        setPrediction(null);
        setFeedbackSent(false);
        predictWound(file);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false
  });

  const predictWound = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
      
      // Show if result was cached
      if (response.data.cached) {
        console.log('Using cached prediction');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error making prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (status) => {
    if (!prediction) return;

    try {
      if (status === 'wrong') {
        // Show wound type selector for incorrect predictions
        setShowCorrectTypeSelector(true);
        return;
      }

      // Handle correct prediction
      await submitFeedback('right', null);
      
    } catch (error) {
      console.error('Feedback error:', error);
      alert('Error sending feedback. Please try again.');
    }
  };

  const submitFeedback = async (status, correctType = null) => {
    try {
      const feedbackData = {
        image_hash: prediction.image_hash,
        feedback_status: status,
        correct_type: correctType,
        predicted_type: prediction.prediction
      };

      await axios.post(`${API_BASE_URL}/feedback`, feedbackData);

      setFeedbackSent(true);
      setShowCorrectTypeSelector(false);
      
      // Show success message
      if (status === 'wrong' && correctType) {
        alert(`Thank you! The model is learning that this wound type should be classified as "${correctType}" instead of "${prediction.prediction}".`);
      } else {
        alert('Thank you! Prediction marked as correct.');
      }
      
    } catch (error) {
      console.error('Feedback error:', error);
      alert('Error sending feedback. Please try again.');
    }
  };

  const handleCorrectTypeSelection = () => {
    if (!selectedCorrectType) {
      alert('Please select the correct wound type.');
      return;
    }
    
    submitFeedback('wrong', selectedCorrectType);
  };

  const resetUpload = () => {
    setImage(null);
    setPrediction(null);
    setFeedbackSent(false);
    setShowCorrectTypeSelector(false);
    setSelectedCorrectType('');
  };

  return (
    <div className="image-upload-container">
      <div className="card">
        <h2 className="card-title">Upload Wound Image</h2>
        
        {!image ? (
          <div
            {...getRootProps()}
            className={`upload-area ${isDragActive ? 'dragover' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="upload-icon">📷</div>
            <div className="upload-text">
              {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
            </div>
            <div className="upload-hint">or click to select a file</div>
          </div>
        ) : (
          <div className="upload-result">
            <img src={image} alt="Uploaded wound" className="image-preview" />
            
            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <span>Analyzing wound...</span>
              </div>
            )}
            
            {prediction && !loading && (
              <div className="prediction-card">
                <div className="prediction-label">
                  Predicted: {prediction.prediction}
                </div>
                <div className="prediction-confidence">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </div>
                {prediction.cached && (
                  <div className="cached-indicator">
                    📋 Using cached prediction
                  </div>
                )}
                
                {!feedbackSent && !showCorrectTypeSelector && (
                  <div className="feedback-buttons">
                    <button
                      className="feedback-btn right"
                      onClick={() => sendFeedback('right')}
                    >
                      ✅ Right
                    </button>
                    <button
                      className="feedback-btn wrong"
                      onClick={() => sendFeedback('wrong')}
                    >
                      ❌ Wrong
                    </button>
                  </div>
                )}

                {showCorrectTypeSelector && (
                  <div className="type-selector">
                    <h3>🤔 What is the correct wound type?</h3>
                    <p className="predicted-info">
                      Predicted: <span className="predicted-type">{prediction.prediction}</span>
                    </p>
                    
                    <div className="wound-type-grid">
                      {['burn', 'cut', 'surgical', 'chronic', 'diabetic', 'abrasion', 'laceration', 'pressure_ulcer'].map((type) => (
                        <button
                          key={type}
                          className={`type-button ${selectedCorrectType === type ? 'selected' : ''}`}
                          onClick={() => setSelectedCorrectType(type)}
                        >
                          {type.replace('_', ' ').toUpperCase()}
                        </button>
                      ))}
                    </div>
                    
                    <div className="selector-actions">
                      <button
                        className="action-btn back"
                        onClick={() => {
                          setShowCorrectTypeSelector(false);
                          setSelectedCorrectType('');
                        }}
                      >
                        ← Back
                      </button>
                      <button
                        className="action-btn submit"
                        onClick={handleCorrectTypeSelection}
                        disabled={!selectedCorrectType}
                      >
                        Submit Correction
                      </button>
                    </div>
                  </div>
                )}
                
                {feedbackSent && (
                  <div className="feedback-sent">
                    <p>Thank you for your feedback!</p>
                    <button className="btn btn-secondary" onClick={resetUpload}>
                      Upload Another Image
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;
