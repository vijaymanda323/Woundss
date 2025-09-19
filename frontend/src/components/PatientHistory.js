import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PatientHistory.css';

const API_BASE_URL = 'http://localhost:5000';

const PatientHistory = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/history`);
      setHistory(response.data.history);
      setError(null);
    } catch (error) {
      console.error('Error fetching history:', error);
      setError('Failed to load patient history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getStatusIcon = (status) => {
    return status === 'right' ? '✅' : '❌';
  };

  const getStatusText = (status) => {
    return status === 'right' ? 'Correct' : 'Incorrect';
  };

  if (loading) {
    return (
      <div className="card">
        <h2 className="card-title">Patient History</h2>
        <div className="loading">
          <div className="spinner"></div>
          <span>Loading history...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h2 className="card-title">Patient History</h2>
        <div className="error-message">
          <p>{error}</p>
          <button className="btn btn-primary" onClick={fetchHistory}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="patient-history-container">
      <div className="card">
        <div className="history-header">
          <h2 className="card-title">Patient History</h2>
          <button className="btn btn-secondary" onClick={fetchHistory}>
            Refresh
          </button>
        </div>
        
        {history.length === 0 ? (
          <div className="empty-history">
            <div className="empty-icon">📋</div>
            <h3>No History Yet</h3>
            <p>Upload some wound images to see your analysis history here.</p>
          </div>
        ) : (
          <div className="history-list">
            {history.map((item, index) => (
              <div key={index} className="history-item">
                <img
                  src={`http://localhost:5000/${item.image_path}`}
                  alt="Wound analysis"
                  className="history-thumbnail"
                  onError={(e) => {
                    e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjRjVGNUY1Ii8+CjxwYXRoIGQ9Ik0yMCAyMEg0MFY0MEgyMFYyMFoiIGZpbGw9IiNDQ0MiLz4KPC9zdmc+';
                  }}
                />
                <div className="history-details">
                  <div className="history-label">
                    {item.predicted_label}
                  </div>
                  <div className="history-meta">
                    Confidence: {(item.confidence * 100).toFixed(1)}% • {formatDate(item.timestamp)}
                  </div>
                </div>
                <div className="history-status-container">
                  <span className={`history-status ${item.feedback_status}`}>
                    {getStatusIcon(item.feedback_status)} {getStatusText(item.feedback_status)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientHistory;


