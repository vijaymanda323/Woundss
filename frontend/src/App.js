import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ImageUpload from './components/ImageUpload';
import PatientHistory from './components/PatientHistory';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="nav-title">🏥 Wound Analysis App</h1>
            <div className="nav-links">
              <Link to="/" className="nav-link">Upload Image</Link>
              <Link to="/history" className="nav-link">Patient History</Link>
            </div>
          </div>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<ImageUpload />} />
            <Route path="/history" element={<PatientHistory />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;


