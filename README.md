# Wound Analysis App with Real-time Learning

A full-stack application that analyzes wound images using machine learning and learns from user feedback to improve predictions over time.

## Features

- **Image Upload**: Drag & drop or click to upload wound images
- **Real-time Prediction**: Instant wound type classification with confidence scores
- **Smart Caching**: Same images return identical predictions instantly (no re-analysis)
- **User Feedback**: Mark predictions as "Right" or "Wrong" to improve the model
- **Patient History**: View all uploaded images with predictions and feedback status
- **Continuous Learning**: Model retrains automatically based on user feedback
- **Database Storage**: SQLite database stores all predictions and feedback
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

- **Frontend**: React.js with modern UI components
- **Backend**: Flask API with PyTorch machine learning model
- **Model**: ResNet18-based classifier trained on 1000+ wound images
- **Database**: SQLite for storing predictions and feedback
- **Caching**: SHA256 image hashing for instant duplicate detection
- **Learning**: Transfer learning with incremental updates

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install torch torchvision flask flask-cors pillow pandas scikit-learn opencv-python

# Start the Flask API server
python app.py
```

The backend will start on `http://localhost:5000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the React development server
npm start
```

The frontend will start on `http://localhost:3000`

### 3. Access the App

Open your browser and go to `http://localhost:3000`

## Usage

1. **Upload Image**: Drag and drop a wound image or click to select
2. **View Prediction**: See the predicted wound type and confidence score
3. **Cached Results**: Same images show "📋 Using cached prediction" for instant results
4. **Provide Feedback**: Click "Right" if correct or "Wrong" if incorrect
5. **View History**: Check the Patient History tab to see all analyses
6. **Consistent Predictions**: Same image always returns identical results

## API Endpoints

### Backend API (`http://localhost:5000`)

- `GET /health` - Health check
- `POST /predict` - Upload image and get prediction
- `POST /feedback` - Submit user feedback
- `GET /history` - Get patient history

### Example API Usage

```bash
# Predict wound type
curl -X POST -F "image=@wound.jpg" http://localhost:5000/predict

# Submit feedback
curl -X POST -H "Content-Type: application/json" \
  -d '{"image_path":"uploads/image.jpg","predicted_label":"burn","feedback_status":"right","confidence":0.95}' \
  http://localhost:5000/feedback
```

## Model Details

- **Architecture**: ResNet18 with custom classification head
- **Training Data**: 1000+ wound images across 22 wound types
- **Classes**: burn, cut, laceration, abrasion, bruise, pressure_ulcer, leg_ulcer, foot_ulcer, etc.
- **Accuracy**: 70%+ validation accuracy, 100% burn detection accuracy
- **Learning**: Incremental fine-tuning based on user feedback

## File Structure

```
├── backend/
│   ├── app.py                 # Flask API server
│   ├── models/                # Trained model files
│   ├── uploads/               # Uploaded images
│   └── feedback_data/         # User feedback data
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.js
│   │   │   └── PatientHistory.js
│   │   ├── App.js
│   │   └── App.css
│   ├── public/
│   └── package.json
└── README.md
```

## Troubleshooting

### Backend Issues

1. **Model not loading**: Ensure `models/wound_classification_model.pth` exists
2. **Port conflicts**: Change port in `app.py` if 5000 is occupied
3. **Dependencies**: Install all required Python packages

### Frontend Issues

1. **API connection**: Ensure backend is running on port 5000
2. **CORS errors**: Backend has CORS enabled for localhost:3000
3. **Build issues**: Try `npm install` and `npm start` again

### Common Solutions

```bash
# Kill existing processes
taskkill /F /IM python.exe
taskkill /F /IM node.exe

# Restart backend
cd backend && python app.py

# Restart frontend
cd frontend && npm start
```

## Development

### Adding New Wound Types

1. Add images to `datasets/` directory
2. Update `labels.csv` with new wound types
3. Retrain model using `train_improved_model.py`
4. Update frontend to handle new classes

### Customizing the Model

- Modify `ImprovedWoundClassifier` in `backend/app.py`
- Adjust training parameters in `train_improved_model.py`
- Update preprocessing in `preprocess_image()` function

## Performance

- **Prediction Time**: ~200ms per new image, ~20ms for cached images
- **Model Size**: ~45MB
- **Memory Usage**: ~200MB RAM
- **Database**: SQLite with automatic indexing for fast lookups
- **Caching**: SHA256 hashing for instant duplicate detection
- **Concurrent Users**: Supports multiple simultaneous predictions

## Security Notes

- This is a development/demo application
- Images are stored locally and not encrypted
- No authentication or user management
- Not suitable for production use without security enhancements

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub