import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  TextInput,
  Modal,
  ActivityIndicator,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import * as ImagePicker from 'expo-image-picker';

const AnalysisComparisonScreen = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [localAnalysis, setLocalAnalysis] = useState(null);
  const [externalAnalysis, setExternalAnalysis] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [selectedAiService, setSelectedAiService] = useState('openai');

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaType.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0]);
      setLocalAnalysis(null);
      setExternalAnalysis(null);
      setComparison(null);
    }
  };

  const analyzeImage = async (useExternalAi = false) => {
    if (!selectedImage) {
      Alert.alert('Error', 'Please select an image first');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('image', {
        uri: selectedImage.uri,
        type: 'image/jpeg',
        name: 'image.jpg',
      });

      let endpoint = 'http://localhost:5000/analyze-intelligent';
      
      if (useExternalAi) {
        if (!apiKey) {
          Alert.alert('Error', 'Please enter your API key');
          setLoading(false);
          return;
        }
        endpoint = 'http://localhost:5000/compare-analysis';
        formData.append('ai_service', selectedAiService);
        formData.append('api_key', apiKey);
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();

      if (response.ok) {
        if (useExternalAi) {
          setComparison(result.comparison);
          setLocalAnalysis(result.comparison.local_analysis);
          setExternalAnalysis(result.comparison.external_analysis);
        } else {
          setLocalAnalysis(result.analysis);
        }
      } else {
        Alert.alert('Error', result.error || 'Analysis failed');
      }
    } catch (error) {
      Alert.alert('Error', `Network error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisCard = (title, analysis, isExternal = false) => {
    if (!analysis) return null;

    const isError = analysis.error;
    const prediction = analysis.prediction || 'Unknown';
    const confidence = analysis.confidence || 0;
    const method = analysis.method || 'Unknown';

    return (
      <View style={styles.analysisCard}>
        <Text style={styles.cardTitle}>{title}</Text>
        {isError ? (
          <Text style={styles.errorText}>Error: {analysis.error}</Text>
        ) : (
          <>
            <View style={styles.predictionRow}>
              <Text style={styles.predictionLabel}>Prediction:</Text>
              <Text style={styles.predictionValue}>{prediction}</Text>
            </View>
            <View style={styles.confidenceRow}>
              <Text style={styles.confidenceLabel}>Confidence:</Text>
              <Text style={styles.confidenceValue}>{confidence.toFixed(3)}</Text>
            </View>
            <View style={styles.methodRow}>
              <Text style={styles.methodLabel}>Method:</Text>
              <Text style={styles.methodValue}>{method}</Text>
            </View>
            {analysis.analysis_details && (
              <View style={styles.detailsContainer}>
                <Text style={styles.detailsTitle}>Analysis Details:</Text>
                {analysis.analysis_details.color_analysis && (
                  <Text style={styles.detailText}>
                    Color (HSV): {analysis.analysis_details.color_analysis.hsv_mean?.map(v => v.toFixed(1)).join(', ')}
                  </Text>
                )}
                {analysis.analysis_details.texture_analysis && (
                  <Text style={styles.detailText}>
                    Edge Density: {analysis.analysis_details.texture_analysis.edge_density?.toFixed(3)}
                  </Text>
                )}
                {analysis.analysis_details.prediction_scores && (
                  <Text style={styles.detailText}>
                    All Scores: {JSON.stringify(analysis.analysis_details.prediction_scores)}
                  </Text>
                )}
              </View>
            )}
          </>
        )}
      </View>
    );
  };

  const renderComparison = () => {
    if (!comparison) return null;

    const comp = comparison.comparison;
    const match = comp.prediction_match;
    const confidenceDiff = comp.confidence_difference;

    return (
      <View style={styles.comparisonCard}>
        <Text style={styles.cardTitle}>Comparison Results</Text>
        <View style={styles.comparisonRow}>
          <Text style={styles.comparisonLabel}>Predictions Match:</Text>
          <Text style={[styles.comparisonValue, { color: match ? '#4CAF50' : '#F44336' }]}>
            {match ? 'Yes' : 'No'}
          </Text>
        </View>
        {confidenceDiff !== null && (
          <View style={styles.comparisonRow}>
            <Text style={styles.comparisonLabel}>Confidence Difference:</Text>
            <Text style={styles.comparisonValue}>{confidenceDiff.toFixed(3)}</Text>
          </View>
        )}
        <View style={styles.comparisonRow}>
          <Text style={styles.comparisonLabel}>Analysis Method:</Text>
          <Text style={styles.comparisonValue}>{comp.analysis_method}</Text>
        </View>
      </View>
    );
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Wound Analysis Comparison</Text>
      
      <TouchableOpacity style={styles.imageButton} onPress={pickImage}>
        <Text style={styles.buttonText}>
          {selectedImage ? 'Change Image' : 'Select Wound Image'}
        </Text>
      </TouchableOpacity>

      {selectedImage && (
        <View style={styles.imageContainer}>
          <Text style={styles.imageText}>Selected: {selectedImage.uri.split('/').pop()}</Text>
        </View>
      )}

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.analyzeButton, styles.localButton]}
          onPress={() => analyzeImage(false)}
          disabled={loading}
        >
          <Text style={styles.buttonText}>Analyze Locally</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.analyzeButton, styles.externalButton]}
          onPress={() => setShowApiKeyModal(true)}
          disabled={loading}
        >
          <Text style={styles.buttonText}>Compare with AI</Text>
        </TouchableOpacity>
      </View>

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2196F3" />
          <Text style={styles.loadingText}>Analyzing...</Text>
        </View>
      )}

      {renderAnalysisCard('Local Analysis', localAnalysis)}
      {renderAnalysisCard('External AI Analysis', externalAnalysis, true)}
      {renderComparison()}

      <Modal
        visible={showApiKeyModal}
        transparent={true}
        animationType="slide"
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>External AI Analysis</Text>
            
            <Text style={styles.inputLabel}>AI Service:</Text>
            <View style={styles.pickerContainer}>
              <Picker
                selectedValue={selectedAiService}
                onValueChange={setSelectedAiService}
                style={styles.picker}
              >
                <Picker.Item label="ChatGPT (OpenAI)" value="openai" />
                <Picker.Item label="Google Gemini" value="gemini" />
                <Picker.Item label="Anthropic Claude" value="claude" />
              </Picker>
            </View>

            <Text style={styles.inputLabel}>API Key:</Text>
            <TextInput
              style={styles.apiKeyInput}
              value={apiKey}
              onChangeText={setApiKey}
              placeholder="Enter your API key"
              secureTextEntry={true}
            />

            <View style={styles.modalButtonContainer}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => setShowApiKeyModal(false)}
              >
                <Text style={styles.modalButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.confirmButton]}
                onPress={() => {
                  setShowApiKeyModal(false);
                  analyzeImage(true);
                }}
              >
                <Text style={styles.modalButtonText}>Compare</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
    color: '#333',
  },
  imageButton: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
  },
  imageContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  imageText: {
    fontSize: 14,
    color: '#666',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  analyzeButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    marginHorizontal: 5,
  },
  localButton: {
    backgroundColor: '#4CAF50',
  },
  externalButton: {
    backgroundColor: '#FF9800',
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#666',
  },
  analysisCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  predictionLabel: {
    fontSize: 14,
    color: '#666',
  },
  predictionValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  methodRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  methodLabel: {
    fontSize: 14,
    color: '#666',
  },
  methodValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  detailsContainer: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  detailsTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  detailText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  errorText: {
    color: '#F44336',
    fontSize: 14,
  },
  comparisonCard: {
    backgroundColor: '#e3f2fd',
    padding: 15,
    borderRadius: 8,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#2196F3',
  },
  comparisonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 5,
  },
  comparisonLabel: {
    fontSize: 14,
    color: '#666',
  },
  comparisonValue: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 8,
    width: '90%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  pickerContainer: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 4,
    marginBottom: 15,
  },
  picker: {
    height: 50,
  },
  apiKeyInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 4,
    padding: 10,
    marginBottom: 20,
    fontSize: 14,
  },
  modalButtonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalButton: {
    flex: 1,
    padding: 15,
    borderRadius: 4,
    marginHorizontal: 5,
  },
  cancelButton: {
    backgroundColor: '#f44336',
  },
  confirmButton: {
    backgroundColor: '#4CAF50',
  },
  modalButtonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
  },
});

export default AnalysisComparisonScreen;

