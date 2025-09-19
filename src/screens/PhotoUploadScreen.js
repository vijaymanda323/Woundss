import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Image,
  Dimensions,
  Platform,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { Button, Card, Title, Paragraph, ProgressBar } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import WebPhotoUpload from '../components/WebPhotoUpload';

const { width, height } = Dimensions.get('window');

export default function PhotoUploadScreen({ navigation, route }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [imageSource, setImageSource] = useState(null); // 'camera', 'gallery', 'files'

  const handleImageSelection = (imageUri, source) => {
    setSelectedImage(imageUri);
    setImageSource(source);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  };

  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Camera Permission Required',
          'Please grant camera permission to take photos.',
          [{ text: 'OK' }]
        );
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
        exif: false,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        handleImageSelection(result.assets[0].uri, 'camera');
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo. Please try again.');
    }
  };

  const pickFromGallery = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'Please grant permission to access your photo library.',
          [{ text: 'OK' }]
        );
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
        allowsMultipleSelection: false,
        exif: false,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        handleImageSelection(result.assets[0].uri, 'gallery');
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image from gallery. Please try again.');
    }
  };

  const pickFromFiles = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'image/*',
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        handleImageSelection(result.assets[0].uri, 'files');
      }
    } catch (error) {
      console.error('Error picking file:', error);
      Alert.alert('Error', 'Failed to pick file. Please try again.');
    }
  };

  const simulateUpload = async () => {
    if (!selectedImage) return;

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setUploadProgress(i);
    }

    setIsUploading(false);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

    // Navigate directly to analysis results with mock data
    const woundTypes = ['burn', 'cut', 'surgical', 'chronic', 'diabetic'];
    const randomType = woundTypes[Math.floor(Math.random() * woundTypes.length)];
    const randomArea = parseFloat((Math.random() * 10 + 1).toFixed(2));
    const randomHealingTime = Math.floor(Math.random() * 60) + 7;
    
    const mockAnalysisResult = {
      wound_classification: {
        wound_type: randomType,
        estimated_days_to_cure: randomHealingTime,
        healing_time_category: 'moderate_healing',
        model_available: true,
      },
      area_cm2: randomArea,
      area_pixels: Math.floor(Math.random() * 2000 + 500),
      perimeter: (Math.random() * 200 + 50).toFixed(2),
      model_confidence: parseFloat((Math.random() * 0.3 + 0.7).toFixed(2)),
    };

    navigation.navigate('AnalysisResults', { 
      imageUri: selectedImage,
      analysisResult: mockAnalysisResult,
      imageSource: imageSource 
    });
  };

  const retakeImage = () => {
    setSelectedImage(null);
    setImageSource(null);
    setUploadProgress(0);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const getImageSourceIcon = (source) => {
    switch (source) {
      case 'camera': return 'camera';
      case 'gallery': return 'images';
      case 'files': return 'folder';
      default: return 'image';
    }
  };

  const getImageSourceText = (source) => {
    switch (source) {
      case 'camera': return 'Camera';
      case 'gallery': return 'Gallery';
      case 'files': return 'Files';
      default: return 'Unknown';
    }
  };

  return (
    <View style={styles.container}>
      <Card style={styles.headerCard}>
        <Card.Content>
          <Title>📷 Upload Wound Photo</Title>
          <Paragraph>
            Choose how you'd like to capture or select your wound image for analysis.
          </Paragraph>
        </Card.Content>
      </Card>

      {!selectedImage ? (
        <View style={styles.uploadOptions}>
          {Platform.OS === 'web' ? (
            <WebPhotoUpload 
              onImageSelected={handleImageSelection}
              onUploadProgress={setUploadProgress}
            />
          ) : (
            <>
              <Card style={styles.optionCard}>
                <Card.Content>
                  <TouchableOpacity style={styles.optionButton} onPress={takePhoto}>
                    <Ionicons name="camera" size={60} color="#3498db" />
                    <Text style={styles.optionTitle}>Take Photo</Text>
                    <Text style={styles.optionDescription}>
                      Use your device camera to capture a new wound image
                    </Text>
                  </TouchableOpacity>
                </Card.Content>
              </Card>

              <Card style={styles.optionCard}>
                <Card.Content>
                  <TouchableOpacity style={styles.optionButton} onPress={pickFromGallery}>
                    <Ionicons name="images" size={60} color="#9b59b6" />
                    <Text style={styles.optionTitle}>Choose from Gallery</Text>
                    <Text style={styles.optionDescription}>
                      Select an existing photo from your device gallery
                    </Text>
                  </TouchableOpacity>
                </Card.Content>
              </Card>

              <Card style={styles.optionCard}>
                <Card.Content>
                  <TouchableOpacity style={styles.optionButton} onPress={pickFromFiles}>
                    <Ionicons name="folder" size={60} color="#e67e22" />
                    <Text style={styles.optionTitle}>Browse Files</Text>
                    <Text style={styles.optionDescription}>
                      Select an image file from your device storage
                    </Text>
                  </TouchableOpacity>
                </Card.Content>
              </Card>
            </>
          )}
        </View>
      ) : (
        <View style={styles.imagePreview}>
          <Card style={styles.previewCard}>
            <Card.Content>
              <View style={styles.imageContainer}>
                <Image source={{ uri: selectedImage }} style={styles.previewImage} />
                <View style={styles.imageInfo}>
                  <Ionicons name={getImageSourceIcon(imageSource)} size={20} color="#667eea" />
                  <Text style={styles.imageSourceText}>
                    From {getImageSourceText(imageSource)}
                  </Text>
                </View>
              </View>

              {isUploading && (
                <View style={styles.uploadProgress}>
                  <Text style={styles.progressText}>Uploading image...</Text>
                  <ProgressBar
                    progress={uploadProgress / 100}
                    color="#667eea"
                    style={styles.progressBar}
                  />
                  <Text style={styles.progressPercent}>{uploadProgress}%</Text>
                </View>
              )}

              <View style={styles.actionButtons}>
                <Button
                  mode="outlined"
                  onPress={retakeImage}
                  style={styles.actionButton}
                  icon="camera-retake"
                  disabled={isUploading}
                >
                  Retake
                </Button>
                
                <Button
                  mode="contained"
                  onPress={simulateUpload}
                  style={[styles.actionButton, { backgroundColor: '#667eea' }]}
                  icon="arrow-right"
                  loading={isUploading}
                  disabled={isUploading}
                >
                  Continue
                </Button>
              </View>
            </Card.Content>
          </Card>
        </View>
      )}

      <Card style={styles.tipsCard}>
        <Card.Content>
          <Title>📸 Photo Tips</Title>
          <View style={styles.tipsList}>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.tipText}>Ensure good lighting for clear visibility</Text>
            </View>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.tipText}>Position the wound in the center of the frame</Text>
            </View>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.tipText}>Keep the camera steady to avoid blur</Text>
            </View>
            <View style={styles.tipItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.tipText}>Include some surrounding skin for context</Text>
            </View>
          </View>
        </Card.Content>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 15,
  },
  headerCard: {
    marginBottom: 20,
    elevation: 4,
  },
  uploadOptions: {
    flex: 1,
  },
  optionCard: {
    marginBottom: 15,
    elevation: 4,
  },
  optionButton: {
    alignItems: 'center',
    padding: 20,
  },
  optionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginTop: 15,
    marginBottom: 10,
  },
  optionDescription: {
    fontSize: 16,
    color: '#7f8c8d',
    textAlign: 'center',
    lineHeight: 22,
  },
  imagePreview: {
    flex: 1,
  },
  previewCard: {
    elevation: 4,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  previewImage: {
    width: width - 60,
    height: 300,
    resizeMode: 'cover',
    borderRadius: 15,
    marginBottom: 15,
  },
  imageInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e3f2fd',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  imageSourceText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#667eea',
    fontWeight: '600',
  },
  uploadProgress: {
    marginBottom: 20,
  },
  progressText: {
    fontSize: 16,
    color: '#2c3e50',
    marginBottom: 10,
    textAlign: 'center',
  },
  progressBar: {
    height: 8,
    marginBottom: 10,
  },
  progressPercent: {
    fontSize: 14,
    color: '#667eea',
    textAlign: 'center',
    fontWeight: '600',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionButton: {
    flex: 0.48,
  },
  tipsCard: {
    marginTop: 20,
    elevation: 4,
  },
  tipsList: {
    marginTop: 15,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  tipText: {
    marginLeft: 10,
    fontSize: 16,
    color: '#2c3e50',
    flex: 1,
  },
});
