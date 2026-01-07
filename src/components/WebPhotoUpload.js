import React, { useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native';
import { Card, Title, Paragraph, Button } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';

export default function WebPhotoUpload({ onImageSelected, onUploadProgress }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  }, []);

  const handleFileSelection = async (file) => {
    if (!file.type.startsWith('image/')) {
      Alert.alert('Invalid File', 'Please select an image file.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      Alert.alert('File Too Large', 'Please select an image smaller than 10MB.');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simulate upload progress
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 100));
        setUploadProgress(i);
        if (onUploadProgress) {
          onUploadProgress(i);
        }
      }

      // Create object URL for preview and pass File for upload
      const imageUrl = URL.createObjectURL(file);
      if (onImageSelected) {
        onImageSelected({
          uri: imageUrl,
          name: file.name,
          type: file.type || 'image/jpeg',
          file,
        });
      }

    } catch (error) {
      console.error('Error processing file:', error);
      Alert.alert('Error', 'Failed to process the image file.');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelection(file);
    }
  };

  const openFileDialog = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  if (Platform.OS !== 'web') {
    return null; // Only render on web
  }

  return (
    <View style={styles.container}>
      <Card style={[styles.uploadCard, isDragOver && styles.dragOverCard]}>
        <Card.Content>
          <div
            style={styles.dropZone}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={openFileDialog}
          >
            <View style={styles.dropZoneContent}>
              <Ionicons 
                name={isDragOver ? "cloud-done" : "cloud-upload"} 
                size={60} 
                color={isDragOver ? "#27ae60" : "#667eea"} 
              />
              <Title style={styles.dropZoneTitle}>
                {isDragOver ? "Drop Image Here" : "Drag & Drop Image"}
              </Title>
              <Paragraph style={styles.dropZoneDescription}>
                {isDragOver 
                  ? "Release to upload" 
                  : "or click to browse files"
                }
              </Paragraph>
              
              {isUploading && (
                <View style={styles.progressContainer}>
                  <Text style={styles.progressText}>
                    Uploading... {uploadProgress}%
                  </Text>
                  <View style={styles.progressBar}>
                    <View 
                      style={[
                        styles.progressFill, 
                        { width: `${uploadProgress}%` }
                      ]} 
                    />
                  </View>
                </View>
              )}
            </View>
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            style={styles.hiddenInput}
          />
        </Card.Content>
      </Card>

      <Card style={styles.infoCard}>
        <Card.Content>
          <Title>📋 Supported Formats</Title>
          <View style={styles.formatList}>
            <View style={styles.formatItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.formatText}>JPEG (.jpg, .jpeg)</Text>
            </View>
            <View style={styles.formatItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.formatText}>PNG (.png)</Text>
            </View>
            <View style={styles.formatItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.formatText}>WebP (.webp)</Text>
            </View>
            <View style={styles.formatItem}>
              <Ionicons name="checkmark-circle" size={20} color="#27ae60" />
              <Text style={styles.formatText}>Maximum size: 10MB</Text>
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
  },
  uploadCard: {
    marginBottom: 20,
    elevation: 4,
    transition: 'all 0.3s ease',
  },
  dragOverCard: {
    borderColor: '#27ae60',
    borderWidth: 2,
    backgroundColor: '#e8f5e8',
  },
  dropZone: {
    border: '3px dashed #667eea',
    borderRadius: 15,
    padding: 40,
    textAlign: 'center',
    backgroundColor: '#f8f9fa',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    minHeight: 200,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dropZoneContent: {
    alignItems: 'center',
  },
  dropZoneTitle: {
    marginTop: 15,
    marginBottom: 10,
    color: '#2c3e50',
  },
  dropZoneDescription: {
    color: '#7f8c8d',
    fontSize: 16,
  },
  progressContainer: {
    marginTop: 20,
    width: '100%',
    maxWidth: 300,
  },
  progressText: {
    fontSize: 16,
    color: '#2c3e50',
    marginBottom: 10,
    textAlign: 'center',
  },
  progressBar: {
    height: 8,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#667eea',
    borderRadius: 4,
    transition: 'width 0.3s ease',
  },
  hiddenInput: {
    display: 'none',
  },
  infoCard: {
    elevation: 4,
  },
  formatList: {
    marginTop: 15,
  },
  formatItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  formatText: {
    marginLeft: 10,
    fontSize: 16,
    color: '#2c3e50',
  },
});




