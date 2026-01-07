import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Dimensions,
  Alert,
  Platform,
  ActivityIndicator,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { Card, Title, Paragraph, Button, TextInput, Switch, Divider } from "react-native-paper";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import WebPhotoUpload from "../components/WebPhotoUpload";
import { fetchWithTimeout } from "../services/apiService";

const { width } = Dimensions.get("window");
const API_BASE_URL = "https://woundanalysis.onrender.com";

const buildImagePart = (image) => {
  if (!image) return null;
  if (Platform.OS === "web" && image.file) {
    return image.file;
  }
  return {
    uri: image.uri,
    name: image.name || "wound_image.jpg",
    type: image.type || "image/jpeg",
  };
};

export default function PhotoUploadScreen() {
  // Patient state
  const [patient, setPatient] = useState({
    name: "",
    dob: "",
    gender: "",
  });
  const [patientData, setPatientData] = useState(null);
  const [isCreatingPatient, setIsCreatingPatient] = useState(false);
  const [showPatientForm, setShowPatientForm] = useState(true);

  // Case state
  const [caseData, setCaseData] = useState(null);
  const [isCreatingCase, setIsCreatingCase] = useState(false);

  // Images state
  const [images, setImages] = useState([]);
  const [isUploadingImages, setIsUploadingImages] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Context state
  const [context, setContext] = useState({
    isDiabetic: false,
    woundType: "",
    durationDays: "",
    recentAntibiotics: false,
    notes: "",
  });
  const [isSubmittingContext, setIsSubmittingContext] = useState(false);
  const [isContextSubmitted, setIsContextSubmitted] = useState(false);
  const [showContextForm, setShowContextForm] = useState(true);

  // Results state
  const [result, setResult] = useState(null);
  const [isFetchingResult, setIsFetchingResult] = useState(false);

  // Step 1: Create Patient
  const createPatient = async () => {
    if (!patient.name || !patient.dob || !patient.gender) {
      Alert.alert("Required Fields", "Please fill in Name, Date of Birth, and Gender.");
        return;
      }

    setIsCreatingPatient(true);
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      }, 60000);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Failed to create patient: ${text}`);
      }

      const data = await response.json();
      setPatientData(data);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      
      // Automatically create case after patient is created
      await createCase(data.id);
    } catch (error) {
      console.error("Create patient error:", error);
      Alert.alert("Error", error.message || "Failed to create patient. Please try again.");
    } finally {
      setIsCreatingPatient(false);
    }
  };

  // Step 2: Create Case (automatically called after patient creation)
  const createCase = async (patientId) => {
    setIsCreatingCase(true);
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/cases`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patientId }),
      }, 60000);

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Failed to create case: ${text}`);
      }

      const data = await response.json();
      console.log("Case created successfully:", JSON.stringify(data, null, 2));
      
      // Validate that the case response has the required 'id' field
      if (!data || !data.id) {
        throw new Error("Case creation response missing 'id' field. Response: " + JSON.stringify(data));
      }
      
      console.log("Case ID to use for all subsequent API calls:", data.id);
      console.log("Full case data structure:", {
        id: data.id,
        patientId: data.patientId,
        patientName: data.patientName,
        status: data.status
      });
      
      // Store the full case response which includes: id, patientId, patientName, createdAt, status, imagePaths
      // The 'id' field is the ONLY identifier to use for images, context, and results endpoints
      setCaseData(data);
      setShowPatientForm(false);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert("Success", `Patient registered and case created (ID: ${data.id.substring(0, 8)}...)`);
    } catch (error) {
      console.error("Create case error:", error);
      Alert.alert("Error", error.message || "Failed to create case. Please try again.");
    } finally {
      setIsCreatingCase(false);
    }
  };

  // Image selection handlers
  const handleAddImage = async (source) => {
    try {
      if (images.length >= 3) {
        Alert.alert("Limit Reached", "Maximum 3 images allowed.");
        return;
      }

      let picked;
      if (source === "camera") {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== "granted") {
          Alert.alert("Permission Required", "Camera access is required.");
          return;
        }
        picked = await ImagePicker.launchCameraAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          quality: 0.8,
        });
      } else if (source === "gallery") {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== "granted") {
          Alert.alert("Permission Required", "Photo library access is required.");
          return;
        }
        picked = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          quality: 0.8,
          allowsMultipleSelection: false,
        });
      } else if (source === "files") {
        picked = await DocumentPicker.getDocumentAsync({
          type: "image/*",
        copyToCacheDirectory: true,
      });
      }

      if (!picked || picked.canceled) return;

      const asset = picked.assets ? picked.assets[0] : picked;
      const imageData = {
        uri: asset.uri,
        name: asset.fileName || asset.name || "wound_image.jpg",
        type: asset.mimeType || asset.type || "image/jpeg",
        file: asset.file || null,
      };

      setImages((prev) => [...prev, imageData]);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    } catch (error) {
      console.error("Image pick error:", error);
      Alert.alert("Error", "Unable to select image. Please try again.");
    }
  };

  const handleWebImageSelected = (imageData) => {
    if (images.length >= 3) {
      Alert.alert("Limit Reached", "Maximum 3 images allowed.");
      return;
    }
    setImages((prev) => [...prev, imageData]);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  };

  const removeImage = (index) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  // Step 3: Upload Images
  const uploadImages = async () => {
    if (!caseData || !caseData.id) {
      Alert.alert("No Case", "Please create a patient and case first.");
      return;
    }

    if (images.length !== 3) {
      Alert.alert("Image Requirement", "Please upload exactly 3 wound images.");
      return;
    }

    setIsUploadingImages(true);
    setUploadProgress(0);

    try {
      // Use caseData.id from case creation response (NOT patientId or taskId)
      const caseId = caseData.id;
      console.log("Uploading images for case ID:", caseId);
      console.log("Full case data:", JSON.stringify(caseData, null, 2));
      
      if (!caseId || typeof caseId !== 'string') {
        throw new Error(`Invalid case ID: ${caseId}. Case data: ${JSON.stringify(caseData)}`);
      }
      
      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const formData = new FormData();
        const imagePart = buildImagePart(img);
        if (imagePart) {
          formData.append("images", imagePart);
        }

        const url = `${API_BASE_URL}/cases/${caseId}/images`;
        console.log(`Uploading image ${i + 1} to: ${url}`);
        
        const response = await fetchWithTimeout(
          url,
          {
            method: "POST",
            body: formData,
          },
          300000
        );

        if (!response.ok) {
          const text = await response.text();
          console.error(`Image upload failed. Status: ${response.status}, Response: ${text}`);
          throw new Error(`Failed to upload image ${i + 1} (Status ${response.status}): ${text}`);
        }

        // Some APIs return empty responses, so try to parse JSON if possible
        try {
          const responseData = await response.json();
          console.log(`Image ${i + 1} uploaded successfully:`, responseData);
        } catch (e) {
          console.log(`Image ${i + 1} uploaded successfully (no response body)`);
        }
        setUploadProgress(((i + 1) / images.length) * 100);
      }

      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert("Success", "Images submitted successfully.");
    } catch (error) {
      console.error("Upload images error:", error);
      Alert.alert("Upload Failed", error.message || "Failed to upload images. Please try again.");
    } finally {
      setIsUploadingImages(false);
      setUploadProgress(0);
    }
  };

  // Step 4: Submit Context
  const submitContext = async () => {
    if (!caseData || !caseData.id) {
      Alert.alert("No Case", "Please create a case first.");
      return;
    }

    setIsSubmittingContext(true);
    try {
      // Use caseData.id from case creation response (NOT patientId or taskId)
      const caseId = caseData.id;
      console.log("Submitting context for case ID:", caseId);
      console.log("Full case data:", JSON.stringify(caseData, null, 2));
      
      if (!caseId || typeof caseId !== 'string') {
        throw new Error(`Invalid case ID: ${caseId}. Case data: ${JSON.stringify(caseData)}`);
      }
      
      // Calculate patient age from DOB
      const calculateAge = (dob) => {
        if (!dob) return 0;
        const birth = new Date(dob);
        const today = new Date();
        return today.getFullYear() - birth.getFullYear();
      };

      const contextPayload = {
        isDiabetic: context.isDiabetic ? "true" : "false",
        woundType: context.woundType || "",
        durationDays: context.durationDays ? Number(context.durationDays) : 0,
        recentAntibiotics: context.recentAntibiotics ? "true" : "false",
        patientAge: calculateAge(patient.dob),
        notes: context.notes || "",
      };

      const url = `${API_BASE_URL}/cases/${caseId}/context`;
      console.log("Submitting context to:", url);
      console.log("Context payload:", JSON.stringify(contextPayload, null, 2));

      const response = await fetchWithTimeout(
        url,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(contextPayload),
        },
        60000
      );

      if (!response.ok) {
        const text = await response.text();
        console.error(`Context submission failed. Status: ${response.status}, Response: ${text}`);
        throw new Error(`Failed to submit context (Status ${response.status}): ${text}`);
      }
      
      const responseData = await response.json();
      console.log("Context submitted successfully:", responseData);

      setIsContextSubmitted(true);
      setShowContextForm(false);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert("Success", "Context submitted successfully.");
    } catch (error) {
      console.error("Submit context error:", error);
      Alert.alert("Error", error.message || "Failed to submit context. Please try again.");
    } finally {
      setIsSubmittingContext(false);
    }
  };

  // Step 5: Get Results (with polling)
  const fetchResult = async () => {
    if (!caseData || !caseData.id) {
      Alert.alert("No Case", "Please create a case first.");
      return;
    }

    setIsFetchingResult(true);
    setResult(null);

    try {
      // Use caseData.id from case creation response (NOT patientId or taskId)
      const caseId = caseData.id;
      console.log("Starting to poll results for case ID:", caseId);
      
      if (!caseId || typeof caseId !== 'string') {
        throw new Error(`Invalid case ID: ${caseId}. Case data: ${JSON.stringify(caseData)}`);
      }
      
      const url = `${API_BASE_URL}/cases/${caseId}/results`;
      const maxRetries = 20;
      const pollInterval = 3000; // 3 seconds
      let attempts = 0;
      
      const pollResults = async () => {
        while (attempts < maxRetries) {
          attempts++;
          console.log(`Polling attempt ${attempts}/${maxRetries}...`);
          
          try {
            // Use AbortController for better timeout handling
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout per request
            
            const response = await fetch(url, {
              method: "GET",
              signal: controller.signal,
              headers: {
                'Accept': 'application/json',
              },
            });
            
            clearTimeout(timeoutId);

            if (!response.ok) {
              const text = await response.text();
              console.error(`Results fetch failed. Status: ${response.status}, Response: ${text}`);
              // For non-200 status codes, wait and retry
              if (attempts < maxRetries) {
                console.log("Retrying after error...");
                await new Promise(resolve => setTimeout(resolve, pollInterval));
                continue;
              }
              throw new Error(`Failed to fetch results (Status ${response.status}): ${text}`);
            }

            const data = await response.json();
            console.log(`Attempt ${attempts} - Response:`, JSON.stringify(data, null, 2));
            
            // Check status
            const status = data.status;
            console.log(`Status: ${status || 'not provided'}`);
            
            // Check if final_report exists (may be at top level or in report object)
            const hasFinalReport = data.final_report || (data.report && data.report.final_report);
            
            if (status === "processing" && !hasFinalReport) {
              console.log(`Analysis still processing (progress: ${data.progress || 'N/A'})...`);
              // Wait 3 seconds before next poll
              await new Promise(resolve => setTimeout(resolve, pollInterval));
              continue; // Continue polling
            } else if (status === "failed") {
              console.error("Analysis failed");
              throw new Error("Analysis failed on the server");
            } else if (hasFinalReport || status === "completed") {
              console.log("Analysis completed! Extracting bacteria type...");
              
              // Check for final_report in multiple possible locations
              let finalReportStr = null;
              if (data.final_report) {
                // Top-level final_report (as shown in your example)
                finalReportStr = data.final_report;
                console.log("Found final_report at top level");
              } else if (data.report && data.report.final_report) {
                // Nested in report object
                finalReportStr = data.report.final_report;
                console.log("Found final_report in data.report");
              } else {
                console.error("Status indicates completion but no final_report found");
                console.log("Available keys in data:", Object.keys(data));
                throw new Error("Analysis completed but final_report data is missing");
              }
              
              // Parse final_report (it's a JSON string)
              let finalReport;
              if (typeof finalReportStr === 'string') {
                console.log("Parsing final_report JSON string...");
                finalReport = JSON.parse(finalReportStr);
              } else {
                console.log("final_report is already an object");
                finalReport = finalReportStr;
              }
              
              console.log("Parsed final report:", JSON.stringify(finalReport, null, 2));
              
              // Extract ONLY predictions[].organism
              const predictions = finalReport.predictions || [];
              console.log("Predictions array:", predictions);
              
              const organisms = predictions
                .map(p => p.organism)
                .filter(Boolean); // Remove any null/undefined values
              
              console.log("Extracted organisms:", organisms);
              
              let bacteriaType = "No bacteria detected";
              if (organisms.length > 0) {
                // Join multiple bacteria with commas
                bacteriaType = organisms.join(", ");
                console.log("Final bacteria type string:", bacteriaType);
              } else {
                console.log("No organisms found in predictions");
              }
              
              // Store result with bacteria type
              setResult({ ...data, bacteriaType });
              
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
              Alert.alert("Analysis Complete", `Bacteria Type: ${bacteriaType}`);
              return; // Exit polling loop
              
            } else {
              // Unknown status or no final_report yet - continue polling
              console.log(`Unknown status or no final_report yet (status: ${status}), continuing to poll...`);
              await new Promise(resolve => setTimeout(resolve, pollInterval));
              continue;
            }
          } catch (error) {
            // Handle timeout and network errors gracefully
            const isAbortError = error.name === 'AbortError' || error.name === 'Aborted';
            const isTimeout = isAbortError || 
                            error.message.includes("timeout") || 
                            error.message.includes("Timeout") ||
                            error.message.includes("aborted");
            const isNetworkError = error.message.includes("fetch") || 
                                 error.message.includes("network") ||
                                 error.message.includes("Network") ||
                                 error.message.includes("Failed to fetch") ||
                                 error.message.includes("Network request failed");
            
            if (isTimeout || isNetworkError) {
              console.log(`${isTimeout ? 'Timeout' : 'Network'} error on attempt ${attempts}, retrying...`);
              // Only retry if we haven't exhausted all attempts
              if (attempts < maxRetries) {
                await new Promise(resolve => setTimeout(resolve, pollInterval));
                continue; // Continue polling
              } else {
                throw new Error(`Request ${isTimeout ? 'timeout' : 'failed'} after ${maxRetries} attempts. Please check your connection and try again.`);
              }
            }
            // For other errors, throw immediately
            throw error;
          }
        }
        
        // If we've exhausted all retries
        throw new Error("Analysis timeout: Results not available after 20 attempts");
      };
      
      await pollResults();
      
    } catch (error) {
      console.error("Fetch result error:", error);
      Alert.alert("Error", error.message || "Failed to fetch results. Please try again.");
      setResult(null);
    } finally {
      setIsFetchingResult(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;

    // Use the bacteriaType that was extracted and stored during polling
    const bacteriaType = result.bacteriaType || "No bacteria detected";

  return (
      <Card style={styles.resultCard}>
        <Card.Content>
          <View style={styles.resultHeader}>
            <Ionicons name="flask" size={24} color="#059669" />
            <Title style={styles.resultTitle}>Analysis Results</Title>
          </View>
          <Divider style={styles.divider} />
          <View style={styles.resultContent}>
            <View style={styles.resultRow}>
              <Text style={styles.resultLabel}>Bacteria Type:</Text>
              <Text style={styles.resultValue}>{bacteriaType}</Text>
            </View>
          </View>
        </Card.Content>
      </Card>
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Ionicons name="medical" size={32} color="#1e40af" />
        <Title style={styles.headerTitle}>Wound Analysis System</Title>
        <Paragraph style={styles.headerSubtitle}>Professional Medical Analysis Platform</Paragraph>
      </View>

      {/* Step 1: Patient Registration */}
      <Card style={styles.card}>
                <Card.Content>
          <View style={styles.sectionHeader}>
            <View style={styles.stepIndicator}>
              <Text style={styles.stepNumber}>1</Text>
            </View>
            <Title style={styles.sectionTitle}>Patient Registration</Title>
            {patientData && (
              <Button
                mode="text"
                onPress={() => setShowPatientForm(!showPatientForm)}
                icon={showPatientForm ? "chevron-up" : "chevron-down"}
                compact
              >
                {showPatientForm ? "Hide" : "Show"}
              </Button>
            )}
          </View>

          {patientData && (
            <View style={styles.statusBadge}>
              <Ionicons name="checkmark-circle" size={20} color="#059669" />
              <Text style={styles.statusText}>
                Registered: {patientData.name} (MRN: {patientData.mrn || "N/A"})
                    </Text>
            </View>
          )}

          {showPatientForm && !patientData && (
            <View style={styles.form}>
              <TextInput
                label="Patient Name *"
                value={patient.name}
                onChangeText={(text) => setPatient({ ...patient, name: text })}
                mode="outlined"
                style={styles.input}
                left={<TextInput.Icon icon="account" />}
              />
              <TextInput
                label="Date of Birth (YYYY-MM-DD) *"
                value={patient.dob}
                onChangeText={(text) => setPatient({ ...patient, dob: text })}
                mode="outlined"
                style={styles.input}
                placeholder="2024-01-15"
                left={<TextInput.Icon icon="calendar" />}
              />
              <TextInput
                label="Gender *"
                value={patient.gender}
                onChangeText={(text) => setPatient({ ...patient, gender: text })}
                mode="outlined"
                style={styles.input}
                placeholder="Male / Female / Other"
                left={<TextInput.Icon icon="gender-male-female" />}
              />
              <Button
                mode="contained"
                onPress={createPatient}
                style={styles.primaryButton}
                loading={isCreatingPatient || isCreatingCase}
                disabled={isCreatingPatient || isCreatingCase}
                icon="account-plus"
              >
                {isCreatingPatient || isCreatingCase ? "Creating..." : "Register Patient"}
              </Button>
            </View>
          )}
                </Card.Content>
              </Card>

      {/* Step 2: Image Upload */}
      {caseData && (
        <Card style={styles.card}>
                <Card.Content>
            <View style={styles.sectionHeader}>
              <View style={styles.stepIndicator}>
                <Text style={styles.stepNumber}>2</Text>
              </View>
              <Title style={styles.sectionTitle}>Wound Images</Title>
            </View>
            <Paragraph style={styles.sectionDescription}>
              Upload exactly 3 clear wound images for analysis
            </Paragraph>

            <View style={styles.imagesContainer}>
              {images.map((img, index) => (
                <View key={index} style={styles.imageWrapper}>
                  <Image source={{ uri: img.uri }} style={styles.imagePreview} />
                  <TouchableOpacity
                    style={styles.removeButton}
                    onPress={() => removeImage(index)}
                  >
                    <Ionicons name="close-circle" size={24} color="#dc2626" />
                  </TouchableOpacity>
                  <View style={styles.imageLabel}>
                    <Text style={styles.imageLabelText}>Image {index + 1}</Text>
                  </View>
                </View>
              ))}
              {images.length < 3 && (
                <TouchableOpacity
                  style={styles.addImageButton}
                  onPress={() => handleAddImage("gallery")}
                >
                  <Ionicons name="add-circle-outline" size={40} color="#6b7280" />
                  <Text style={styles.addImageText}>Add Image</Text>
                  </TouchableOpacity>
              )}
            </View>

            {images.length < 3 && (
              <View style={styles.imageSourceButtons}>
                <Button
                  mode="outlined"
                  onPress={() => handleAddImage("camera")}
                  icon="camera"
                  style={styles.sourceButton}
                >
                  Camera
                </Button>
                <Button
                  mode="outlined"
                  onPress={() => handleAddImage("gallery")}
                  icon="image"
                  style={styles.sourceButton}
                >
                  Gallery
                </Button>
                <Button
                  mode="outlined"
                  onPress={() => handleAddImage("files")}
                  icon="folder"
                  style={styles.sourceButton}
                >
                  Files
                </Button>
              </View>
            )}

            {images.length === 3 && (
              <>
                <Button
                  mode="contained"
                  onPress={uploadImages}
                  style={styles.primaryButton}
                  loading={isUploadingImages}
                  disabled={isUploadingImages}
                  icon="cloud-upload"
                >
                  {isUploadingImages ? "Uploading..." : "Submit Images"}
                </Button>
                {isUploadingImages && (
                  <View style={styles.progressContainer}>
                    <Text style={styles.progressText}>
                      Uploading: {Math.round(uploadProgress)}%
                    </Text>
                  </View>
                )}
              </>
            )}
                </Card.Content>
              </Card>
      )}

      {/* Step 3: Clinical Context */}
      {caseData && images.length === 3 && (
        <Card style={styles.card}>
            <Card.Content>
            <View style={styles.sectionHeader}>
              <View style={styles.stepIndicator}>
                <Text style={styles.stepNumber}>3</Text>
                </View>
              <Title style={styles.sectionTitle}>Clinical Context</Title>
              {isContextSubmitted && (
                <Button
                  mode="text"
                  onPress={() => setShowContextForm(!showContextForm)}
                  icon={showContextForm ? "chevron-up" : "chevron-down"}
                  compact
                >
                  {showContextForm ? "Hide" : "Show"}
                </Button>
              )}
              </View>

            {isContextSubmitted && (
              <View style={styles.statusBadge}>
                <Ionicons name="checkmark-circle" size={20} color="#059669" />
                <Text style={styles.statusText}>Context Submitted</Text>
                </View>
              )}

            {showContextForm && (
              <View style={styles.form}>
                <View style={styles.switchRow}>
                  <Text style={styles.switchLabel}>Diabetic Patient</Text>
                  <Switch
                    value={context.isDiabetic}
                    onValueChange={(value) => setContext({ ...context, isDiabetic: value })}
                    disabled={isContextSubmitted}
                  />
                </View>
                <View style={styles.switchRow}>
                  <Text style={styles.switchLabel}>Recent Antibiotics</Text>
                  <Switch
                    value={context.recentAntibiotics}
                    onValueChange={(value) => setContext({ ...context, recentAntibiotics: value })}
                    disabled={isContextSubmitted}
                  />
                </View>
                <TextInput
                  label="Wound Type"
                  value={context.woundType}
                  onChangeText={(text) => setContext({ ...context, woundType: text })}
                  mode="outlined"
                  style={styles.input}
                  editable={!isContextSubmitted}
                  left={<TextInput.Icon icon="bandage" />}
                />
                <TextInput
                  label="Duration (days)"
                  value={context.durationDays}
                  onChangeText={(text) => setContext({ ...context, durationDays: text })}
                  mode="outlined"
                  keyboardType="numeric"
                  style={styles.input}
                  editable={!isContextSubmitted}
                  left={<TextInput.Icon icon="clock-outline" />}
                />
                <TextInput
                  label="Clinical Notes"
                  value={context.notes}
                  onChangeText={(text) => setContext({ ...context, notes: text })}
                  mode="outlined"
                  multiline
                  numberOfLines={4}
                  style={styles.input}
                  editable={!isContextSubmitted}
                  left={<TextInput.Icon icon="note-text" />}
                />
                {!isContextSubmitted && (
                <Button
                    mode="contained"
                    onPress={submitContext}
                    style={styles.primaryButton}
                    loading={isSubmittingContext}
                    disabled={isSubmittingContext}
                    icon="file-document-check"
                  >
                    Submit Context
                </Button>
                )}
              </View>
            )}
          </Card.Content>
        </Card>
      )}

      {/* Step 4: Get Results */}
      {caseData && isContextSubmitted && (
        <Card style={styles.card}>
          <Card.Content>
            <View style={styles.sectionHeader}>
              <View style={styles.stepIndicator}>
                <Text style={styles.stepNumber}>4</Text>
              </View>
              <Title style={styles.sectionTitle}>Analysis Results</Title>
            </View>
            <Paragraph style={styles.sectionDescription}>
              Retrieve the wound analysis results from the server
            </Paragraph>
                
                <Button
                  mode="contained"
              onPress={fetchResult}
              style={[styles.primaryButton, styles.resultButton]}
              loading={isFetchingResult}
              disabled={isFetchingResult}
              icon="file-find"
            >
              {isFetchingResult ? "Fetching Results..." : "Get Results"}
                </Button>

            {isFetchingResult && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#1e40af" />
                <Text style={styles.loadingText}>Polling for analysis results...</Text>
              </View>
            )}

            {renderResult()}
            </Card.Content>
          </Card>
      )}

      {/* Case Info Footer */}
      {caseData && (
        <Card style={styles.footerCard}>
        <Card.Content>
            <View style={styles.footerRow}>
              <Ionicons name="information-circle" size={20} color="#6b7280" />
              <Text style={styles.footerText}>
                Case ID: {caseData.id} | Status: {caseData.status || "Active"}
              </Text>
          </View>
        </Card.Content>
      </Card>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f0f4f8",
  },
  header: {
    alignItems: "center",
    paddingVertical: 24,
    backgroundColor: "#ffffff",
    borderBottomWidth: 1,
    borderBottomColor: "#e5e7eb",
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#1e40af",
    marginTop: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: "#6b7280",
    marginTop: 4,
  },
  card: {
    marginHorizontal: 16,
    marginTop: 16,
    elevation: 2,
    borderRadius: 12,
    backgroundColor: "#ffffff",
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  stepIndicator: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#1e40af",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  stepNumber: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    color: "#1f2937",
    flex: 1,
  },
  sectionDescription: {
    fontSize: 14,
    color: "#6b7280",
    marginBottom: 16,
    marginLeft: 44,
  },
  statusBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#d1fae5",
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  statusText: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: "600",
    color: "#059669",
  },
  form: {
    marginTop: 8,
  },
  input: {
    marginBottom: 12,
    backgroundColor: "#ffffff",
  },
  primaryButton: {
    marginTop: 8,
    backgroundColor: "#1e40af",
    paddingVertical: 4,
  },
  resultButton: {
    backgroundColor: "#059669",
  },
  imagesContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginVertical: 16,
  },
  imageWrapper: {
    width: (width - 64) / 3,
    marginBottom: 12,
    position: "relative",
  },
  imagePreview: {
    width: "100%",
    height: 120,
    borderRadius: 8,
    backgroundColor: "#f3f4f6",
  },
  removeButton: {
    position: "absolute",
    top: -8,
    right: -8,
    backgroundColor: "#ffffff",
    borderRadius: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  imageLabel: {
    position: "absolute",
    bottom: 4,
    left: 4,
    right: 4,
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 4,
  },
  imageLabelText: {
    color: "#ffffff",
    fontSize: 12,
    fontWeight: "600",
    textAlign: "center",
  },
  addImageButton: {
    width: (width - 64) / 3,
    height: 120,
    borderWidth: 2,
    borderColor: "#d1d5db",
    borderStyle: "dashed",
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f9fafb",
  },
  addImageText: {
    marginTop: 8,
    fontSize: 12,
    color: "#6b7280",
  },
  imageSourceButtons: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
  },
  sourceButton: {
    flex: 1,
    marginHorizontal: 4,
  },
  progressContainer: {
    marginTop: 12,
    alignItems: "center",
  },
  progressText: {
    fontSize: 14,
    color: "#6b7280",
  },
  switchRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
    paddingVertical: 8,
  },
  switchLabel: {
    fontSize: 16,
    color: "#374151",
    fontWeight: "500",
  },
  loadingContainer: {
    alignItems: "center",
    marginTop: 24,
    paddingVertical: 16,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: "#6b7280",
  },
  resultCard: {
    marginTop: 16,
    backgroundColor: "#f0fdf4",
    borderWidth: 2,
    borderColor: "#059669",
  },
  resultHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  resultTitle: {
    marginLeft: 8,
    fontSize: 20,
    fontWeight: "700",
    color: "#059669",
  },
  divider: {
    marginVertical: 12,
    backgroundColor: "#059669",
    height: 2,
  },
  resultContent: {
    marginTop: 8,
  },
  resultRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#d1fae5",
  },
  resultLabel: {
    fontSize: 16,
    fontWeight: "600",
    color: "#374151",
    flex: 1,
  },
  resultValue: {
    fontSize: 16,
    color: "#059669",
    fontWeight: "700",
    flex: 1,
    textAlign: "right",
  },
  footerCard: {
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 24,
    backgroundColor: "#f9fafb",
    elevation: 1,
  },
  footerRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  footerText: {
    marginLeft: 8,
    fontSize: 12,
    color: "#6b7280",
  },
});

