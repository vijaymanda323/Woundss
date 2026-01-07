import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  RefreshControl,
} from "react-native";
import { Card, Title, Paragraph, Divider, Button } from "react-native-paper";
import { Ionicons } from "@expo/vector-icons";
import { fetchWithTimeout } from "../services/apiService";

const API_BASE_URL = "https://woundanalysis.onrender.com";

export default function PatientDetailScreen({ route, navigation }) {
  const { patientId, patientName } = route.params || {};
  
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [patientCase, setPatientCase] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(false);

  useEffect(() => {
    if (patientId) {
      loadPatientAnalysis();
    } else {
      setError("Patient ID is required");
      setLoading(false);
    }
  }, [patientId]);

  const loadPatientAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Step 1: Fetch recent cases
      console.log("Fetching recent cases...");
      const casesResponse = await fetchWithTimeout(
        `${API_BASE_URL}/cases/recent`,
        { method: "GET" },
        60000
      );

      if (!casesResponse.ok) {
        throw new Error(`Failed to fetch cases: ${casesResponse.status}`);
      }

      const allCases = await casesResponse.json();
      console.log("All cases received:", allCases.length);

      // Step 2: Filter cases by patientId
      const patientCases = allCases.filter(
        (caseItem) => caseItem.patientId === patientId
      );

      if (patientCases.length === 0) {
        setError("No cases found for this patient");
        setLoading(false);
        return;
      }

      console.log(`Found ${patientCases.length} case(s) for patient ${patientId}`);

      // Step 3: Select the most recent case by createdAt
      const sortedCases = patientCases.sort((a, b) => {
        const dateA = new Date(a.createdAt);
        const dateB = new Date(b.createdAt);
        return dateB - dateA; // Most recent first
      });

      const latestCase = sortedCases[0];
      console.log("Latest case:", latestCase);
      setPatientCase(latestCase);

      // Step 4: Fetch results for the latest case
      await fetchCaseResults(latestCase.id);

    } catch (err) {
      console.error("Error loading patient analysis:", err);
      setError(err.message || "Failed to load patient analysis");
    } finally {
      setLoading(false);
    }
  };

  const fetchCaseResults = async (caseId) => {
    if (!caseId) {
      setError("Case ID is required");
      return;
    }

    setIsPolling(true);
    setAnalysisData(null);

    try {
      const maxRetries = 20;
      const pollInterval = 3000; // 3 seconds
      let attempts = 0;

      while (attempts < maxRetries) {
        attempts++;
        console.log(`Polling results attempt ${attempts}/${maxRetries} for case ${caseId}`);

        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 30000);

          const response = await fetch(`${API_BASE_URL}/cases/${caseId}/results`, {
            method: "GET",
            signal: controller.signal,
            headers: {
              Accept: "application/json",
            },
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            if (attempts < maxRetries) {
              console.log(`Request failed, retrying... (${response.status})`);
              await new Promise((resolve) => setTimeout(resolve, pollInterval));
              continue;
            }
            throw new Error(`Failed to fetch results: ${response.status}`);
          }

          const data = await response.json();
          console.log(`Attempt ${attempts} - Response:`, JSON.stringify(data, null, 2));

          // Check if still processing
          if (data.status === "processing") {
            console.log("Analysis still processing...");
            setAnalysisData({ status: "processing", progress: data.progress });
            await new Promise((resolve) => setTimeout(resolve, pollInterval));
            continue;
          }

          // Check if final_report exists
          if (data.final_report || (data.report && data.report.final_report)) {
            console.log("final_report found, parsing...");
            const finalReportStr = data.final_report || data.report.final_report;
            const parsed = parseFinalReport(finalReportStr);
            setAnalysisData({ status: "completed", predictions: parsed });
            setIsPolling(false);
            return;
          }

          // If no final_report and no processing status, wait and retry
          if (attempts < maxRetries) {
            await new Promise((resolve) => setTimeout(resolve, pollInterval));
            continue;
          }

        } catch (fetchError) {
          const isAbortError = fetchError.name === "AbortError" || fetchError.name === "Aborted";
          const isTimeout = isAbortError || fetchError.message.includes("timeout");

          if (isTimeout && attempts < maxRetries) {
            console.log("Request timeout, retrying...");
            await new Promise((resolve) => setTimeout(resolve, pollInterval));
            continue;
          }

          if (attempts >= maxRetries) {
            throw new Error("Analysis timeout: Results not available after 20 attempts");
          }
        }
      }

      throw new Error("Analysis timeout: Results not available after 20 attempts");
    } catch (err) {
      console.error("Error fetching case results:", err);
      setError(err.message || "Failed to fetch analysis results");
      setIsPolling(false);
    }
  };

  const parseFinalReport = (finalReportStr) => {
    try {
      if (!finalReportStr) {
        return [];
      }

      let finalReport;
      if (typeof finalReportStr === "string") {
        finalReport = JSON.parse(finalReportStr);
      } else {
        finalReport = finalReportStr;
      }

      const predictions = finalReport.predictions || [];
      
      return predictions.map((pred) => ({
        organism: pred.organism || "Unknown",
        confidence: pred.confidence ? Math.round(pred.confidence * 100) : 0,
        class: pred.class || null,
      }));
    } catch (parseError) {
      console.error("Error parsing final_report:", parseError);
      return [];
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadPatientAnalysis();
    setRefreshing(false);
  };

  const renderAnalysisContent = () => {
    if (!analysisData) {
      return null;
    }

    if (analysisData.status === "processing") {
      return (
        <Card style={styles.statusCard}>
          <Card.Content style={styles.statusContent}>
            <ActivityIndicator size="large" color="#1e40af" />
            <Text style={styles.statusText}>Analysis in progress...</Text>
            {analysisData.progress && (
              <Text style={styles.progressText}>Progress: {analysisData.progress}%</Text>
            )}
          </Card.Content>
        </Card>
      );
    }

    if (analysisData.status === "completed") {
      const predictions = analysisData.predictions || [];

      if (predictions.length === 0) {
        return (
          <Card style={styles.emptyCard}>
            <Card.Content style={styles.emptyContent}>
              <Ionicons name="checkmark-circle" size={64} color="#9ca3af" />
              <Text style={styles.emptyText}>No bacteria detected</Text>
              <Text style={styles.emptySubtext}>
                The analysis did not identify any bacterial organisms in this sample.
              </Text>
            </Card.Content>
          </Card>
        );
      }

      return (
        <View style={styles.predictionsContainer}>
          <View style={styles.sectionHeader}>
            <Ionicons name="flask" size={28} color="#059669" />
            <Title style={styles.sectionTitle}>Bacterial Analysis Results</Title>
          </View>
          <Divider style={styles.divider} />

          {predictions.map((pred, index) => (
            <Card key={index} style={styles.organismCard}>
              <Card.Content>
                <View style={styles.organismHeader}>
                  <Ionicons name="bug" size={24} color="#dc2626" />
                  <View style={styles.organismInfo}>
                    <Text style={styles.organismName}>{pred.organism}</Text>
                    {pred.class && (
                      <Text style={styles.organismClass}>
                        {pred.class}
                      </Text>
                    )}
                  </View>
                </View>

                <View style={styles.confidenceContainer}>
                  <View style={styles.confidenceBar}>
                    <View
                      style={[
                        styles.confidenceFill,
                        { width: `${pred.confidence}%` },
                      ]}
                    />
                  </View>
                  <Text style={styles.confidenceText}>
                    {pred.confidence}% Confidence
                  </Text>
                </View>
              </Card.Content>
            </Card>
          ))}
        </View>
      );
    }

    return null;
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#1e40af" />
        <Text style={styles.loadingText}>Loading patient analysis...</Text>
      </View>
    );
  }

  if (error && !patientCase) {
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.errorContainer}>
        <Card style={styles.errorCard}>
          <Card.Content>
            <Ionicons name="alert-circle" size={48} color="#dc2626" />
            <Title style={styles.errorTitle}>Error</Title>
            <Paragraph style={styles.errorText}>{error}</Paragraph>
            <Button
              mode="contained"
              onPress={loadPatientAnalysis}
              style={styles.retryButton}
              icon="refresh"
            >
              Retry
            </Button>
          </Card.Content>
        </Card>
      </ScrollView>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Patient Header */}
      <View style={styles.header}>
        <Ionicons name="person-circle" size={48} color="#1e40af" />
        <Title style={styles.patientName}>
          {patientName || "Patient"}
        </Title>
        <Text style={styles.patientId}>Patient ID: {patientId}</Text>
        {patientCase && (
          <View style={styles.caseInfo}>
            <Text style={styles.caseText}>
              Case: {patientCase.id.substring(0, 8)}...
            </Text>
            <Text style={styles.caseDate}>
              {new Date(patientCase.createdAt).toLocaleDateString("en-US", {
                year: "numeric",
                month: "long",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </Text>
          </View>
        )}
      </View>

      {/* Error Banner */}
      {error && (
        <Card style={styles.errorBanner}>
          <Card.Content>
            <View style={styles.errorBannerContent}>
              <Ionicons name="warning" size={20} color="#dc2626" />
              <Text style={styles.errorBannerText}>{error}</Text>
            </View>
          </Card.Content>
        </Card>
      )}

      {/* Analysis Results */}
      {renderAnalysisContent()}

      {/* Polling Indicator */}
      {isPolling && analysisData?.status !== "processing" && (
        <Card style={styles.pollingCard}>
          <Card.Content style={styles.pollingContent}>
            <ActivityIndicator size="small" color="#1e40af" />
            <Text style={styles.pollingText}>Fetching analysis results...</Text>
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
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f0f4f8",
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: "#6b7280",
  },
  errorContainer: {
    flex: 1,
    justifyContent: "center",
    padding: 20,
  },
  errorCard: {
    backgroundColor: "#fef2f2",
    borderLeftWidth: 4,
    borderLeftColor: "#dc2626",
  },
  errorTitle: {
    color: "#dc2626",
    marginTop: 12,
    marginBottom: 8,
  },
  errorText: {
    color: "#991b1b",
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: "#dc2626",
  },
  header: {
    alignItems: "center",
    paddingVertical: 32,
    paddingHorizontal: 20,
    backgroundColor: "#ffffff",
    borderBottomWidth: 1,
    borderBottomColor: "#e5e7eb",
  },
  patientName: {
    fontSize: 28,
    fontWeight: "700",
    color: "#1e2937",
    marginTop: 12,
  },
  patientId: {
    fontSize: 14,
    color: "#6b7280",
    marginTop: 4,
    fontWeight: "500",
  },
  caseInfo: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: "#e5e7eb",
    width: "100%",
    alignItems: "center",
  },
  caseText: {
    fontSize: 13,
    color: "#374151",
    fontWeight: "600",
  },
  caseDate: {
    fontSize: 12,
    color: "#6b7280",
    marginTop: 4,
  },
  errorBanner: {
    margin: 16,
    marginBottom: 0,
    backgroundColor: "#fef2f2",
    borderLeftWidth: 4,
    borderLeftColor: "#dc2626",
  },
  errorBannerContent: {
    flexDirection: "row",
    alignItems: "center",
  },
  errorBannerText: {
    marginLeft: 8,
    color: "#991b1b",
    fontSize: 14,
  },
  statusCard: {
    margin: 16,
    backgroundColor: "#eff6ff",
    borderLeftWidth: 4,
    borderLeftColor: "#1e40af",
  },
  statusContent: {
    alignItems: "center",
    paddingVertical: 32,
  },
  statusText: {
    marginTop: 16,
    fontSize: 18,
    fontWeight: "600",
    color: "#1e40af",
  },
  progressText: {
    marginTop: 8,
    fontSize: 14,
    color: "#6b7280",
  },
  predictionsContainer: {
    padding: 16,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  sectionTitle: {
    marginLeft: 12,
    fontSize: 24,
    fontWeight: "700",
    color: "#059669",
  },
  divider: {
    marginBottom: 20,
    backgroundColor: "#059669",
    height: 2,
  },
  organismCard: {
    marginBottom: 16,
    backgroundColor: "#ffffff",
    elevation: 2,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: "#dc2626",
  },
  organismHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  organismInfo: {
    marginLeft: 12,
    flex: 1,
  },
  organismName: {
    fontSize: 20,
    fontWeight: "700",
    color: "#1f2937",
    marginBottom: 4,
  },
  organismClass: {
    fontSize: 14,
    color: "#6b7280",
    fontWeight: "500",
    textTransform: "capitalize",
  },
  confidenceContainer: {
    marginTop: 8,
  },
  confidenceBar: {
    height: 10,
    backgroundColor: "#e5e7eb",
    borderRadius: 5,
    marginBottom: 8,
    overflow: "hidden",
  },
  confidenceFill: {
    height: "100%",
    backgroundColor: "#059669",
    borderRadius: 5,
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#374151",
  },
  emptyCard: {
    margin: 16,
    backgroundColor: "#f9fafb",
    borderWidth: 2,
    borderColor: "#e5e7eb",
    borderStyle: "dashed",
  },
  emptyContent: {
    alignItems: "center",
    paddingVertical: 48,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 20,
    fontWeight: "600",
    color: "#374151",
  },
  emptySubtext: {
    marginTop: 8,
    fontSize: 14,
    color: "#6b7280",
    textAlign: "center",
    paddingHorizontal: 20,
  },
  pollingCard: {
    margin: 16,
    backgroundColor: "#f9fafb",
  },
  pollingContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
  },
  pollingText: {
    marginLeft: 12,
    fontSize: 14,
    color: "#6b7280",
  },
});

