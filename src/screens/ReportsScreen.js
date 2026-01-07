import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import { Card, Title, Paragraph, Divider } from "react-native-paper";
import { Ionicons } from "@expo/vector-icons";
import { fetchWithTimeout } from "../services/apiService";

const API_BASE = "https://woundanalysis.onrender.com";

/* -------------------- HELPERS -------------------- */

const parseBacteriaReport = (data) => {
  try {
    if (!data.report || !data.report.final_report) {
      return { predictions: [], recommendation: "" };
    }

    const finalReportStr = data.report.final_report;
    const final = JSON.parse(finalReportStr);

    return {
      predictions: (final.predictions || []).map((p) => ({
        organism: p.organism || "Unknown",
        confidence: p.confidence ? Math.round(p.confidence * 100) : 0,
        rationale: p.rationale ? (Array.isArray(p.rationale) ? p.rationale.join(" ") : p.rationale) : "",
      })),
      recommendation: final.recommendation || "",
    };
  } catch (e) {
    console.error("Error parsing bacteria report:", e);
    return { predictions: [], recommendation: "" };
  }
};

/* -------------------- SCREEN -------------------- */

export default function ReportsScreen({ navigation }) {
  const [patients, setPatients] = useState([]);
  const [loadingCases, setLoadingCases] = useState(true);
  const [expandedPatients, setExpandedPatients] = useState(new Set());
  const [selectedCase, setSelectedCase] = useState(null);
  const [report, setReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);

  useEffect(() => {
    loadRecentCases();
  }, []);

  /* -------- Load Recent Cases and Group by Patient -------- */
  const loadRecentCases = async () => {
    setLoadingCases(true);
    try {
      const res = await fetchWithTimeout(
        `${API_BASE}/cases/recent`,
        { method: "GET" },
        60000
      );

      if (!res.ok) {
        throw new Error(`Failed to fetch cases: ${res.status}`);
      }

      const cases = await res.json();
      console.log("Recent cases received:", cases);

      // Group cases by patientId
      const patientMap = {};
      cases.forEach((caseItem) => {
        const patientId = caseItem.patientId;
        if (!patientMap[patientId]) {
          patientMap[patientId] = {
            patientId: patientId,
            patientName: caseItem.patientName || "Unknown Patient",
            cases: [],
          };
        }
        patientMap[patientId].cases.push({
          id: caseItem.id,
          createdAt: caseItem.createdAt,
          status: caseItem.status,
        });
      });

      // Convert to array and sort by most recent case
      const patientsArray = Object.values(patientMap).map((patient) => ({
        ...patient,
        cases: patient.cases.sort((a, b) => 
          new Date(b.createdAt) - new Date(a.createdAt)
        ),
      }));

      setPatients(patientsArray);
      console.log("Grouped patients:", patientsArray);
    } catch (e) {
      console.error("Error loading cases:", e);
      Alert.alert("Error", "Failed to load recent cases. Please try again.");
    } finally {
      setLoadingCases(false);
    }
  };

  /* -------- Toggle Patient Expansion -------- */
  const togglePatient = (patientId) => {
    const newExpanded = new Set(expandedPatients);
    if (newExpanded.has(patientId)) {
      newExpanded.delete(patientId);
      setSelectedCase(null);
      setReport(null);
    } else {
      newExpanded.add(patientId);
    }
    setExpandedPatients(newExpanded);
  };

  /* -------- Load Case Result -------- */
  const loadCaseReport = async (caseItem, patientName) => {
    setSelectedCase({ ...caseItem, patientName });
    setReport(null);
    setLoadingReport(true);

    try {
      console.log("Fetching results for case ID:", caseItem.id);
      const res = await fetchWithTimeout(
        `${API_BASE}/cases/${caseItem.id}/results`,
        { method: "GET" },
        300000
      );

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Failed to fetch results: ${res.status} - ${text}`);
      }

      const data = await res.json();
      console.log("Results received:", JSON.stringify(data, null, 2));
      const parsedReport = parseBacteriaReport(data);
      console.log("Parsed report:", parsedReport);
      setReport(parsedReport);
    } catch (e) {
      console.error("Error loading case report:", e);
      Alert.alert("Error", `Failed to load case results: ${e.message}`);
    } finally {
      setLoadingReport(false);
    }
  };

  /* -------------------- UI -------------------- */

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="document-text" size={32} color="#1e40af" />
        <Title style={styles.headerTitle}>Patient Reports</Title>
        <Paragraph style={styles.headerSubtitle}>
          View wound analysis reports grouped by patient
          </Paragraph>
          </View>

      {/* -------- Loading State -------- */}
      {loadingCases ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#1e40af" />
          <Text style={styles.loadingText}>Loading recent cases...</Text>
        </View>
      ) : patients.length === 0 ? (
        <Card style={styles.emptyCard}>
        <Card.Content>
            <Ionicons name="document-outline" size={48} color="#9ca3af" />
            <Text style={styles.emptyText}>No recent cases found</Text>
        </Card.Content>
      </Card>
      ) : (
        /* -------- Patient Cards -------- */
        <>
          {patients.map((patient) => {
            const isExpanded = expandedPatients.has(patient.patientId);
            return (
              <Card key={patient.patientId} style={styles.patientCard}>
        <Card.Content>
                {/* Patient Header */}
                <TouchableOpacity
                  onPress={() => {
                    // Navigate to PatientDetailScreen
                    navigation.navigate("PatientDetail", {
                      patientId: patient.patientId,
                      patientName: patient.patientName,
                    });
                  }}
                  style={styles.patientHeader}
                >
                  <View style={styles.patientHeaderLeft}>
                    <Ionicons
                      name={isExpanded ? "chevron-down" : "chevron-forward"}
                      size={24}
                      color="#1e40af"
                    />
                    <View style={styles.patientInfo}>
                      <Title style={styles.patientName}>
                        {patient.patientName}
                      </Title>
                      <Text style={styles.patientId}>
                        Patient ID: {patient.patientId}
              </Text>
            </View>
            </View>
                  <View style={styles.caseCountBadge}>
                    <Text style={styles.caseCountText}>
                      {patient.cases.length} {patient.cases.length === 1 ? "Case" : "Cases"}
                    </Text>
            </View>
                </TouchableOpacity>

                {/* Expanded Cases List */}
                {isExpanded && (
                  <View style={styles.casesContainer}>
                    <Divider style={styles.divider} />
                    {patient.cases.map((caseItem) => {
                      const isSelected = selectedCase?.id === caseItem.id;
                      return (
                        <TouchableOpacity
                          key={caseItem.id}
                          onPress={() => loadCaseReport(caseItem, patient.patientName)}
                          style={[
                            styles.caseItem,
                            isSelected && styles.caseItemSelected,
                          ]}
                        >
                          <View style={styles.caseItemLeft}>
                            <Ionicons
                              name="medical"
                              size={20}
                              color={isSelected ? "#059669" : "#6b7280"}
                            />
                            <View style={styles.caseInfo}>
                              <Text style={styles.caseId}>
                                Case: {caseItem.id.substring(0, 8)}...
                              </Text>
                              <Text style={styles.caseDate}>
                                {new Date(caseItem.createdAt).toLocaleDateString()}
                              </Text>
          </View>
                          </View>
                          <View
                            style={[
                              styles.statusBadge,
                              caseItem.status === "completed" && styles.statusCompleted,
                            ]}
                          >
                            <Text style={styles.statusText}>{caseItem.status}</Text>
                          </View>
                        </TouchableOpacity>
                      );
                    })}
          </View>
                )}

                {/* Case Report Display */}
                {isExpanded && selectedCase && selectedCase.patientId === patient.patientId && (
                  <View style={styles.reportContainer}>
                    <Divider style={styles.divider} />
                    {loadingReport ? (
                      <View style={styles.reportLoading}>
                        <ActivityIndicator size="small" color="#059669" />
                        <Text style={styles.reportLoadingText}>
                          Loading analysis results...
                        </Text>
                      </View>
                    ) : report ? (
                      <Card style={styles.reportCard}>
        <Card.Content>
                          <View style={styles.reportHeader}>
                            <Ionicons name="flask" size={24} color="#059669" />
                            <Title style={styles.reportTitle}>Analysis Results</Title>
          </View>
          
                          {/* Bacteria Detection */}
                          <View style={styles.section}>
                            <Title style={styles.sectionTitle}>Bacteria Detected</Title>
                            {report.predictions.length === 0 ? (
                              <View style={styles.noBacteriaContainer}>
                                <Ionicons name="checkmark-circle" size={32} color="#9ca3af" />
                                <Text style={styles.noBacteriaText}>
                                  No bacteria detected
                                </Text>
                </View>
                            ) : (
                              report.predictions.map((b, i) => (
                                <View key={`${b.organism}-${i}`} style={styles.bacteriaItem}>
                                  <View style={styles.bacteriaHeader}>
                                    <Ionicons name="bug" size={20} color="#dc2626" />
                                    <Text style={styles.bacteriaName}>{b.organism}</Text>
                </View>
                                  <View style={styles.confidenceBar}>
                                    <View
                                      style={[
                                        styles.confidenceFill,
                                        { width: `${b.confidence}%` },
                                      ]}
                                    />
                </View>
                                  <Text style={styles.confidenceText}>
                                    Confidence: {b.confidence}%
                  </Text>
                                  {b.rationale && (
                                    <Text style={styles.rationaleText}>{b.rationale}</Text>
                                  )}
                </View>
                              ))
                            )}
              </View>
              
                          {/* Clinical Recommendation */}
                          {report.recommendation && (
                            <View style={styles.section}>
                              <Title style={styles.sectionTitle}>
                                Clinical Recommendation
                              </Title>
                              <Text style={styles.recommendationText}>
                                {report.recommendation}
                              </Text>
            </View>
          )}
        </Card.Content>
      </Card>
                    ) : null}
                  </View>
                )}
        </Card.Content>
      </Card>
            );
          })}
        </>
      )}
    </ScrollView>
  );
}

/* -------------------- STYLES -------------------- */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f0f4f8",
  },
  header: {
    alignItems: "center",
    paddingVertical: 24,
    paddingHorizontal: 16,
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
    textAlign: "center",
  },
  loadingContainer: {
    alignItems: "center",
    paddingVertical: 40,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: "#6b7280",
  },
  emptyCard: {
    margin: 16,
    alignItems: "center",
    paddingVertical: 40,
  },
  emptyText: {
    marginTop: 12,
    fontSize: 16,
    color: "#9ca3af",
  },
  patientCard: {
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    elevation: 2,
    borderRadius: 12,
    backgroundColor: "#ffffff",
  },
  patientHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 8,
  },
  patientHeaderLeft: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  patientInfo: {
    marginLeft: 12,
    flex: 1,
  },
  patientName: {
    fontSize: 18,
    fontWeight: "600",
    color: "#1f2937",
  },
  patientId: {
    fontSize: 12,
    color: "#6b7280",
    marginTop: 2,
  },
  caseCountBadge: {
    backgroundColor: "#dbeafe",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  caseCountText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#1e40af",
  },
  casesContainer: {
    marginTop: 8,
  },
  divider: {
    marginVertical: 12,
    backgroundColor: "#e5e7eb",
  },
  caseItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    marginBottom: 8,
    backgroundColor: "#f9fafb",
  },
  caseItemSelected: {
    backgroundColor: "#f0fdf4",
    borderWidth: 2,
    borderColor: "#059669",
  },
  caseItemLeft: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  caseInfo: {
    marginLeft: 12,
  },
  caseId: {
    fontSize: 14,
    fontWeight: "600",
    color: "#374151",
  },
  caseDate: {
    fontSize: 12,
    color: "#6b7280",
    marginTop: 2,
  },
  statusBadge: {
    backgroundColor: "#f3f4f6",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusCompleted: {
    backgroundColor: "#d1fae5",
  },
  statusText: {
    fontSize: 11,
    fontWeight: "600",
    color: "#374151",
    textTransform: "capitalize",
  },
  reportContainer: {
    marginTop: 8,
  },
  reportLoading: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 20,
  },
  reportLoadingText: {
    marginLeft: 8,
    fontSize: 14,
    color: "#6b7280",
  },
  reportCard: {
    marginTop: 8,
    backgroundColor: "#f0fdf4",
    borderWidth: 2,
    borderColor: "#059669",
  },
  reportHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  reportTitle: {
    marginLeft: 8,
    fontSize: 20,
    fontWeight: "700",
    color: "#059669",
  },
  section: {
    marginTop: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#374151",
    marginBottom: 12,
  },
  noBacteriaContainer: {
    alignItems: "center",
    paddingVertical: 20,
  },
  noBacteriaText: {
    marginTop: 8,
    fontSize: 14,
    color: "#6b7280",
    fontWeight: "500",
  },
  bacteriaItem: {
    backgroundColor: "#ffffff",
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: "#dc2626",
  },
  bacteriaHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  bacteriaName: {
    marginLeft: 8,
    fontSize: 16,
    fontWeight: "700",
    color: "#dc2626",
  },
  confidenceBar: {
    height: 8,
    backgroundColor: "#e5e7eb",
    borderRadius: 4,
    marginBottom: 8,
    overflow: "hidden",
  },
  confidenceFill: {
    height: "100%",
    backgroundColor: "#059669",
    borderRadius: 4,
  },
  confidenceText: {
    fontSize: 12,
    color: "#6b7280",
    marginBottom: 4,
  },
  rationaleText: {
    fontSize: 12,
    color: "#6b7280",
    fontStyle: "italic",
    marginTop: 4,
  },
  recommendationText: {
    fontSize: 14,
    color: "#374151",
    lineHeight: 20,
  },
});
