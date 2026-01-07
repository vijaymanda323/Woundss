import { Platform } from "react-native";

// Hosted API base URL
const API_BASE_URL = "https://woundanalysis.onrender.com";

// Helper: fetch with timeout (kept for existing callers)
export const fetchWithTimeout = (url, options = {}, timeout = 10000) => {
  return Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timeout")), timeout)
    ),
  ]);
};

const apiFetch = async (path, options = {}) => {
  const url = path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
  const { timeout = 300000, headers = {}, ...rest } = options; // 5 minutes default for Render free tier

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...rest,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...headers,
      },
    });
    return response;
  } finally {
    clearTimeout(id);
  }
};

const getJson = async (path, options = {}) => {
  const res = await apiFetch(path, options);
  const data = await res.json();
  return { data, status: res.status };
};

const buildImagePart = (image) => {
  if (!image) {
    throw new Error("No image supplied for upload");
  }

  if (Platform.OS === "web" && image.file) {
    return image.file;
  }

  return {
    uri: image.uri || image.path || image.localUri,
    name: image.name || "wound_image.jpg",
    type: image.type || "image/jpeg",
  };
};

const normalizeWoundResult = (apiResult, patient, caseData) => {
  const woundType =
    apiResult?.wound_type ||
    apiResult?.analysis?.wound_type ||
    apiResult?.report?.wound_type ||
    apiResult?.wound_classification?.wound_type ||
    "unknown";

  const estimatedHealing =
    apiResult?.estimated_days_to_cure ||
    apiResult?.estimated_healing_time ||
    apiResult?.report?.estimated_days_to_cure ||
    21;

  const confidence =
    apiResult?.model_confidence ||
    apiResult?.confidence ||
    apiResult?.report?.confidence ||
    0.85;

  const bacteriaType =
    apiResult?.bacteria_type ||
    apiResult?.bacteria ||
    apiResult?.report?.bacteria_type ||
    apiResult?.analysis?.bacteria_type ||
    null;
      
      return {
    status: "success",
    case_id: caseData?.id,
    patient_id: patient?.id,
        wound_classification: {
          wound_type: woundType,
      estimated_days_to_cure: estimatedHealing,
      healing_time_category: "moderate_healing",
          model_available: true,
        },
    area_cm2:
      apiResult?.area_cm2 ||
      apiResult?.report?.area_cm2 ||
      apiResult?.analysis?.area_cm2 ||
      null,
    area_pixels:
      apiResult?.area_pixels ||
      apiResult?.report?.area_pixels ||
      apiResult?.analysis?.area_pixels ||
      null,
    perimeter:
      apiResult?.perimeter ||
      apiResult?.report?.perimeter ||
      apiResult?.analysis?.perimeter ||
      null,
        model_confidence: confidence,
    bacteria_type: bacteriaType,
        enhanced_analysis: {
          wound_type: woundType,
      estimated_healing_time: estimatedHealing,
      age_group: apiResult?.age_group || "adult",
      size_category: apiResult?.size_category || "medium",
      precautions: apiResult?.precautions || [],
      treatment_recommendations: apiResult?.treatment_recommendations || [],
      follow_up_schedule: apiResult?.follow_up_schedule || [],
      risk_factors: apiResult?.risk_factors || [],
      healing_stages: apiResult?.healing_stages || {},
      bacteria_type: bacteriaType,
        },
        patient_info: {
      ...patient,
          analysis_date: new Date().toISOString(),
        },
    raw_result: apiResult,
      };
};

export const createPatient = async (patientInfo = {}) => {
  const payload = {
    name: patientInfo.name || "Unnamed Patient",
    mrn: patientInfo.id || patientInfo.mrn || undefined,
    dob: patientInfo.dob || undefined,
    gender: patientInfo.gender || undefined,
  };
    
  const { data } = await getJson("/patients", {
    method: "POST",
      headers: {
      "Content-Type": "application/json",
      },
    body: JSON.stringify(payload),
  });

  return data;
};

export const createCase = async (patientId) => {
  if (!patientId) {
    throw new Error("patientId is required to create a case");
  }

  const { data } = await getJson("/cases", {
    method: "POST",
      headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ patientId }),
    });

  return data;
};

export const uploadCaseImages = async (caseId, images = []) => {
  if (!caseId) throw new Error("caseId is required for image upload");
  if (!images.length) throw new Error("At least one image is required");

  const formData = new FormData();
  images.forEach((img) => formData.append("images", buildImagePart(img)));

  const res = await apiFetch(`/cases/${caseId}/images`, {
    method: "POST",
    body: formData,
    timeout: 300000, // 5 minutes for Render free tier cold starts + large image uploads
  });

  if (!res.ok) {
    throw new Error(`Image upload failed with status ${res.status}`);
    }

  const data = await res.json();
  return data;
};

export const submitCaseContext = async (caseId, patientInfo = {}, context = {}) => {
  if (!caseId) throw new Error("caseId is required for context submission");

  const payload = {
    isDiabetic: patientInfo.injuryType?.toLowerCase()?.includes("diabetic") || undefined,
    woundType: patientInfo.injuryType || undefined,
    durationDays: context.durationDays || undefined,
    recentAntibiotics: context.recentAntibiotics,
    patientAge: patientInfo.age ? Number(patientInfo.age) : undefined,
    notes: patientInfo.notes || context.notes || undefined,
  };

  const { data } = await getJson(`/cases/${caseId}/context`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return data;
};

export const fetchCaseResults = async (caseId) => {
  const res = await apiFetch(`/cases/${caseId}/results`, { 
    method: "GET",
    timeout: 300000, // 5 minutes for Render free tier cold starts
  });
  const data = await res.json();
  return { status: res.status, data };
};

export const pollCaseResults = async (
  caseId,
  { maxAttempts = 60, intervalMs = 5000 } = {}, // 60 attempts × 5s = 5 minutes max for Render free tier
) => {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const { status, data } = await fetchCaseResults(caseId);

    if (status === 200) {
      return data;
    }

    if (status === 202) {
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
      continue;
    }

    throw new Error(`Unexpected status while polling results: ${status}`);
  }

  throw new Error("Timed out waiting for wound analysis results");
};

export const analyzeWound = async (images, patientInfo = {}, context = {}) => {
  const imageArray = Array.isArray(images) ? images : [images];
  const patient = await createPatient(patientInfo);
  const caseData = await createCase(patient.id);
  await uploadCaseImages(caseData.id, imageArray);
  await submitCaseContext(caseData.id, patientInfo, context);
  const result = await pollCaseResults(caseData.id);

  return normalizeWoundResult(result, patient, caseData);
};

export const getPatientHistory = async () => {
  const { data } = await getJson("/cases/recent", { method: "GET" });
  return (data || []).map((entry) => ({
    id: entry.id,
    patient_id: entry.patient_id || entry.patientId,
    filename: entry.image || entry.imagePaths?.[0] || "Unknown",
    timestamp: entry.created_at || entry.createdAt,
    wound_type: entry.wound_type || entry.status || "unknown",
    healing_time_category: entry.healing_time_category || "processing",
    estimated_days_to_cure: entry.estimated_days_to_cure || null,
    notes: entry.notes || "",
  }));
};

export const getPatientById = async (patientId) => {
  const { data } = await getJson(`/patients/${patientId}`, { method: "GET" });
  return data;
};

export const searchPatients = async (query) => {
  const { data } = await getJson(`/patients/search?q=${encodeURIComponent(query)}`, {
    method: "GET",
  });
  return data;
};

export const generateReport = async (analysisData, reportType = "patient") => {
  const woundType = analysisData?.wound_classification?.wound_type || "unknown";
  const healingDays =
    analysisData?.wound_classification?.estimated_days_to_cure || "N/A";
  const confidence = analysisData?.model_confidence
    ? `${(analysisData.model_confidence * 100).toFixed(1)}%`
    : "N/A";

    return {
      success: true,
    report_type: reportType,
    report_content: `Wound type: ${woundType}\nEstimated healing: ${healingDays} days\nConfidence: ${confidence}\n\nRecommendations:\n${
      (analysisData?.enhanced_analysis?.treatment_recommendations || [])
        .map((rec) => `- ${rec}`)
        .join("\n") || "Not provided"
    }`,
    };
};
