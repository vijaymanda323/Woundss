import axios from 'axios';
import { API_BASE_URL } from '../utils/env';

/**
 * Axios instance configured with base URL and interceptors
 * 
 * To integrate with real external APIs later:
 * 1. Add authentication headers in interceptors
 * 2. Handle API key management
 * 3. Add request/response transformation
 * 4. Implement retry logic for failed requests
 */

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 10000, // 10 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging and auth
apiClient.interceptors.request.use(
  (config) => {
    // Log request in development
    if (__DEV__) {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    }
    
    // Add authentication token if available
    // const token = getAuthToken(); // Implement your auth token logic
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging and error handling
apiClient.interceptors.response.use(
  (response) => {
    // Log response in development
    if (__DEV__) {
      console.log(`API Response: ${response.status} ${response.config.url}`);
    }
    
    return response;
  },
  (error) => {
    // Log error details
    console.error('API Error:', {
      message: error.message,
      status: error.response?.status,
      url: error.config?.url,
      data: error.response?.data,
    });
    
    // Handle specific error cases
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      console.log('Unauthorized - redirect to login');
    } else if (error.response?.status >= 500) {
      // Handle server errors
      console.log('Server error - show error message');
    } else if (!error.response) {
      // Network error
      console.log('Network error - check connection');
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;






