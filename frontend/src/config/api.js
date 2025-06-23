// frontend/src/config/api.js
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// Helper function to build API URLs
export const buildApiUrl = (endpoint) => {
  return `${API_BASE_URL}${endpoint}`;
};

// Remove /api prefix from endpoint if present to avoid duplication
export const buildApiUrlSafe = (endpoint) => {
  const cleanEndpoint = endpoint.startsWith('/api') ? endpoint.substring(4) : endpoint;
  return `${API_BASE_URL}${cleanEndpoint}`;
};
