import axios from 'axios';

const buildApiUrl = (endpoint) => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
  return `${baseUrl}${endpoint}`;
};

const fetchRepositories = async (token) => {
  try {
    const response = await axios.get(buildApiUrl('/repositories'), {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching repositories:', error);
    throw error;
  }
};

export { buildApiUrl, fetchRepositories };
