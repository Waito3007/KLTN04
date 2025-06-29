// frontend/src/services/api.js
import axios from 'axios';
import { message } from 'antd';

// Centralized API configuration - get from environment variable
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// Axios instance với common config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// Create a separate client for long-running operations like sync
const apiClientLongTimeout = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds for sync operations
});

// Create a separate API client without auth for public endpoints (currently unused)
// const publicApiClient = axios.create({
//   baseURL: API_BASE_URL,
//   timeout: 10000,
// });

// Request interceptor để tự động thêm token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    console.log(`🔑 Token exists: ${!!token}`);
    if (token) {
      config.headers.Authorization = `token ${token}`;
      console.log(`🔑 Authorization header set: token ${token.substring(0, 10)}...`);
    }
    return config;
  },
  (error) => {
    console.error('❌ Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor để xử lý lỗi chung
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} for ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error(`❌ API Error:`, {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message
    });
    
    if (error.response?.status === 401) {
      message.error('Phiên đăng nhập đã hết hạn');
      // Có thể redirect đến login page
    } else if (error.response?.status === 429) {
      message.warning('Quá nhiều requests, vui lòng thử lại sau');
    }
    return Promise.reject(error);
  }
);

// Add interceptors for long timeout client
apiClientLongTimeout.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `token ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiClientLongTimeout.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      message.error('Phiên đăng nhập đã hết hạn');
      // Có thể redirect đến login page
    } else if (error.response?.status === 429) {
      message.warning('Quá nhiều requests, vui lòng thử lại sau');
    }
    return Promise.reject(error);
  }
);

// ==================== REPOSITORY API ====================
export const repositoryAPI = {  // Lấy repos từ database (requires authentication to filter by user)
  getFromDatabase: async () => {
    console.log('🔍 repositoryAPI.getFromDatabase: Making request to /repositories');
    const response = await apiClient.get('/repositories');
    console.log('✅ repositoryAPI.getFromDatabase: Response received', {
      status: response.status,
      dataLength: response.data?.length
    });
    return response.data || [];
  },

  // Lấy repos từ GitHub API (fallback)
  getFromGitHub: async () => {
    const response = await apiClient.get('/github/repos');
    return response.data || [];
  },

  // Intelligent fetch với fallback
  getIntelligent: async () => {
    try {
      console.log('🔍 Trying database first...');
      const data = await repositoryAPI.getFromDatabase();
      if (data && data.length > 0) {
        console.log('✅ Loaded from database');
        return { data, source: 'database' };
      }
    } catch (error) {
      console.log('❌ Database failed:', error.message);
    }

    try {
      console.log('🔍 Trying GitHub API fallback...');
      const data = await repositoryAPI.getFromGitHub();
      console.log('⚠️ Loaded from GitHub API');
      return { data, source: 'github' };
    } catch (error) {
      console.log('❌ GitHub API failed:', error.message);
      throw new Error('Không thể tải repositories từ bất kỳ nguồn nào');
    }
  }
};

// ==================== TASK API ====================
export const taskAPI = {
  // Lấy tasks theo repo cụ thể
  getByRepo: async (owner, repoName) => {
    const response = await apiClient.get(`/projects/${owner}/${repoName}/tasks`);
    return response.data || [];
  },

  // Lấy tất cả tasks (fallback)
  getAll: async (owner, repoName) => {
    const response = await apiClient.get('/tasks', {
      params: { limit: 100, offset: 0 }
    });
    const allTasks = response.data || [];
    return allTasks.filter(task => 
      task.repo_owner === owner && task.repo_name === repoName
    );
  },

  // Intelligent fetch với fallback
  getIntelligent: async (owner, repoName) => {
    try {
      console.log('🔍 Trying repo-specific endpoint...');
      const data = await taskAPI.getByRepo(owner, repoName);
      if (data && data.length > 0) {
        console.log('✅ Loaded repo-specific tasks');
        return { data, source: 'database' };
      }
    } catch (error) {
      console.log('❌ Repo-specific failed:', error.message);
    }

    try {
      console.log('🔍 Trying general tasks with filter...');
      const data = await taskAPI.getAll(owner, repoName);
      console.log('⚠️ Loaded from general tasks');
      return { data, source: 'fallback' };
    } catch (error) {
      console.log('❌ All task sources failed:', error.message);
      return { data: [], source: 'failed' };
    }
  },

  // Tạo task mới
  create: async (owner, repoName, taskData) => {
    const response = await apiClient.post(`/projects/${owner}/${repoName}/tasks`, taskData);
    return response.data;
  },

  // Cập nhật task
  update: async (owner, repoName, taskId, taskData) => {
    const response = await apiClient.put(`/projects/${owner}/${repoName}/tasks/${taskId}`, taskData);
    return response.data;
  },

  // Xóa task
  delete: async (owner, repoName, taskId) => {
    await apiClient.delete(`/projects/${owner}/${repoName}/tasks/${taskId}`);
  }
};

// ==================== COLLABORATOR API ====================
export const collaboratorAPI = {
  // 📊 Lấy collaborators từ database
  getCollaborators: async (owner, repoName) => {
    console.log(`🔍 Getting collaborators from database for ${owner}/${repoName}`);
    const timestamp = Date.now();
    const response = await apiClient.get(`/contributors/${owner}/${repoName}?t=${timestamp}`);
    
    const result = response.data;
    console.log('📊 Database response:', result);
    
    return {
      collaborators: result?.collaborators || [],
      hasSyncedData: result?.has_synced_data || false,
      message: result?.message || '',
      repository: result?.repository
    };
  },
  // � Sync collaborators từ GitHub vào database
  sync: async (owner, repoName) => {
    console.log(`🔄 Syncing collaborators for ${owner}/${repoName}`);
    const response = await apiClientLongTimeout.post(`/contributors/${owner}/${repoName}/sync`);
    console.log('✅ Sync response:', response.data);
    return response.data;  }
};

// ==================== BRANCH API ====================
export const branchAPI = {  // 🌿 Lấy branches từ database
  getBranches: async (owner, repoName) => {
    console.log(`🔍 Getting branches from database for ${owner}/${repoName}`);
    const response = await apiClient.get(`/${owner}/${repoName}/branches`);
    
    console.log('🌿 Branch database response:', response.data);
    return response.data?.branches || [];
  },

  // 🔄 Sync branches từ GitHub vào database
  sync: async (owner, repoName) => {
    console.log(`🔄 Syncing branches for ${owner}/${repoName}`);
    const response = await apiClientLongTimeout.post(`/github/${owner}/${repoName}/sync-branches`);
    console.log('✅ Branch sync response:', response.data);
    return response.data;
  }
};

export default apiClient;
