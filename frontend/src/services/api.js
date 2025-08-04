// frontend/src/services/api.js
import axios from 'axios';
import { message } from 'antd';

// Centralized API configuration
const API_BASE_URL = 'http://localhost:8000/api';

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
      config.headers.Authorization = `Bearer ${token}`;
      console.log(`🔑 Authorization header set: Bearer ${token.substring(0, 10)}...`);
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
    // Log chi tiết lỗi API
    if (error.response) {
      console.error(`❌ API Error:`, {
        url: error.config?.url,
        method: error.config?.method,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message
      });
      // Log thêm chi tiết nếu có
      if (error.response.data) {
        console.error('❌ API Error Data:', error.response.data);
      }
    } else {
      console.error('❌ API Error:', error.message || error);
    }
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
      config.headers.Authorization = `Bearer ${token}`;
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

// ==================== SYNC API ====================
export const syncAPI = {
  // 🔄 Đồng bộ toàn bộ repository (repo, branches, commits, issues, PRs)
  syncAll: async (owner, repoName) => {
    console.log(`🔄 Starting complete sync for ${owner}/${repoName}`);
    try {
      const response = await apiClientLongTimeout.post(`/github/${owner}/${repoName}/sync-all`);
      console.log('✅ Complete sync response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Complete sync error:', error);
      throw error;
    }
  },

  // 🔄 Đồng bộ cơ bản (chỉ repository info)
  syncBasic: async (owner, repoName) => {
    console.log(`🔄 Starting basic sync for ${owner}/${repoName}`);
    try {
      const response = await apiClientLongTimeout.post(`/github/${owner}/${repoName}/sync-basic`);
      console.log('✅ Basic sync response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Basic sync error:', error);
      throw error;
    }
  },

  // 🔄 Đồng bộ nâng cao (repo + branches với thông tin chi tiết)
  syncEnhanced: async (owner, repoName) => {
    console.log(`🔄 Starting enhanced sync for ${owner}/${repoName}`);
    try {
      const response = await apiClientLongTimeout.post(`/github/${owner}/${repoName}/sync-enhanced`);
      console.log('✅ Enhanced sync response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Enhanced sync error:', error);
      throw error;
    }
  },

  // 📊 Kiểm tra trạng thái GitHub API
  checkGitHubStatus: async () => {
    console.log('🔍 Checking GitHub API status');
    try {
      const response = await apiClient.get('/github/status');
      console.log('📊 GitHub status response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ GitHub status check error:', error);
      throw error;
    }
  },

  // 📋 Lấy danh sách repositories
  getRepositories: async (perPage = 30, page = 1) => {
    console.log('📋 Getting user repositories from GitHub');
    try {
      const response = await apiClient.get(`/github/repositories?per_page=${perPage}&page=${page}`);
      console.log('📋 Repositories response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Get repositories error:', error);
      throw error;
    }
  },

  // 📊 Lấy thống kê repository
  getRepositoryStats: async (owner, repoName) => {
    console.log(`📊 Getting repository stats for ${owner}/${repoName}`);
    try {
      const response = await apiClient.get(`/github/${owner}/${repoName}/stats`);
      console.log('📊 Repository stats response:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Repository stats error:', error);
      throw error;
    }
  }
};

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

// ==================== ASSIGNMENT RECOMMENDATION API ====================
export const assignmentRecommendationAPI = {
  // 🎯 Lấy kỹ năng của các thành viên (simplified endpoint)
  getMemberSkills: async (owner, repoName) => {
    console.log(`🔍 Getting member skills for ${owner}/${repoName}`);
    const response = await apiClient.get(`/assignment-recommendation/member-skills-simple/${owner}/${repoName}`);
    console.log('🎯 Member skills response:', response.data);
    return response.data;
  },

  // 🤖 Lấy gợi ý phân công cho task
  getRecommendations: async (owner, repoName, taskDescription, requiredSkills = [], maxRecommendations = 3) => {
    console.log(`🤖 Getting recommendations for task: "${taskDescription}"`);
    const response = await apiClient.post(`/assignment-recommendation/recommend/${owner}/${repoName}`, {
      task_description: taskDescription,
      required_skills: requiredSkills,
      max_recommendations: maxRecommendations
    });
    console.log('🤖 Recommendations response:', response.data);
    return response.data;
  },

  // ⚖️ Phân công thông minh với cân bằng workload (simplified endpoint)
  getSmartAssignment: async (owner, repoName, taskDescription, requiredSkills = [], considerWorkload = true) => {
    console.log(`⚖️ Getting smart assignment for task: "${taskDescription}"`);
    const response = await apiClient.post(`/assignment-recommendation/smart-assign-simple/${owner}/${repoName}`, {
      task_description: taskDescription,
      required_skills: requiredSkills,
      consider_workload: considerWorkload
    });
    console.log('⚖️ Smart assignment response:', response.data);
    return response.data;
  },

  // 📊 Lấy insights về team
  getTeamInsights: async (owner, repoName) => {
    console.log(`📊 Getting team insights for ${owner}/${repoName}`);
    const response = await apiClient.get(`/assignment-recommendation/team-insights/${owner}/${repoName}`);
    console.log('📊 Team insights response:', response.data);
    return response.data;
  },

  // 📈 Lấy phân tích workload
  getWorkloadAnalysis: async (owner, repoName) => {
    console.log(`📈 Getting workload analysis for ${owner}/${repoName}`);
    const response = await apiClient.get(`/assignment-recommendation/workload-analysis/${owner}/${repoName}`);
    console.log('📈 Workload analysis response:', response.data);
    return response.data;
  },

  // 🔍 Lấy chi tiết kỹ năng của một thành viên
  getMemberSkillDetails: async (owner, repoName, username) => {
    console.log(`🔍 Getting skill details for member: ${username}`);
    const response = await apiClient.get(`/assignment-recommendation/member-skills/${owner}/${repoName}/${username}`);
    console.log('🔍 Member skill details response:', response.data);
    return response.data;
  }
};

// Repository Sync Manager API
export const repoSyncAPI = {
  // Get sync status for all repositories
  getSyncStatus: async () => {
    console.log('📊 Getting repositories sync status');
    const response = await apiClient.get('/repositories/sync-status');
    console.log('📊 Sync status response:', response.data);
    return response.data;
  },

  // Get user repositories from GitHub
  getUserRepositories: async (page = 1, perPage = 30) => {
    console.log(`📚 Getting user repositories (page ${page})`);
    const response = await apiClient.get(`/github/user/repositories?page=${page}&per_page=${perPage}`);
    console.log('📚 User repositories response:', response.data);
    return response.data;
  },

  // Sync single repository
  syncRepository: async (owner, repo, syncType = 'optimized') => {
    console.log(`🔄 Syncing repository ${owner}/${repo} with type ${syncType}`);
    // Map sync types to actual backend endpoints
    let endpoint;
    switch (syncType) {
      case 'basic':
        endpoint = `/github/${owner}/${repo}/sync-basic`;
        break;
      case 'enhanced':
        endpoint = `/github/${owner}/${repo}/sync-enhanced`;
        break;
      case 'optimized':
      default:
        endpoint = `/github/${owner}/${repo}/sync-all`;
        break;
    }
    
    const response = await apiClientLongTimeout.post(endpoint);
    console.log('🔄 Sync response:', response.data);
    return response.data;
  },

  // Get sync events for a repository
  getRepoEvents: async (owner, repo) => {
    console.log(`📈 Getting sync events for ${owner}/${repo}`);
    const response = await apiClient.get(`/sync-events/repositories/${owner}/${repo}/events`);
    console.log('📈 Repo events response:', response.data);
    return response.data;
  },

  // Get all sync events
  getAllSyncEvents: async () => {
    console.log('📈 Getting all sync events');
    const response = await apiClient.get('/sync-events/sync-events');
    console.log('📈 All sync events response:', response.data);
    return response.data;
  },

  // Clear sync events for a repository
  clearRepoEvents: async (owner, repo) => {
    console.log(`🗑️ Clearing sync events for ${owner}/${repo}`);
    const response = await apiClient.delete(`/sync-events/repositories/${owner}/${repo}/events`);
    console.log('🗑️ Clear events response:', response.data);
    return response.data;
  }
};

export default apiClient;
