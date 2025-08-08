/**
 * Authentication Service - Tương tác với backend OAuth system
 * Tuân thủ nguyên tắc KLTN04: Tách biệt logic auth
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

class AuthService {
  constructor() {
    this.token = localStorage.getItem('access_token');
    this.userInfo = null;
    
    // Setup axios interceptor cho authentication
    this.setupAxiosInterceptor();
  }

  /**
   * Setup axios interceptor để tự động thêm Bearer token
   */
  setupAxiosInterceptor() {
    axios.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token && config.url?.includes(API_BASE_URL)) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Setup response interceptor để xử lý 401 errors
    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.logout();
        }
        return Promise.reject(error);
      }
    );
  }

  /**
   * Kiểm tra user hiện tại thông qua backend OAuth
   */
  async getCurrentUser() {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No access token found');
      }

      // Test token với backend OAuth bằng cách gọi một endpoint protected
      const response = await axios.get(`${API_BASE_URL}/api/v1/tasks/health`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      if (response.status === 200) {
        // Token hợp lệ, lấy thông tin user từ localStorage hoặc GitHub API
        const storedProfile = localStorage.getItem('github_profile');
        if (storedProfile) {
          this.userInfo = JSON.parse(storedProfile);
          return this.userInfo;
        }
        
        // Nếu không có profile trong localStorage, có thể gọi GitHub API
        // hoặc backend endpoint để lấy user info
        return this.fetchUserFromGitHub(token);
      }
    } catch (error) {
      console.error('Error verifying current user:', error);
      // Nếu token không hợp lệ, logout
      if (error.response?.status === 401) {
        this.logout();
      }
      throw error;
    }
  }

  /**
   * Lấy thông tin user từ GitHub API (fallback)
   */
  async fetchUserFromGitHub(token) {
    try {
      const response = await axios.get('https://api.github.com/user', {
        headers: {
          Authorization: `token ${token}`,
          Accept: 'application/vnd.github.v3+json'
        }
      });

      const userProfile = response.data;
      
      // Lưu vào localStorage để dùng lần sau
      localStorage.setItem('github_profile', JSON.stringify(userProfile));
      
      this.userInfo = userProfile;
      return userProfile;
    } catch (error) {
      console.error('Error fetching user from GitHub:', error);
      throw error;
    }
  }

  /**
   * Kiểm tra xem user đã đăng nhập hay chưa
   */
  isAuthenticated() {
    const token = localStorage.getItem('access_token');
    const profile = localStorage.getItem('github_profile');
    return !!(token && profile);
  }

  /**
   * Logout user
   */
  logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('github_profile');
    this.userInfo = null;
    this.token = null;
    
    // Redirect to login page
    window.location.href = '/login';
  }

  /**
   * Lấy token hiện tại
   */
  getToken() {
    return localStorage.getItem('access_token');
  }

  /**
   * Lấy user info từ memory hoặc localStorage
   */
  getUserInfo() {
    if (this.userInfo) {
      return this.userInfo;
    }
    
    const storedProfile = localStorage.getItem('github_profile');
    if (storedProfile) {
      this.userInfo = JSON.parse(storedProfile);
      return this.userInfo;
    }
    
    return null;
  }

  /**
   * Test connection với backend OAuth system
   */
  async testOAuthConnection() {
    try {
      const token = this.getToken();
      if (!token) {
        return { success: false, error: 'No token available' };
      }

      // Test với task API health check (không cần auth)
      const healthResponse = await axios.get(`${API_BASE_URL}/api/v1/tasks/health`);
      
      // Test với protected endpoint
      const protectedResponse = await axios.get(`${API_BASE_URL}/api/v1/tasks/?page=1&page_size=1`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      return {
        success: true,
        healthCheck: healthResponse.status === 200,
        oauthWorking: protectedResponse.status === 200,
        userAuthenticated: true
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || error.message,
        status: error.response?.status
      };
    }
  }
}

// Export singleton instance
export default new AuthService();
