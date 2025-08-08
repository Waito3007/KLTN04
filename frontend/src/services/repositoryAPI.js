/**
 * repositoryAPI - Service quản lý API cho repositories
 * Tuân thủ quy tắc KLTN04: Defensive programming, error handling, validation
 */

import apiClient from './api';

// Constants
const API_ENDPOINTS = {
  REPOSITORIES: '/repositories',
  REPOSITORY_BY_ID: (id) => `/repositories/${id}`,
  USER_REPOSITORIES: '/repositories/user',
  REPOSITORY_COLLABORATORS: (id) => `/repositories/${id}/collaborators`,
  REPOSITORY_STATS: (id) => `/repositories/${id}/stats`
};

// Request timeout (30 seconds)
const REQUEST_TIMEOUT = 30000;

class RepositoryAPI {
  /**
   * Validate repository data
   */
  validateRepositoryData(data) {
    if (!data) {
      throw new Error('Dữ liệu repository không được để trống');
    }

    if (!data.name?.trim()) {
      throw new Error('Tên repository không được để trống');
    }

    if (!data.owner?.trim()) {
      throw new Error('Owner repository không được để trống');
    }

    return true;
  }

  /**
   * Lấy danh sách repositories của user
   */
  async getRepositories(filters = {}) {
    try {
      const params = new URLSearchParams();

      // Thêm filters vào params
      if (filters.type) {
        params.append('type', filters.type); // 'all', 'public', 'private'
      }
      if (filters.sort) {
        params.append('sort', filters.sort); // 'name', 'created', 'updated'
      }
      if (filters.direction) {
        params.append('direction', filters.direction); // 'asc', 'desc'
      }
      if (filters.search) {
        params.append('search', filters.search.trim());
      }

      const url = params.toString() 
        ? `${API_ENDPOINTS.USER_REPOSITORIES}?${params}`
        : API_ENDPOINTS.USER_REPOSITORIES;

      const response = await apiClient.get(url, {
        timeout: REQUEST_TIMEOUT
      });

      if (!Array.isArray(response.data)) {
        throw new Error('Dữ liệu repositories không hợp lệ');
      }

      // Validate repository structure
      const validRepositories = response.data.filter(repo => {
        try {
          return repo && 
                 typeof repo.id === 'number' && 
                 typeof repo.name === 'string' && 
                 typeof repo.owner === 'string';
        } catch {
          return false;
        }
      });

      return {
        status: 'success',
        data: validRepositories,
        total: validRepositories.length
      };
    } catch (error) {
      console.error('Lỗi khi tải repositories:', error);
      throw this.handleError(error, 'tải danh sách repositories');
    }
  }

  /**
   * Lấy chi tiết repository theo ID
   */
  async getRepositoryById(repositoryId) {
    try {
      if (!repositoryId) {
        throw new Error('Repository ID không được để trống');
      }

      const response = await apiClient.get(API_ENDPOINTS.REPOSITORY_BY_ID(repositoryId), {
        timeout: REQUEST_TIMEOUT
      });

      if (!response.data) {
        throw new Error('Không tìm thấy repository');
      }

      return {
        status: 'success',
        data: response.data
      };
    } catch (error) {
      console.error('Lỗi khi tải repository:', error);
      throw this.handleError(error, 'tải thông tin repository');
    }
  }

  /**
   * Lấy danh sách collaborators của repository
   */
  async getRepositoryCollaborators(repositoryId) {
    try {
      if (!repositoryId) {
        throw new Error('Repository ID không được để trống');
      }

      const response = await apiClient.get(API_ENDPOINTS.REPOSITORY_COLLABORATORS(repositoryId), {
        timeout: REQUEST_TIMEOUT
      });

      if (!Array.isArray(response.data)) {
        console.warn('Dữ liệu collaborators không hợp lệ, trả về mảng rỗng');
        return {
          status: 'success',
          data: [],
          total: 0
        };
      }

      return {
        status: 'success',
        data: response.data,
        total: response.data.length
      };
    } catch (error) {
      console.error('Lỗi khi tải collaborators:', error);
      throw this.handleError(error, 'tải danh sách collaborators');
    }
  }

  /**
   * Lấy thống kê repository
   */
  async getRepositoryStats(repositoryId) {
    try {
      if (!repositoryId) {
        throw new Error('Repository ID không được để trống');
      }

      const response = await apiClient.get(API_ENDPOINTS.REPOSITORY_STATS(repositoryId), {
        timeout: REQUEST_TIMEOUT
      });

      return {
        status: 'success',
        data: response.data || {
          commits: 0,
          branches: 0,
          contributors: 0,
          issues: 0,
          pull_requests: 0,
          stars: 0,
          forks: 0
        }
      };
    } catch (error) {
      console.error('Lỗi khi tải thống kê repository:', error);
      throw this.handleError(error, 'tải thống kê repository');
    }
  }

  /**
   * Sync repository với GitHub
   */
  async syncRepository(repositoryId) {
    try {
      if (!repositoryId) {
        throw new Error('Repository ID không được để trống');
      }

      const response = await apiClient.post(`${API_ENDPOINTS.REPOSITORY_BY_ID(repositoryId)}/sync`, {}, {
        timeout: 60000 // 60 seconds for sync operation
      });

      return {
        status: 'success',
        data: response.data,
        message: 'Sync repository thành công'
      };
    } catch (error) {
      console.error('Lỗi khi sync repository:', error);
      throw this.handleError(error, 'đồng bộ repository');
    }
  }

  /**
   * Tìm kiếm repositories
   */
  async searchRepositories(query, filters = {}) {
    try {
      if (!query?.trim()) {
        throw new Error('Từ khóa tìm kiếm không được để trống');
      }

      const params = new URLSearchParams({
        q: query.trim()
      });

      // Thêm filters
      if (filters.type) {
        params.append('type', filters.type);
      }
      if (filters.sort) {
        params.append('sort', filters.sort);
      }
      if (filters.direction) {
        params.append('direction', filters.direction);
      }

      const response = await apiClient.get(`${API_ENDPOINTS.REPOSITORIES}/search?${params}`, {
        timeout: REQUEST_TIMEOUT
      });

      if (!Array.isArray(response.data)) {
        throw new Error('Dữ liệu tìm kiếm không hợp lệ');
      }

      return {
        status: 'success',
        data: response.data,
        total: response.data.length,
        query: query.trim()
      };
    } catch (error) {
      console.error('Lỗi khi tìm kiếm repositories:', error);
      throw this.handleError(error, 'tìm kiếm repositories');
    }
  }

  /**
   * Refresh repository data
   */
  async refreshRepository(repositoryId) {
    try {
      if (!repositoryId) {
        throw new Error('Repository ID không được để trống');
      }

      const response = await apiClient.post(`${API_ENDPOINTS.REPOSITORY_BY_ID(repositoryId)}/refresh`, {}, {
        timeout: REQUEST_TIMEOUT
      });

      return {
        status: 'success',
        data: response.data,
        message: 'Làm mới repository thành công'
      };
    } catch (error) {
      console.error('Lỗi khi làm mới repository:', error);
      throw this.handleError(error, 'làm mới repository');
    }
  }

  /**
   * Xử lý lỗi chung
   */
  handleError(error, action) {
    let errorMessage = `Không thể ${action}`;
    let errorCode = 'UNKNOWN_ERROR';

    if (error.response) {
      // Server trả về response với status code lỗi
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          errorMessage = data?.message || 'Dữ liệu không hợp lệ';
          errorCode = 'BAD_REQUEST';
          break;
        case 401:
          errorMessage = 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.';
          errorCode = 'UNAUTHORIZED';
          break;
        case 403:
          errorMessage = 'Bạn không có quyền truy cập repository này';
          errorCode = 'FORBIDDEN';
          break;
        case 404:
          errorMessage = 'Không tìm thấy repository';
          errorCode = 'NOT_FOUND';
          break;
        case 409:
          errorMessage = data?.message || 'Repository đã tồn tại';
          errorCode = 'CONFLICT';
          break;
        case 422:
          errorMessage = data?.message || 'Dữ liệu repository không hợp lệ';
          errorCode = 'VALIDATION_ERROR';
          break;
        case 500:
          errorMessage = 'Lỗi server. Vui lòng thử lại sau.';
          errorCode = 'SERVER_ERROR';
          break;
        case 502:
          errorMessage = 'Lỗi kết nối với GitHub. Vui lòng thử lại sau.';
          errorCode = 'GITHUB_ERROR';
          break;
        case 503:
          errorMessage = 'Dịch vụ tạm thời không khả dụng. Vui lòng thử lại sau.';
          errorCode = 'SERVICE_UNAVAILABLE';
          break;
        default:
          errorMessage = data?.message || `Lỗi ${status}: ${action}`;
          errorCode = `HTTP_${status}`;
      }
    } else if (error.request) {
      // Request được gửi nhưng không nhận được response
      errorMessage = 'Không thể kết nối tới server. Vui lòng kiểm tra kết nối mạng.';
      errorCode = 'NETWORK_ERROR';
    } else if (error.message) {
      // Lỗi khác
      errorMessage = error.message;
      errorCode = 'CUSTOM_ERROR';
    }

    const formattedError = new Error(errorMessage);
    formattedError.code = errorCode;
    formattedError.originalError = error;
    
    return formattedError;
  }
}

// Export singleton instance
export const repositoryAPI = new RepositoryAPI();
export default repositoryAPI;
