/**
 * taskAPI - Service quản lý API cho tasks
 * Tuân thủ quy tắc KLTN04: Defensive programming, error handling, validation
 */

import apiClient from './api';

// Constants
const API_ENDPOINTS = {
  TASKS: '/v1/tasks',
  TASK_BY_ID: (id) => `/v1/tasks/${id}`,
  TASK_STATUS: (id) => `/v1/tasks/${id}/status`,
  REPOSITORY_TASKS: (repoId) => `/v1/tasks?repository_id=${repoId}`
};

// Request timeout (30 seconds)
const REQUEST_TIMEOUT = 30000;

class TaskAPI {
  /**
   * Validate task data trước khi gửi request
   */
  validateTaskData(data) {
    if (!data) {
      throw new Error('Dữ liệu task không được để trống');
    }

    if (!data.title?.trim()) {
      throw new Error('Tiêu đề task không được để trống');
    }

    if (data.title.trim().length < 3) {
      throw new Error('Tiêu đề task phải có ít nhất 3 ký tự');
    }

    if (data.title.trim().length > 255) {
      throw new Error('Tiêu đề task không được quá 255 ký tự');
    }

    if (data.description && data.description.length > 1000) {
      throw new Error('Mô tả task không được quá 1000 ký tự');
    }

    return true;
  }

  /**
   * Tạo task mới
   */
  async createTask(taskData) {
    try {
      this.validateTaskData(taskData);

      // Backend schema cần repo_owner và repo_name thay vì repository_id
      if (!taskData.repo_owner) {
        throw new Error('Repo owner không được để trống');
      }
      if (!taskData.repo_name) {
        throw new Error('Repo name không được để trống');
      }

      const cleanData = {
        title: taskData.title.trim(),
        description: taskData.description?.trim() || '',
        repo_owner: taskData.repo_owner,
        repo_name: taskData.repo_name,
        status: taskData.status || 'TODO',
        priority: taskData.priority || 'MEDIUM',
        assignee_github_username: taskData.assignee_github_username?.trim() || null,
        due_date: taskData.due_date || null
      };

      const response = await apiClient.post(API_ENDPOINTS.TASKS, cleanData, {
        timeout: REQUEST_TIMEOUT
      });

      if (!response.data) {
        throw new Error('Không nhận được dữ liệu task mới từ server');
      }

      return {
        status: 'success',
        data: response.data,
        message: 'Tạo task thành công'
      };
    } catch (error) {
      console.error('Lỗi khi tạo task:', error);
      throw this.handleError(error, 'tạo task');
    }
  }

  /**
   * Lấy danh sách tasks theo repository
   */
  async getTasks(repoOwner, repoName, filters = {}) {
    try {
      if (!repoOwner) {
        throw new Error('Repo owner không được để trống');
      }
      if (!repoName) {
        throw new Error('Repo name không được để trống');
      }

      const params = new URLSearchParams({
        repo_owner: repoOwner,
        repo_name: repoName
      });

      // Thêm filters vào params
      if (filters.status) {
        params.append('status', filters.status);
      }
      if (filters.priority) {
        params.append('priority', filters.priority);
      }
      if (filters.assignee) {
        params.append('assignee_github_username', filters.assignee);
      }
      if (filters.search) {
        // Backend không hỗ trợ search trực tiếp, sẽ filter ở frontend
      }

      const response = await apiClient.get(`${API_ENDPOINTS.TASKS}?${params}`, {
        timeout: REQUEST_TIMEOUT
      });

      let tasks = response.data?.tasks || [];

      // Filter theo search term ở frontend nếu cần
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        tasks = tasks.filter(task => 
          task.title?.toLowerCase().includes(searchLower) ||
          task.description?.toLowerCase().includes(searchLower)
        );
      }

      return {
        status: 'success',
        data: tasks,
        total: response.data?.total || tasks.length
      };
    } catch (error) {
      console.error('Lỗi khi tải tasks:', error);
      throw this.handleError(error, 'tải danh sách tasks');
    }
  }

  /**
   * Lấy chi tiết task theo ID
   */
  async getTaskById(taskId) {
    try {
      if (!taskId) {
        throw new Error('Task ID không được để trống');
      }

      const response = await apiClient.get(API_ENDPOINTS.TASK_BY_ID(taskId), {
        timeout: REQUEST_TIMEOUT
      });

      if (!response.data) {
        throw new Error('Không tìm thấy task');
      }

      return {
        status: 'success',
        data: response.data
      };
    } catch (error) {
      console.error('Lỗi khi tải task:', error);
      throw this.handleError(error, 'tải thông tin task');
    }
  }

  /**
   * Cập nhật task
   */
  async updateTask(taskId, updateData) {
    try {
      if (!taskId) {
        throw new Error('Task ID không được để trống');
      }

      this.validateTaskData(updateData);

      const cleanData = {
        title: updateData.title.trim(),
        description: updateData.description?.trim() || '',
        status: updateData.status,
        priority: updateData.priority,
        assignee_github_username: updateData.assignee_github_username?.trim() || null,
        due_date: updateData.due_date || null
      };

      const response = await apiClient.put(API_ENDPOINTS.TASK_BY_ID(taskId), cleanData, {
        timeout: REQUEST_TIMEOUT
      });

      if (!response.data) {
        throw new Error('Không nhận được dữ liệu task đã cập nhật');
      }

      return {
        status: 'success',
        data: response.data,
        message: 'Cập nhật task thành công'
      };
    } catch (error) {
      console.error('Lỗi khi cập nhật task:', error);
      throw this.handleError(error, 'cập nhật task');
    }
  }

  /**
   * Cập nhật trạng thái task
   */
  async updateTaskStatus(taskId, newStatus) {
    try {
      if (!taskId) {
        throw new Error('Task ID không được để trống');
      }

      if (!newStatus) {
        throw new Error('Trạng thái mới không được để trống');
      }

      const validStatuses = ['TODO', 'IN_PROGRESS', 'DONE', 'CANCELLED'];
      if (!validStatuses.includes(newStatus)) {
        throw new Error('Trạng thái task không hợp lệ');
      }

      const response = await apiClient.patch(API_ENDPOINTS.TASK_STATUS(taskId), {
        status: newStatus
      }, {
        timeout: REQUEST_TIMEOUT
      });

      if (!response.data) {
        throw new Error('Không nhận được dữ liệu task đã cập nhật');
      }

      return {
        status: 'success',
        data: response.data,
        message: 'Cập nhật trạng thái task thành công'
      };
    } catch (error) {
      console.error('Lỗi khi cập nhật trạng thái task:', error);
      throw this.handleError(error, 'cập nhật trạng thái task');
    }
  }

  /**
   * Xóa task
   */
  async deleteTask(taskId) {
    try {
      if (!taskId) {
        throw new Error('Task ID không được để trống');
      }

      await apiClient.delete(API_ENDPOINTS.TASK_BY_ID(taskId), {
        timeout: REQUEST_TIMEOUT
      });

      return {
        status: 'success',
        message: 'Xóa task thành công'
      };
    } catch (error) {
      console.error('Lỗi khi xóa task:', error);
      throw this.handleError(error, 'xóa task');
    }
  }

  /**
   * Lấy thống kê tasks theo repository
   */
  async getTaskStats(repoOwner, repoName) {
    try {
      if (!repoOwner) {
        throw new Error('Repo owner không được để trống');
      }
      if (!repoName) {
        throw new Error('Repo name không được để trống');
      }

      const response = await apiClient.get(`${API_ENDPOINTS.TASKS}/stats/summary?repo_owner=${repoOwner}&repo_name=${repoName}`, {
        timeout: REQUEST_TIMEOUT
      });

      return {
        status: 'success',
        data: response.data || {
          total: 0,
          todo: 0,
          in_progress: 0,
          done: 0,
          cancelled: 0,
          overdue: 0
        }
      };
    } catch (error) {
      console.error('Lỗi khi tải thống kê tasks:', error);
      throw this.handleError(error, 'tải thống kê tasks');
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
          errorMessage = 'Bạn không có quyền thực hiện thao tác này';
          errorCode = 'FORBIDDEN';
          break;
        case 404:
          errorMessage = 'Không tìm thấy task hoặc repository';
          errorCode = 'NOT_FOUND';
          break;
        case 409:
          errorMessage = data?.message || 'Xung đột dữ liệu';
          errorCode = 'CONFLICT';
          break;
        case 422:
          errorMessage = data?.message || 'Dữ liệu không hợp lệ';
          errorCode = 'VALIDATION_ERROR';
          break;
        case 500:
          errorMessage = 'Lỗi server. Vui lòng thử lại sau.';
          errorCode = 'SERVER_ERROR';
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
export const taskAPI = new TaskAPI();
export default taskAPI;
