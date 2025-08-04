/**
 * useTaskAssign - Custom hook cho quản lý task assignment
 * Tuân thủ quy tắc KLTN04: Defensive programming, validation, immutability
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { message } from 'antd';
import { taskAPI } from '../services/api';

// Constants
const TASK_STATUSES = ['TODO', 'IN_PROGRESS', 'DONE', 'CANCELLED'];
const TASK_PRIORITIES = ['LOW', 'MEDIUM', 'HIGH', 'URGENT'];

const useTaskAssign = (selectedRepo) => {
  // State management
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    status: null,
    priority: null,
    assignee: null,
    search: ''
  });

  // Get repository data từ selectedRepo object
  const repoOwner = selectedRepo?.owner ? 
    (typeof selectedRepo.owner === 'string' ? selectedRepo.owner : selectedRepo.owner.login) : 
    null;
  const repoName = selectedRepo?.name;

  // Validation helper
  const validateTaskData = useCallback((taskData) => {
    if (!taskData) {
      throw new Error('Dữ liệu task không được để trống');
    }

    if (!taskData.title?.trim()) {
      throw new Error('Tiêu đề task không được để trống');
    }

    if (taskData.title.trim().length < 3) {
      throw new Error('Tiêu đề task phải có ít nhất 3 ký tự');
    }

    if (taskData.title.trim().length > 255) {
      throw new Error('Tiêu đề task không được quá 255 ký tự');
    }

    if (taskData.description && taskData.description.length > 1000) {
      throw new Error('Mô tả task không được quá 1000 ký tự');
    }

    if (taskData.status && !TASK_STATUSES.includes(taskData.status)) {
      throw new Error('Trạng thái task không hợp lệ');
    }

    if (taskData.priority && !TASK_PRIORITIES.includes(taskData.priority)) {
      throw new Error('Độ ưu tiên task không hợp lệ');
    }

    return true;
  }, []);

  // Error handler
  const handleError = useCallback((error, action = 'thực hiện thao tác') => {
    console.error(`Lỗi khi ${action}:`, error);
    
    let errorMessage = `Không thể ${action}`;
    
    if (error.response?.data?.message) {
      errorMessage = error.response.data.message;
    } else if (error.message) {
      errorMessage = error.message;
    }
    
    setError(errorMessage);
    message.error(errorMessage);
  }, []);

  // Load tasks từ API
  const loadTasks = useCallback(async (showLoading = true) => {
    if (!repoOwner || !repoName) {
      setTasks([]);
      return;
    }

    try {
      if (showLoading) {
        setLoading(true);
      }
      setError(null);

      const response = await taskAPI.getIntelligent(repoOwner, repoName);
      
      // Validation response
      if (!Array.isArray(response.data)) {
        throw new Error('Dữ liệu trả về không hợp lệ');
      }

      setTasks(response.data);
    } catch (error) {
      handleError(error, 'tải danh sách task');
      setTasks([]);
    } finally {
      setLoading(false);
    }
  }, [repoOwner, repoName, handleError]);

  // Create task
  const createTask = useCallback(async (taskData) => {
    try {
      validateTaskData(taskData);
      
      if (!repoOwner || !repoName) {
        throw new Error('Repository không được để trống');
      }

      setLoading(true);
      setError(null);

      const createData = {
        ...taskData,
        title: taskData.title.trim(),
        description: taskData.description?.trim() || '',
        status: taskData.status || 'TODO',
        priority: taskData.priority || 'MEDIUM'
      };

      const response = await taskAPI.create(repoOwner, repoName, createData);
      
      if (!response) {
        throw new Error('Không nhận được dữ liệu task mới');
      }

      // Immutable update
      setTasks(prevTasks => [...prevTasks, response]);
      message.success('Tạo task thành công');
      
      return response;
    } catch (error) {
      handleError(error, 'tạo task');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [repoOwner, repoName, validateTaskData, handleError]);

  // Update task
  const updateTask = useCallback(async (taskId, updateData) => {
    try {
      if (!taskId) {
        throw new Error('ID task không được để trống');
      }

      validateTaskData(updateData);
      setLoading(true);
      setError(null);

      const cleanUpdateData = {
        ...updateData,
        title: updateData.title.trim(),
        description: updateData.description?.trim() || ''
      };

      const response = await taskAPI.update(repoOwner, repoName, taskId, cleanUpdateData);
      
      if (!response) {
        throw new Error('Không nhận được dữ liệu task đã cập nhật');
      }

      // Immutable update
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.id === taskId 
            ? { ...task, ...response }
            : task
        )
      );
      
      message.success('Cập nhật task thành công');
      return response;
    } catch (error) {
      handleError(error, 'cập nhật task');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [repoOwner, repoName, validateTaskData, handleError]);

  // Update task status
  const updateTaskStatus = useCallback(async (taskId, newStatus) => {
    try {
      if (!taskId) {
        throw new Error('ID task không được để trống');
      }

      if (!TASK_STATUSES.includes(newStatus)) {
        throw new Error('Trạng thái task không hợp lệ');
      }

      setLoading(true);
      setError(null);

      const response = await taskAPI.updateTaskStatus(taskId, newStatus);
      
      if (!response.data) {
        throw new Error('Không nhận được dữ liệu task đã cập nhật');
      }

      // Immutable update
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.id === taskId 
            ? { ...task, status: newStatus, updated_at: new Date().toISOString() }
            : task
        )
      );
      
      return response.data;
    } catch (error) {
      handleError(error, 'cập nhật trạng thái task');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  // Delete task
  const deleteTask = useCallback(async (taskId) => {
    try {
      if (!taskId) {
        throw new Error('ID task không được để trống');
      }

      setLoading(true);
      setError(null);

      await taskAPI.delete(repoOwner, repoName, taskId);
      
      // Immutable update
      setTasks(prevTasks => prevTasks.filter(task => task.id !== taskId));
      message.success('Xóa task thành công');
    } catch (error) {
      handleError(error, 'xóa task');
      throw error;
    } finally {
      setLoading(false);
    }
  }, [repoOwner, repoName, handleError]);

  // Filter tasks
  const filteredTasks = useMemo(() => {
    if (!Array.isArray(tasks)) return [];

    return tasks.filter(task => {
      // Search filter
      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        const matchesSearch = 
          task.title?.toLowerCase().includes(searchLower) ||
          task.description?.toLowerCase().includes(searchLower) ||
          task.assignee_github_username?.toLowerCase().includes(searchLower);
        
        if (!matchesSearch) return false;
      }

      // Status filter
      if (filters.status && task.status !== filters.status) {
        return false;
      }

      // Priority filter
      if (filters.priority && task.priority !== filters.priority) {
        return false;
      }

      // Assignee filter
      if (filters.assignee && task.assignee_github_username !== filters.assignee) {
        return false;
      }

      return true;
    });
  }, [tasks, filters]);

  // Group tasks by status
  const groupedTasks = useMemo(() => {
    const groups = {
      TODO: [],
      IN_PROGRESS: [],
      DONE: [],
      CANCELLED: []
    };

    filteredTasks.forEach(task => {
      if (groups[task.status]) {
        groups[task.status].push(task);
      }
    });

    return groups;
  }, [filteredTasks]);

  // Task statistics
  const taskStats = useMemo(() => {
    return {
      total: tasks.length,
      todo: tasks.filter(t => t.status === 'TODO').length,
      inProgress: tasks.filter(t => t.status === 'IN_PROGRESS').length,
      done: tasks.filter(t => t.status === 'DONE').length,
      cancelled: tasks.filter(t => t.status === 'CANCELLED').length,
      overdue: tasks.filter(t => {
        if (!t.due_date) return false;
        return new Date(t.due_date) < new Date() && t.status !== 'DONE';
      }).length
    };
  }, [tasks]);

  // Update filters
  const updateFilters = useCallback((newFilters) => {
    setFilters(prevFilters => ({
      ...prevFilters,
      ...newFilters
    }));
  }, []);

  // Clear filters
  const clearFilters = useCallback(() => {
    setFilters({
      status: null,
      priority: null,
      assignee: null,
      search: ''
    });
  }, []);

  // Refresh tasks
  const refreshTasks = useCallback(() => {
    loadTasks(true);
  }, [loadTasks]);

  // Load tasks khi repository thay đổi
  useEffect(() => {
    loadTasks();
  }, [loadTasks]);

  return {
    // Data
    tasks: filteredTasks,
    groupedTasks,
    taskStats,
    
    // State
    loading,
    error,
    filters,
    
    // Actions
    createTask,
    updateTask,
    updateTaskStatus,
    deleteTask,
    loadTasks,
    refreshTasks,
    
    // Filters
    updateFilters,
    clearFilters,
    
    // Utils
    validateTaskData
  };
};

export default useTaskAssign;
