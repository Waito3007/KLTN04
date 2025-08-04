// Task Management Utilities
// Tuân thủ KLTN04 coding principles

/**
 * Normalize repository owner data - xử lý cả string và object
 * @param {string|object} owner - Owner data từ API
 * @returns {string} - Normalized owner name
 */
export const normalizeOwner = (owner) => {
  if (!owner) return 'unknown';
  
  if (typeof owner === 'string') {
    return owner;
  }
  
  if (typeof owner === 'object') {
    return owner.login || owner.name || 'unknown';
  }
  
  return 'unknown';
};

/**
 * Validate task data trước khi submit
 * @param {object} taskData - Task data cần validate
 * @returns {object} - { isValid: boolean, errors: string[] }
 */
export const validateTaskData = (taskData) => {
  const errors = [];
  
  if (!taskData.title?.trim()) {
    errors.push('Tiêu đề task không được để trống');
  }
  
  if (!taskData.description?.trim()) {
    errors.push('Mô tả task không được để trống');
  }
  
  if (!taskData.priority) {
    errors.push('Độ ưu tiên phải được chọn');
  }
  
  if (!taskData.assignee?.trim()) {
    errors.push('Người được giao không được để trống');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Format priority text for display
 * @param {string} priority - Priority value
 * @returns {string} - Formatted priority text
 */
export const formatPriority = (priority) => {
  const priorityMap = {
    LOW: 'Thấp',
    MEDIUM: 'Trung bình', 
    HIGH: 'Cao',
    URGENT: 'Khẩn cấp'
  };
  
  return priorityMap[priority] || priority;
};

/**
 * Format status text for display
 * @param {string} status - Status value
 * @returns {string} - Formatted status text
 */
export const formatStatus = (status) => {
  const statusMap = {
    TODO: 'Cần làm',
    IN_PROGRESS: 'Đang làm',
    REVIEW: 'Đang review',
    DONE: 'Hoàn thành'
  };
  
  return statusMap[status] || status;
};

/**
 * Get priority color for UI
 * @param {string} priority - Priority value
 * @returns {string} - Color value
 */
export const getPriorityColor = (priority) => {
  const colorMap = {
    LOW: '#52c41a',
    MEDIUM: '#faad14',
    HIGH: '#fa8c16',
    URGENT: '#f5222d'
  };
  
  return colorMap[priority] || '#d9d9d9';
};

/**
 * Get status color for UI
 * @param {string} status - Status value
 * @returns {string} - Color value
 */
export const getStatusColor = (status) => {
  const colorMap = {
    TODO: '#d9d9d9',
    IN_PROGRESS: '#1890ff',
    REVIEW: '#faad14',
    DONE: '#52c41a'
  };
  
  return colorMap[status] || '#d9d9d9';
};

/**
 * Sort tasks by priority and created date
 * @param {array} tasks - Array of tasks
 * @returns {array} - Sorted tasks
 */
export const sortTasks = (tasks) => {
  const priorityOrder = { URGENT: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
  
  return [...tasks].sort((a, b) => {
    // Sort by priority first
    const priorityDiff = (priorityOrder[b.priority] || 0) - (priorityOrder[a.priority] || 0);
    if (priorityDiff !== 0) return priorityDiff;
    
    // Then by created date (newest first)
    return new Date(b.created_at) - new Date(a.created_at);
  });
};

/**
 * Filter tasks based on criteria
 * @param {array} tasks - Array of tasks
 * @param {object} filters - Filter criteria
 * @returns {array} - Filtered tasks
 */
export const filterTasks = (tasks, filters) => {
  return tasks.filter(task => {
    // Status filter
    if (filters.status && task.status !== filters.status) {
      return false;
    }
    
    // Priority filter
    if (filters.priority && task.priority !== filters.priority) {
      return false;
    }
    
    // Assignee filter
    if (filters.assignee && task.assignee !== filters.assignee) {
      return false;
    }
    
    // Search filter (title và description)
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      const titleMatch = task.title?.toLowerCase().includes(searchLower);
      const descMatch = task.description?.toLowerCase().includes(searchLower);
      if (!titleMatch && !descMatch) {
        return false;
      }
    }
    
    return true;
  });
};

/**
 * Group tasks by status for Kanban board
 * @param {array} tasks - Array of tasks
 * @returns {object} - Grouped tasks by status
 */
export const groupTasksByStatus = (tasks) => {
  const groups = {
    TODO: [],
    IN_PROGRESS: [],
    REVIEW: [],
    DONE: []
  };
  
  tasks.forEach(task => {
    if (groups[task.status]) {
      groups[task.status].push(task);
    }
  });
  
  return groups;
};

/**
 * Get task statistics
 * @param {array} tasks - Array of tasks
 * @returns {object} - Task statistics
 */
export const getTaskStats = (tasks) => {
  const stats = {
    total: tasks.length,
    todo: 0,
    inProgress: 0,
    review: 0,
    done: 0,
    byPriority: {
      LOW: 0,
      MEDIUM: 0,
      HIGH: 0,
      URGENT: 0
    }
  };
  
  tasks.forEach(task => {
    // Count by status
    switch (task.status) {
      case 'TODO':
        stats.todo++;
        break;
      case 'IN_PROGRESS':
        stats.inProgress++;
        break;
      case 'REVIEW':
        stats.review++;
        break;
      case 'DONE':
        stats.done++;
        break;
    }
    
    // Count by priority
    if (stats.byPriority[task.priority] !== undefined) {
      stats.byPriority[task.priority]++;
    }
  });
  
  return stats;
};
