// frontend/src/utils/taskUtils.jsx
import React from 'react';
import { 
  ClockCircleOutlined, 
  ExclamationCircleOutlined, 
  CheckCircleOutlined 
} from '@ant-design/icons';

// ==================== TASK STATUS UTILITIES ====================
export const getStatusIcon = (status) => {
  const statusMap = {
    'todo': React.createElement(ClockCircleOutlined, { style: { color: '#faad14' } }),
    'in_progress': React.createElement(ExclamationCircleOutlined, { style: { color: '#1890ff' } }),
    'done': React.createElement(CheckCircleOutlined, { style: { color: '#52c41a' } }),
  };
  return statusMap[status] || React.createElement(ClockCircleOutlined);
};

export const getStatusText = (status) => {
  const statusMap = {
    'todo': 'Chờ thực hiện',
    'in_progress': 'Đang thực hiện',
    'done': 'Hoàn thành'
  };
  return statusMap[status] || 'Không xác định';
};

export const getStatusColor = (status) => {
  const colorMap = {
    'todo': '#faad14',
    'in_progress': '#1890ff', 
    'done': '#52c41a'
  };
  return colorMap[status] || '#d9d9d9';
};

// ==================== PRIORITY UTILITIES ====================
export const getPriorityColor = (priority) => {
  const colorMap = {
    'high': '#f5222d',
    'medium': '#fa8c16',
    'low': '#52c41a'
  };
  return colorMap[priority] || '#d9d9d9';
};

export const getPriorityText = (priority) => {
  const textMap = {
    'high': 'Cao',
    'medium': 'Trung bình',
    'low': 'Thấp'
  };
  return textMap[priority] || 'Không xác định';
};

export const getPriorityValue = (priority) => {
  const valueMap = {
    'high': 3,
    'medium': 2,
    'low': 1
  };
  return valueMap[priority] || 0;
};

// ==================== TASK FILTERING ====================
export const filterTasks = (tasks, filters) => {
  const { searchText, statusFilter, priorityFilter, assigneeFilter } = filters;
  
  return tasks.filter(task => {
    // Search filter
    const matchesSearch = !searchText || (
      task.title.toLowerCase().includes(searchText.toLowerCase()) ||
      task.description?.toLowerCase().includes(searchText.toLowerCase()) ||
      task.assignee.toLowerCase().includes(searchText.toLowerCase())
    );
    
    // Status filter
    const matchesStatus = statusFilter === 'all' || task.status === statusFilter;
    
    // Priority filter
    const matchesPriority = priorityFilter === 'all' || task.priority === priorityFilter;
    
    // Assignee filter
    const matchesAssignee = assigneeFilter === 'all' || task.assignee === assigneeFilter;
    
    return matchesSearch && matchesStatus && matchesPriority && matchesAssignee;
  });
};

// ==================== TASK STATISTICS ====================
export const calculateTaskStats = (tasks) => {
  const total = tasks.length;
  const completed = tasks.filter(t => t.status === 'done').length;
  const inProgress = tasks.filter(t => t.status === 'in_progress').length;
  const todo = tasks.filter(t => t.status === 'todo').length;
  const highPriority = tasks.filter(t => t.priority === 'high').length;
  
  return {
    total,
    completed,
    inProgress,
    todo,
    highPriority,
    completionRate: total > 0 ? Math.round((completed / total) * 100) : 0
  };
};

// ==================== TASK GROUPING ====================
export const groupTasksByStatus = (tasks) => {
  return {
    todo: tasks.filter(t => t.status === 'todo'),
    in_progress: tasks.filter(t => t.status === 'in_progress'),
    done: tasks.filter(t => t.status === 'done')
  };
};

export const groupTasksByPriority = (tasks) => {
  return {
    high: tasks.filter(t => t.priority === 'high'),
    medium: tasks.filter(t => t.priority === 'medium'),
    low: tasks.filter(t => t.priority === 'low')
  };
};

export const groupTasksByAssignee = (tasks) => {
  const groups = {};
  tasks.forEach(task => {
    const assignee = task.assignee || 'unassigned';
    if (!groups[assignee]) {
      groups[assignee] = [];
    }
    groups[assignee].push(task);
  });
  return groups;
};

// ==================== TASK SORTING ====================
export const sortTasks = (tasks, sortBy = 'created_at', sortOrder = 'desc') => {
  return [...tasks].sort((a, b) => {
    let aVal = a[sortBy];
    let bVal = b[sortBy];
    
    // Special handling for different data types
    if (sortBy === 'priority') {
      aVal = getPriorityValue(aVal);
      bVal = getPriorityValue(bVal);
    } else if (sortBy === 'due_date' || sortBy === 'created_at') {
      aVal = new Date(aVal || 0);
      bVal = new Date(bVal || 0);
    } else if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }
    
    if (sortOrder === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });
};

// ==================== TASK VALIDATION ====================
export const validateTask = (taskData) => {
  const errors = {};
  
  if (!taskData.title || taskData.title.trim() === '') {
    errors.title = 'Tiêu đề task không được để trống';
  }
  
  if (!taskData.assignee || taskData.assignee.trim() === '') {
    errors.assignee = 'Phải chỉ định người thực hiện';
  }
  
  if (!taskData.priority) {
    errors.priority = 'Phải chọn mức độ ưu tiên';
  }
  
  if (taskData.due_date && new Date(taskData.due_date) < new Date()) {
    errors.due_date = 'Ngày hết hạn không thể là quá khứ';
  }
  
  return {
    isValid: Object.keys(errors).length === 0,
    errors
  };
};

// ==================== FORMAT HELPERS ====================
export const formatTaskForAPI = (formValues) => {
  return {
    ...formValues,
    due_date: formValues.dueDate ? formValues.dueDate.format('YYYY-MM-DD') : null,
    status: formValues.status || 'todo'
  };
};

export const formatTaskForForm = (task) => {
  return {
    title: task.title,
    description: task.description,
    assignee: task.assignee,
    priority: task.priority,
    status: task.status,
    dueDate: task.due_date ? new Date(task.due_date) : null
  };
};

// ==================== TASK OPERATIONS ====================
export const getNextStatus = (currentStatus) => {
  const statusFlow = {
    'todo': 'in_progress',
    'in_progress': 'done',
    'done': 'todo' // Reset cycle
  };
  return statusFlow[currentStatus] || 'todo';
};

export const canEditTask = (task, currentUser) => {
  // Business logic for task editing permissions
  return task.assignee === currentUser.login || 
         task.created_by === currentUser.login ||
         currentUser.role === 'admin';
};

export const getTaskDeadlineStatus = (dueDate) => {
  if (!dueDate) return 'no-deadline';
  
  const today = new Date();
  const deadline = new Date(dueDate);
  const diffDays = Math.ceil((deadline - today) / (1000 * 60 * 60 * 24));
  
  if (diffDays < 0) return 'overdue';
  if (diffDays === 0) return 'due-today';
  if (diffDays <= 3) return 'due-soon';
  return 'on-track';
};

export default {
  getStatusIcon,
  getStatusText,
  getStatusColor,
  getPriorityColor,
  getPriorityText,
  filterTasks,
  calculateTaskStats,
  groupTasksByStatus,
  sortTasks,
  validateTask,
  formatTaskForAPI,
  formatTaskForForm,
  getNextStatus,
  canEditTask,
  getTaskDeadlineStatus
};
