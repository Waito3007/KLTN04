// kanbanUtils.js
import { TASK_CARD_CONFIG } from './kanbanConstants';

/**
 * Lọc tasks theo status
 */
export const getTasksByStatus = (tasks, status) => {
  return tasks.filter(task => task.status === status);
};

/**
 * Truncate description 
 */
export const truncateDescription = (description) => {
  if (!description) return '';
  
  return description.length > TASK_CARD_CONFIG.DESCRIPTION_MAX_LENGTH
    ? description.substring(0, TASK_CARD_CONFIG.DESCRIPTION_MAX_LENGTH) + '...'
    : description;
};

/**
 * Format date cho display
 */
export const formatDate = (dateString) => {
  if (!dateString) return null;
  
  return new Date(dateString).toLocaleDateString('vi-VN', { 
    month: 'short', 
    day: 'numeric' 
  });
};

/**
 * Format full date cho tooltip
 */
export const formatFullDate = (dateString) => {
  if (!dateString) return '';
  
  return new Date(dateString).toLocaleDateString('vi-VN');
};
