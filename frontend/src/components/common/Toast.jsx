// Toast component để hiển thị thông báo nhất quán với context support
import { App, notification } from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import theme from './theme';
import './toast.css'; // Import CSS styling cho toast overlay

// Cấu hình mặc định cho notification - đảm bảo hiển thị đúng vị trí
notification.config({
  placement: 'topRight',
  top: 24,
  duration: 4.5,
  rtl: false,
  getContainer: () => document.body, // Đảm bảo hiển thị trên body
});

// Hook để sử dụng Toast với context support (RECOMMENDED - không có warning)
export const useToast = () => {
  const { notification: contextNotification } = App.useApp();
  
  return {
    success: (content, duration = 3) => {
      return contextNotification.success({
        message: content,
        duration,
        icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
        className: 'custom-notification-success',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.success}30`,
          boxShadow: '0 4px 12px rgba(82, 196, 26, 0.15)'
        }
      });
    },

    error: (content, duration = 4) => {
      return contextNotification.error({
        message: content,
        duration,
        icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
        className: 'custom-notification-error',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.danger}30`,
          boxShadow: '0 4px 12px rgba(255, 77, 79, 0.15)'
        }
      });
    },

    warning: (content, duration = 3) => {
      return contextNotification.warning({
        message: content,
        duration,
        icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
        className: 'custom-notification-warning',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.warning}30`,
          boxShadow: '0 4px 12px rgba(250, 173, 20, 0.15)'
        }
      });
    },

    info: (content, duration = 3) => {
      return contextNotification.info({
        message: content,
        duration,
        icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
        className: 'custom-notification-info',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.info}30`,
          boxShadow: '0 4px 12px rgba(24, 144, 255, 0.15)'
        }
      });
    },

    loading: (content = 'Đang xử lý...', duration = 0) => {
      return contextNotification.open({
        message: content,
        duration,
        icon: <LoadingOutlined style={{ color: theme.colors.info }} />,
        className: 'custom-notification-loading',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.info}30`,
          boxShadow: '0 4px 12px rgba(24, 144, 255, 0.15)'
        }
      });
    },

    // Notification chi tiết với mô tả
    notifySuccess: (title, description, duration = 4) => {
      return contextNotification.success({
        message: title,
        description,
        duration,
        icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
        className: 'custom-notification-success-detail',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.success}30`,
          boxShadow: '0 4px 12px rgba(82, 196, 26, 0.15)'
        }
      });
    },

    notifyError: (title, description, duration = 5) => {
      return contextNotification.error({
        message: title,
        description,
        duration,
        icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
        className: 'custom-notification-error-detail',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.danger}30`,
          boxShadow: '0 4px 12px rgba(255, 77, 79, 0.15)'
        }
      });
    },

    notifyWarning: (title, description, duration = 4) => {
      return contextNotification.warning({
        message: title,
        description,
        duration,
        icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
        className: 'custom-notification-warning-detail',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.warning}30`,
          boxShadow: '0 4px 12px rgba(250, 173, 20, 0.15)'
        }
      });
    },

    notifyInfo: (title, description, duration = 4) => {
      return contextNotification.info({
        message: title,
        description,
        duration,
        icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
        className: 'custom-notification-info-detail',
        style: {
          borderRadius: '8px',
          border: `1px solid ${theme.colors.info}30`,
          boxShadow: '0 4px 12px rgba(24, 144, 255, 0.15)'
        }
      });
    },

    destroy: () => {
      contextNotification.destroy();
    }
  };
};

// Legacy Toast object for backward compatibility (sẽ có warning nhưng vẫn hoạt động)
const Toast = {
  // Sử dụng notification thay vì message để tránh context warning
  success: (content, duration = 3) => {
    return notification.success({
      message: content,
      duration,
      icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
      className: 'custom-notification-success',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.success}30`,
        boxShadow: '0 4px 12px rgba(82, 196, 26, 0.15)'
      }
    });
  },

  error: (content, duration = 4) => {
    return notification.error({
      message: content,
      duration,
      icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
      className: 'custom-notification-error',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.danger}30`,
        boxShadow: '0 4px 12px rgba(255, 77, 79, 0.15)'
      }
    });
  },

  warning: (content, duration = 3) => {
    return notification.warning({
      message: content,
      duration,
      icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
      className: 'custom-notification-warning',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.warning}30`,
        boxShadow: '0 4px 12px rgba(250, 173, 20, 0.15)'
      }
    });
  },

  info: (content, duration = 3) => {
    return notification.info({
      message: content,
      duration,
      icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
      className: 'custom-notification-info',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.info}30`,
        boxShadow: '0 4px 12px rgba(24, 144, 255, 0.15)'
      }
    });
  },

  loading: (content = 'Đang xử lý...', duration = 0) => {
    return notification.open({
      message: content,
      duration,
      icon: <LoadingOutlined style={{ color: theme.colors.info }} />,
      className: 'custom-notification-loading',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.info}30`,
        boxShadow: '0 4px 12px rgba(24, 144, 255, 0.15)'
      }
    });
  },

  // Notification chi tiết với mô tả
  notifySuccess: (title, description, duration = 4) => {
    return notification.success({
      message: title,
      description,
      duration,
      icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
      className: 'custom-notification-success-detail',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.success}30`,
        boxShadow: '0 4px 12px rgba(82, 196, 26, 0.15)'
      }
    });
  },

  notifyError: (title, description, duration = 5) => {
    return notification.error({
      message: title,
      description,
      duration,
      icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
      className: 'custom-notification-error-detail',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.danger}30`,
        boxShadow: '0 4px 12px rgba(255, 77, 79, 0.15)'
      }
    });
  },

  notifyWarning: (title, description, duration = 4) => {
    return notification.warning({
      message: title,
      description,
      duration,
      icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
      className: 'custom-notification-warning-detail',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.warning}30`,
        boxShadow: '0 4px 12px rgba(250, 173, 20, 0.15)'
      }
    });
  },

  notifyInfo: (title, description, duration = 4) => {
    return notification.info({
      message: title,
      description,
      duration,
      icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
      className: 'custom-notification-info-detail',
      style: {
        borderRadius: '8px',
        border: `1px solid ${theme.colors.info}30`,
        boxShadow: '0 4px 12px rgba(24, 144, 255, 0.15)'
      }
    });
  },

  // Cấu hình global
  config: (globalConfig) => {
    notification.config(globalConfig);
  },

  // Đóng tất cả toast
  destroy: () => {
    notification.destroy();
  }
};

export default Toast;
