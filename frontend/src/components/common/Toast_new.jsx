// Toast component để hiển thị thông báo nhất quán
import { message, notification } from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import theme from './theme';
import './toast.css'; // Import CSS styling cho toast overlay

// Cấu hình mặc định cho message - đảm bảo hiển thị đúng vị trí
message.config({
  top: 24,
  duration: 3,
  maxCount: 3,
  getContainer: () => document.body, // Đảm bảo hiển thị trên body
});

// Cấu hình mặc định cho notification - đảm bảo hiển thị đúng vị trí
notification.config({
  placement: 'topRight',
  top: 24,
  duration: 4.5,
  rtl: false,
  getContainer: () => document.body, // Đảm bảo hiển thị trên body
});

const Toast = {
  // Message toast - ngắn gọn, hiển thị đúng overlay
  success: (content, duration = 3) => {
    return message.success({
      content,
      duration,
      icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
      className: 'custom-message-success',
      style: {
        marginTop: '24px',
        zIndex: 9999,
      }
    });
  },

  error: (content, duration = 4) => {
    return message.error({
      content,
      duration,
      icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
      className: 'custom-message-error',
      style: {
        marginTop: '24px',
        zIndex: 9999,
      }
    });
  },

  warning: (content, duration = 3) => {
    return message.warning({
      content,
      duration,
      icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
      className: 'custom-message-warning',
      style: {
        marginTop: '24px',
        zIndex: 9999,
      }
    });
  },

  info: (content, duration = 3) => {
    return message.info({
      content,
      duration,
      icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
      className: 'custom-message-info',
      style: {
        marginTop: '24px',
        zIndex: 9999,
      }
    });
  },

  loading: (content = 'Đang xử lý...', duration = 0) => {
    return message.loading({
      content,
      duration,
      className: 'custom-message-loading',
      style: {
        marginTop: '24px',
        zIndex: 9999,
      }
    });
  },

  // Notification - chi tiết hơn
  notify: {
    success: (config) => {
      return notification.success({
        message: 'Thành công',
        description: '',
        ...config,
        icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
        style: {
          borderRadius: theme.borderRadius.lg,
          boxShadow: theme.shadows.lg,
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(8px)',
          border: `1px solid ${theme.colors.success}20`,
        }
      });
    },

    error: (config) => {
      return notification.error({
        message: 'Lỗi',
        description: '',
        ...config,
        icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
        style: {
          borderRadius: theme.borderRadius.lg,
          boxShadow: theme.shadows.lg,
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(8px)',
          border: `1px solid ${theme.colors.danger}20`,
        }
      });
    },

    warning: (config) => {
      return notification.warning({
        message: 'Cảnh báo',
        description: '',
        ...config,
        icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
        style: {
          borderRadius: theme.borderRadius.lg,
          boxShadow: theme.shadows.lg,
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(8px)',
          border: `1px solid ${theme.colors.warning}20`,
        }
      });
    },

    info: (config) => {
      return notification.info({
        message: 'Thông tin',
        description: '',
        ...config,
        icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
        style: {
          borderRadius: theme.borderRadius.lg,
          boxShadow: theme.shadows.lg,
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(8px)',
          border: `1px solid ${theme.colors.info}20`,
        }
      });
    },

    // Progress notification
    progress: (config) => {
      return notification.open({
        message: 'Đang xử lý',
        description: 'Vui lòng chờ...',
        ...config,
        duration: 0, // Không tự động đóng
        className: 'custom-notification-progress',
        style: {
          borderRadius: theme.borderRadius.lg,
          boxShadow: theme.shadows.lg,
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(8px)',
        }
      });
    }
  },

  // Cấu hình global
  config: (globalConfig) => {
    message.config(globalConfig);
    notification.config(globalConfig);
  },

  // Đóng tất cả toast
  destroy: () => {
    message.destroy();
    notification.destroy();
  }
};

export default Toast;
