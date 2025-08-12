// ToastProvider - Wrapper để cung cấp Toast context-aware
import React from 'react';
import { App } from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import theme from '../common/theme';

export const ToastProvider = ({ children }) => {
  return children;
};

// Context-aware Toast utilities
export const createToastUtils = () => {
  const { message, notification } = App.useApp();
  
  return {
    success: (content, duration = 3) => {
      return message.success({
        content,
        duration,
        icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
        className: 'custom-message-success',
      });
    },

    error: (content, duration = 4) => {
      return message.error({
        content,
        duration,
        icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
        className: 'custom-message-error',
      });
    },

    warning: (content, duration = 3) => {
      return message.warning({
        content,
        duration,
        icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
        className: 'custom-message-warning',
      });
    },

    info: (content, duration = 3) => {
      return message.info({
        content,
        duration,
        icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
        className: 'custom-message-info',
      });
    },

    loading: (content = 'Đang xử lý...', duration = 0) => {
      return message.loading({
        content,
        duration,
        className: 'custom-message-loading',
      });
    },

    notify: {
      success: (config) => {
        return notification.success({
          message: 'Thành công',
          description: '',
          ...config,
          icon: <CheckCircleOutlined style={{ color: theme.colors.success }} />,
        });
      },

      error: (config) => {
        return notification.error({
          message: 'Lỗi',
          description: '',
          ...config,
          icon: <CloseCircleOutlined style={{ color: theme.colors.danger }} />,
        });
      },

      warning: (config) => {
        return notification.warning({
          message: 'Cảnh báo',
          description: '',
          ...config,
          icon: <ExclamationCircleOutlined style={{ color: theme.colors.warning }} />,
        });
      },

      info: (config) => {
        return notification.info({
          message: 'Thông tin',
          description: '',
          ...config,
          icon: <InfoCircleOutlined style={{ color: theme.colors.info }} />,
        });
      }
    }
  };
};

export default ToastProvider;
