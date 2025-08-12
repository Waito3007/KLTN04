// Modal component với design hiện đại
import React from 'react';
import { Modal, Typography, Button, Space } from 'antd';
import { CloseOutlined, ExclamationCircleOutlined, CheckCircleOutlined, InfoCircleOutlined } from '@ant-design/icons';
import theme from './theme';

const { Title, Text } = Typography;

const CustomModal = ({
  visible = false,
  onClose,
  onConfirm,
  title = 'Thông báo',
  content,
  type = 'default', // default, confirm, info, success, warning, danger
  size = 'default', // small, default, large
  confirmText = 'Xác nhận',
  cancelText = 'Hủy',
  confirmLoading = false,
  showCancel = true,
  showConfirm = true,
  closable = true,
  maskClosable = true,
  children,
  ...props
}) => {
  const getIcon = () => {
    const iconStyle = { 
      fontSize: '24px',
      marginBottom: theme.spacing.sm,
    };

    switch (type) {
      case 'success':
        return <CheckCircleOutlined style={{ ...iconStyle, color: theme.colors.success }} />;
      case 'warning':
        return <ExclamationCircleOutlined style={{ ...iconStyle, color: theme.colors.warning }} />;
      case 'danger':
        return <ExclamationCircleOutlined style={{ ...iconStyle, color: theme.colors.danger }} />;
      case 'info':
        return <InfoCircleOutlined style={{ ...iconStyle, color: theme.colors.info }} />;
      default:
        return null;
    }
  };

  const getWidth = () => {
    switch (size) {
      case 'small':
        return 400;
      case 'large':
        return 800;
      default:
        return 520;
    }
  };

  const getModalStyle = () => {
    return {
      borderRadius: theme.borderRadius.xl,
      overflow: 'hidden',
      boxShadow: theme.shadows.xl,
    };
  };

  const getBodyStyle = () => {
    return {
      padding: theme.spacing.xl,
      background: theme.colors.gradient.subtle,
      minHeight: '120px',
    };
  };

  const getHeaderStyle = () => {
    return {
      background: theme.colors.gradient.primary,
      borderBottom: 'none',
      padding: theme.spacing.lg,
      color: theme.colors.white,
    };
  };

  const renderFooter = () => {
    if (!showCancel && !showConfirm) return null;

    return (
      <div style={{
        padding: theme.spacing.lg,
        background: theme.colors.bg.secondary,
        borderTop: `1px solid ${theme.colors.border.light}`,
        display: 'flex',
        justifyContent: 'flex-end',
        gap: theme.spacing.sm,
      }}>
        <Space>
          {showCancel && (
            <Button 
              onClick={onClose}
              size="large"
              style={{
                borderRadius: theme.borderRadius.lg,
                fontWeight: theme.fontWeights.medium,
                height: 40,
                paddingLeft: theme.spacing.lg,
                paddingRight: theme.spacing.lg,
                borderColor: theme.colors.border.default,
                transition: theme.transitions.normal,
              }}
            >
              {cancelText}
            </Button>
          )}
          {showConfirm && (
            <Button 
              type="primary" 
              onClick={onConfirm}
              loading={confirmLoading}
              size="large"
              style={{
                background: theme.colors.gradient.primary,
                border: 'none',
                borderRadius: theme.borderRadius.lg,
                fontWeight: theme.fontWeights.medium,
                height: 40,
                paddingLeft: theme.spacing.lg,
                paddingRight: theme.spacing.lg,
                boxShadow: theme.shadows.md,
                transition: theme.transitions.normal,
              }}
            >
              {confirmText}
            </Button>
          )}
        </Space>
      </div>
    );
  };

  return (
    <Modal
      title={
        <div style={{ 
          color: theme.colors.white, 
          display: 'flex', 
          alignItems: 'center', 
          gap: theme.spacing.sm,
          fontSize: theme.fontSizes.lg,
          fontWeight: theme.fontWeights.semibold,
        }}>
          {getIcon()}
          {title}
        </div>
      }
      open={visible}
      onCancel={onClose}
      width={getWidth()}
      style={getModalStyle()}
      styles={{
        header: getHeaderStyle(),
        body: getBodyStyle(),
      }}
      footer={renderFooter()}
      closeIcon={
        <CloseOutlined 
          style={{ 
            color: theme.colors.white, 
            fontSize: theme.fontSizes.md,
            padding: theme.spacing.xs,
          }} 
        />
      }
      closable={closable}
      maskClosable={maskClosable}
      centered
      {...props}
    >
      <div style={{ minHeight: '60px' }}>
        {content && (
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: theme.spacing.sm,
            marginBottom: theme.spacing.md,
          }}>
            {type !== 'default' && getIcon()}
            <div style={{ flex: 1 }}>
              <Text style={{ 
                fontSize: theme.fontSizes.md, 
                lineHeight: '1.6',
                color: theme.colors.text.primary,
              }}>
                {content}
              </Text>
            </div>
          </div>
        )}
        {children}
      </div>
    </Modal>
  );
};

// Utility functions để tạo modal nhanh
CustomModal.confirm = (config) => {
  return Modal.confirm({
    ...config,
    okButtonProps: {
      style: {
        background: theme.colors.gradient.primary,
        border: 'none',
        borderRadius: theme.borderRadius.lg,
        fontWeight: theme.fontWeights.medium,
      }
    },
    cancelButtonProps: {
      style: {
        borderRadius: theme.borderRadius.lg,
        fontWeight: theme.fontWeights.medium,
      }
    }
  });
};

CustomModal.info = (config) => {
  return Modal.info({
    ...config,
    okButtonProps: {
      style: {
        background: theme.colors.gradient.info,
        border: 'none',
        borderRadius: theme.borderRadius.lg,
        fontWeight: theme.fontWeights.medium,
      }
    }
  });
};

CustomModal.success = (config) => {
  return Modal.success({
    ...config,
    okButtonProps: {
      style: {
        background: theme.colors.gradient.success,
        border: 'none',
        borderRadius: theme.borderRadius.lg,
        fontWeight: theme.fontWeights.medium,
      }
    }
  });
};

CustomModal.error = (config) => {
  return Modal.error({
    ...config,
    okButtonProps: {
      style: {
        background: theme.colors.gradient.danger,
        border: 'none',
        borderRadius: theme.borderRadius.lg,
        fontWeight: theme.fontWeights.medium,
      }
    }
  });
};

CustomModal.warning = (config) => {
  return Modal.warning({
    ...config,
    okButtonProps: {
      style: {
        background: theme.colors.gradient.warning,
        border: 'none',
        borderRadius: theme.borderRadius.lg,
        fontWeight: theme.fontWeights.medium,
      }
    }
  });
};

export default CustomModal;
