// Loading component với các variant khác nhau
import React from 'react';
import { Spin, Typography, Card, Space } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import theme from './theme';

const { Text } = Typography;

// Custom loading icon với animation đẹp hơn
const CustomLoadingIcon = ({ size = 24, color }) => (
  <div 
    style={{
      width: size,
      height: size,
      borderRadius: '50%',
      background: `conic-gradient(from 0deg, ${color || theme.colors.primary}, transparent)`,
      animation: 'spin 1s linear infinite',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}
  >
    <div style={{
      width: size - 4,
      height: size - 4,
      borderRadius: '50%',
      background: theme.colors.white,
    }} />
  </div>
);

const Loading = ({ 
  size = 'default', 
  text = 'Đang tải...', 
  variant = 'default',
  spinning = true,
  style = {},
  children 
}) => {
  // Kích thước icon
  const iconSizes = {
    small: 16,
    default: 24,
    large: 32,
  };

  const iconSize = iconSizes[size] || iconSizes.default;
  
  const customIndicator = (
    <CustomLoadingIcon 
      size={iconSize} 
      color={variant === 'gradient' ? theme.colors.white : theme.colors.primary}
    />
  );

  const getContainerStyle = () => {
    const baseStyle = {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      flexDirection: 'column',
      gap: theme.spacing.md,
      ...style
    };

    switch (variant) {
      case 'fullscreen':
        return {
          ...baseStyle,
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: theme.colors.bg.glass,
          backdropFilter: 'blur(8px)',
          zIndex: theme.zIndex.modal,
        };
      
      case 'overlay':
        return {
          ...baseStyle,
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(255, 255, 255, 0.9)',
          backdropFilter: 'blur(4px)',
          borderRadius: theme.borderRadius.md,
          zIndex: theme.zIndex.popover,
        };
      
      case 'gradient':
        return {
          ...baseStyle,
          minHeight: '200px',
          background: theme.colors.gradient.primary,
          borderRadius: theme.borderRadius.xl,
          padding: theme.spacing.xxl,
          boxShadow: theme.shadows.lg,
        };
      
      case 'card':
        return {
          ...baseStyle,
          padding: theme.spacing.xl,
          background: theme.colors.bg.glass,
          borderRadius: theme.borderRadius.lg,
          boxShadow: theme.shadows.md,
          border: 'none',
        };
      
      case 'inline':
        return {
          ...baseStyle,
          padding: theme.spacing.md,
          flexDirection: 'row',
          gap: theme.spacing.sm,
        };
      
      default:
        return {
          ...baseStyle,
          padding: theme.spacing.lg,
        };
    }
  };

  const getTextStyle = () => {
    const baseStyle = {
      fontSize: theme.fontSizes.sm,
      fontWeight: theme.fontWeights.medium,
    };

    switch (variant) {
      case 'gradient':
        return { 
          ...baseStyle,
          color: theme.colors.text.inverse,
          fontSize: theme.fontSizes.md,
        };
      case 'fullscreen':
        return { 
          ...baseStyle,
          color: theme.colors.text.primary,
          fontSize: theme.fontSizes.lg,
        };
      default:
        return { 
          ...baseStyle,
          color: theme.colors.text.secondary,
        };
    }
  };

  // Nếu có children, sử dụng Spin wrapper
  if (children) {
    return (
      <Spin 
        spinning={spinning} 
        size={size}
        indicator={customIndicator}
        style={{ 
          borderRadius: theme.borderRadius.md,
        }}
      >
        <div style={{
          transition: theme.transitions.normal,
          opacity: spinning ? 0.5 : 1,
        }}>
          {children}
        </div>
      </Spin>
    );
  }

  // Variant đặc biệt cho fullscreen với animation
  if (variant === 'fullscreen') {
    return (
      <div style={getContainerStyle()}>
        <div style={{
          background: theme.colors.gradient.primary,
          borderRadius: theme.borderRadius.round,
          width: 80,
          height: 80,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: theme.spacing.lg,
          boxShadow: theme.shadows.glow,
          animation: 'pulse 2s infinite',
        }}>
          <CustomLoadingIcon size={36} color={theme.colors.white} />
        </div>
        <Text style={getTextStyle()}>{text}</Text>
        <Text style={{ 
          color: theme.colors.text.tertiary,
          fontSize: theme.fontSizes.sm,
          marginTop: theme.spacing.sm,
        }}>
          Vui lòng đợi trong giây lát...
        </Text>
      </div>
    );
  }

  // Variant card với design đẹp
  if (variant === 'card') {
    return (
      <Card
        style={{
          borderRadius: theme.borderRadius.lg,
          border: 'none',
          boxShadow: theme.shadows.md,
          background: theme.colors.bg.glass,
        }}
        styles={{
          body: {
            padding: theme.spacing.xl,
            textAlign: 'center',
          }
        }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div style={{
            background: theme.colors.gradient.primary,
            borderRadius: theme.borderRadius.round,
            width: 50,
            height: 50,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto',
            boxShadow: theme.shadows.md,
            animation: 'rotate 2s linear infinite',
          }}>
            <CustomLoadingIcon size={24} color={theme.colors.white} />
          </div>
          <div>
            <Text style={{ 
              fontSize: theme.fontSizes.md,
              fontWeight: theme.fontWeights.medium,
              color: theme.colors.text.primary,
              display: 'block',
              marginBottom: theme.spacing.xs,
            }}>
              {text}
            </Text>
            <Text style={{ 
              fontSize: theme.fontSizes.sm,
              color: theme.colors.text.tertiary,
            }}>
              Đang xử lý yêu cầu của bạn
            </Text>
          </div>
        </Space>
      </Card>
    );
  }

  // Standalone loading mặc định
  return (
    <div style={getContainerStyle()}>
      <Spin 
        size={size} 
        spinning={spinning}
        indicator={customIndicator}
      />
      {text && (
        <Text style={getTextStyle()}>
          {text}
        </Text>
      )}
    </div>
  );
};

export default Loading;

// CSS animations
const loadingStyles = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
  }
  
  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;

// Inject styles vào document
if (typeof document !== 'undefined' && !document.getElementById('loading-styles')) {
  const styleElement = document.createElement('style');
  styleElement.id = 'loading-styles';
  styleElement.textContent = loadingStyles;
  document.head.appendChild(styleElement);
}
