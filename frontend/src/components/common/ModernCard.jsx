import React from 'react';
import { Card as AntCard } from 'antd';

const ModernCard = ({ 
  variant = 'default', // 'default', 'glass', 'gradient', 'minimal', 'elevated'
  children, 
  style = {},
  className = '',
  hoverable = true,
  loading = false,
  shadow = 'medium',
  ...props 
}) => {

  const getVariantStyle = () => {
    const shadows = {
      none: 'none',
      small: '0 2px 8px rgba(0, 0, 0, 0.06)',
      medium: '0 8px 24px rgba(0, 0, 0, 0.08)',
      large: '0 16px 40px rgba(0, 0, 0, 0.12)',
      glass: '0 8px 32px rgba(31, 38, 135, 0.37)'
    };

    const baseStyle = {
      borderRadius: '16px',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      overflow: 'hidden',
      position: 'relative'
    };

    switch (variant) {
      case 'glass':
        return {
          ...baseStyle,
          background: 'rgba(255, 255, 255, 0.85)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: shadows.glass,
        };
      
      case 'gradient':
        return {
          ...baseStyle,
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%)',
          border: '1px solid rgba(226, 232, 240, 0.3)',
          boxShadow: shadows[shadow] || shadows.medium,
        };
      
      case 'minimal':
        return {
          ...baseStyle,
          background: '#ffffff',
          border: '1px solid #f0f0f0',
          boxShadow: shadows.small,
        };
      
      case 'elevated':
        return {
          ...baseStyle,
          background: '#ffffff',
          border: 'none',
          boxShadow: shadows.large,
        };
      
      default:
        return {
          ...baseStyle,
          background: '#ffffff',
          border: '1px solid #f0f0f0',
          boxShadow: shadows[shadow] || shadows.medium,
        };
    }
  };

  const hoverStyle = hoverable ? {
    cursor: 'pointer',
    '&:hover': {
      transform: 'translateY(-2px)',
      boxShadow: variant === 'glass' 
        ? '0 12px 40px rgba(31, 38, 135, 0.5)' 
        : '0 12px 32px rgba(0, 0, 0, 0.15)'
    }
  } : {};

  const cardStyle = {
    ...getVariantStyle(),
    ...hoverStyle,
    ...style
  };

  return (
    <AntCard
      {...props}
      className={`modern-card ${className}`}
      loading={loading}
      hoverable={false} // Disable default hover, use custom
      styles={{
        body: { 
          padding: '24px',
          position: 'relative'
        }
      }}
      style={cardStyle}
    >
      {variant === 'gradient' && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '4px',
          background: 'linear-gradient(90deg, #1890ff 0%, #52c41a 50%, #722ed1 100%)',
          zIndex: 1
        }} />
      )}
      {children}
    </AntCard>
  );
};

export default ModernCard;
