import React from 'react';
import { Card as AntCard } from 'antd';
import theme from './theme';

const Card = ({ 
  variant = 'default',
  glassMorphism = false,
  hoverable = true,
  children, 
  style = {},
  ...props 
}) => {
  const getVariantStyle = () => {
    // Defensive programming: fallback nếu theme không được import đúng hoặc thiếu properties
    const safeTheme = theme || {};
    const safeBorderRadius = safeTheme.borderRadius || { lg: '8px' };
    const safeTransitions = safeTheme.transitions || { all: 'all 0.3s ease' };
    const safeShadows = safeTheme.shadows || { 
      sm: '0 1px 3px rgba(0,0,0,0.1)', 
      md: '0 4px 6px rgba(0,0,0,0.1)', 
      lg: '0 10px 15px rgba(0,0,0,0.1)', 
      glass: '0 8px 32px rgba(31, 38, 135, 0.37)', 
      modern: '0 8px 24px rgba(0, 0, 0, 0.12)' 
    };
    const safeColors = safeTheme.colors || {
      white: '#ffffff',
      primary: '#1890ff',
      secondary: '#6C757D',
      border: { light: '#f0f0f0' },
      bg: { glass: 'rgba(255, 255, 255, 0.95)' },
      gradient: { 
        info: 'linear-gradient(135deg, #1890ff 0%, #13c2c2 100%)',
        primary: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }
    };
    
    const baseStyle = {
      borderRadius: safeBorderRadius.lg,
      transition: safeTransitions.all,
    };

    if (glassMorphism || variant === 'glassMorphism') {
      return {
        ...baseStyle,
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: safeShadows.glass,
      };
    }

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          background: `linear-gradient(135deg, ${safeColors.primary} 0%, ${safeColors.secondary} 100%)`,
          color: 'white',
          border: 'none',
          boxShadow: safeShadows.lg,
        };
      
      case 'gradient':
        return {
          ...baseStyle,
          background: safeColors.gradient.primary,
          color: 'white',
          border: 'none',
          boxShadow: safeShadows.lg,
        };

      case 'stats':
        return {
          ...baseStyle,
          background: safeColors.gradient.info,
          color: 'white',
          border: 'none',
          boxShadow: safeShadows.md,
        };

      case 'modern':
        return {
          ...baseStyle,
          background: safeColors.bg.glass,
          backdropFilter: 'blur(12px)',
          border: `1px solid ${safeColors.border.light}`,
          boxShadow: safeShadows.modern,
        };

      default:
        return {
          ...baseStyle,
          background: safeColors.white,
          border: `1px solid ${safeColors.border.light}`,
          boxShadow: safeShadows.sm,
        };
    }
  };

  const cardStyle = {
    ...getVariantStyle(),
    ...style,
  };

  return (
    <AntCard
      hoverable={hoverable}
      style={cardStyle}
      styles={{
        body: { 
          padding: theme?.spacing?.md || '16px',
          color: variant === 'primary' || variant === 'gradient' || variant === 'stats' ? 'white' : 'inherit'
        }
      }}
      {...props}
    >
      {children}
    </AntCard>
  );
};

export default Card;
