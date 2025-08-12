import React from 'react';
import { Button as AntButton } from 'antd';
import theme from './theme';

const Button = ({ 
  variant = 'primary', 
  size = 'medium', 
  gradient = false,
  glow = false,
  children, 
  style = {}, 
  ...props 
}) => {
  const getButtonStyle = () => {
    const baseStyle = {
      borderRadius: theme.borderRadius.medium,
      fontWeight: 600,
      transition: 'all 0.3s ease',
      cursor: 'pointer',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      ...style
    };

    // Gradient styles
    if (gradient) {
      baseStyle.background = theme.colors.gradient.tech;
      baseStyle.border = 'none';
      baseStyle.color = 'white';
    }

    // Glow effect
    if (glow) {
      baseStyle.boxShadow = theme.shadows.glow;
    }

    // Size variants
    const sizeStyles = {
      small: {
        height: '32px',
        padding: '0 16px',
        fontSize: theme.fontSizes.sm
      },
      medium: {
        height: '40px',
        padding: '0 20px',
        fontSize: theme.fontSizes.md
      },
      large: {
        height: '48px',
        padding: '0 24px',
        fontSize: theme.fontSizes.lg
      }
    };

    return { ...baseStyle, ...sizeStyles[size] };
  };

  return (
    <AntButton
      {...props}
      style={getButtonStyle()}
    >
      {children}
    </AntButton>
  );
};

export default Button;
