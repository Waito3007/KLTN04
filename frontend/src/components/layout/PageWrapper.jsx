import React from 'react';
import { Layout } from 'antd';
import { theme } from '@components/common';

const { Content } = Layout;

const PageWrapper = ({ 
  children, 
  variant = 'default', // 'default', 'gradient', 'minimal', 'glass', 'modern'
  padding = 24,
  maxWidth = '1440px',
  backgroundOverride = null,
  centered = false,
  fullHeight = true,
  title,
  subtitle,
  showDecorations = true
}) => {
  
  const getBackgroundStyle = () => {
    if (backgroundOverride) return { background: backgroundOverride };
    
    switch (variant) {
      case 'gradient':
        return {
          background: `
            linear-gradient(135deg, #667eea 0%, #764ba2 100%),
            radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.05) 0%, transparent 50%)
          `,
          minHeight: fullHeight ? '100vh' : 'auto',
          position: 'relative',
          overflow: 'hidden'
        };
      case 'glass':
        return {
          background: `
            linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)),
            linear-gradient(135deg, #f0f2f5 0%, #e6f7ff 100%)
          `,
          backdropFilter: 'blur(20px)',
          minHeight: fullHeight ? '100vh' : 'auto'
        };
      case 'modern':
        return {
          background: `
            linear-gradient(180deg, #fafbfc 0%, #f5f7fa 100%),
            radial-gradient(ellipse at top, rgba(24, 144, 255, 0.1) 0%, transparent 50%)
          `,
          minHeight: fullHeight ? '100vh' : 'auto'
        };
      case 'minimal':
        return {
          background: theme?.colors?.bg?.primary || '#ffffff',
          minHeight: fullHeight ? '100vh' : 'auto'
        };
      default:
        return {
          background: `
            linear-gradient(180deg, #fafbfc 0%, #f0f2f5 100%)
          `,
          minHeight: fullHeight ? '100vh' : 'auto'
        };
    }
  };

  const containerStyle = {
    ...getBackgroundStyle(),
    padding: padding,
    position: 'relative'
  };

  const innerStyle = {
    maxWidth: maxWidth,
    margin: '0 auto',
    width: '100%',
    position: 'relative',
    zIndex: 1,
    ...(centered && {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: fullHeight ? 'calc(100vh - 48px)' : 'auto'
    })
  };

  return (
    <Layout style={containerStyle}>
      {/* Decorative elements for enhanced visual appeal */}
      {showDecorations && variant === 'gradient' && (
        <>
          <div style={{
            position: 'absolute',
            top: '10%',
            right: '10%',
            width: '100px',
            height: '100px',
            borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.1)',
            filter: 'blur(40px)',
            zIndex: 0
          }} />
          <div style={{
            position: 'absolute',
            bottom: '20%',
            left: '5%',
            width: '150px',
            height: '150px',
            borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.05)',
            filter: 'blur(60px)',
            zIndex: 0
          }} />
        </>
      )}
      
      {showDecorations && variant === 'modern' && (
        <>
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '200px',
            background: 'linear-gradient(180deg, rgba(24, 144, 255, 0.05) 0%, transparent 100%)',
            zIndex: 0
          }} />
          <div style={{
            position: 'absolute',
            top: '15%',
            right: '15%',
            width: '80px',
            height: '80px',
            borderRadius: '50%',
            background: 'rgba(24, 144, 255, 0.1)',
            filter: 'blur(30px)',
            zIndex: 0
          }} />
        </>
      )}
      
      <Content style={innerStyle}>
        {(title || subtitle) && (
          <div style={{
            textAlign: centered ? 'center' : 'left',
            marginBottom: '32px',
            ...(variant === 'gradient' && { color: '#ffffff' })
          }}>
            {title && (
              <h1 style={{
                fontSize: '32px',
                fontWeight: 700,
                margin: '0 0 8px 0',
                background: variant === 'gradient' 
                  ? 'linear-gradient(45deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7))'
                  : 'linear-gradient(45deg, #1890ff, #722ed1)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: variant === 'gradient' ? 'white' : 'transparent',
                backgroundClip: 'text'
              }}>
                {title}
              </h1>
            )}
            {subtitle && (
              <p style={{
                fontSize: '16px',
                color: variant === 'gradient' ? 'rgba(255,255,255,0.8)' : '#595959',
                margin: 0
              }}>
                {subtitle}
              </p>
            )}
          </div>
        )}
        {children}
      </Content>
    </Layout>
  );
};

export default PageWrapper;
