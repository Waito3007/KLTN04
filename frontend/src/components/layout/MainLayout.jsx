import React from 'react';
import { Layout, Typography, Space, Divider } from 'antd';
import { GithubOutlined, HeartFilled, RocketOutlined } from '@ant-design/icons';
import { theme } from '@components/common';

const { Content, Footer } = Layout;
const { Text, Link } = Typography;

const MainLayout = ({ 
  children, 
  variant = 'default', // 'default', 'gradient', 'minimal', 'glass', 'modern'
  padding = 24,
  maxWidth = '1440px',
  backgroundOverride = null,
  centered = false,
  fullHeight = true
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
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    minHeight: fullHeight ? '100vh' : 'auto'
  };

  const innerStyle = {
    maxWidth: maxWidth,
    margin: '0 auto',
    width: '100%',
    position: 'relative',
    zIndex: 1,
    flex: 1, // Cho phép Content mở rộng và đẩy footer xuống
    display: 'flex',
    flexDirection: 'column',
    ...(centered && {
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: fullHeight ? 'calc(100vh - 48px)' : 'auto'
    })
  };

  return (
    <Layout style={containerStyle}>
      {/* Decorative elements for enhanced visual appeal */}
      {variant === 'gradient' && (
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
      
      {variant === 'modern' && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '200px',
          background: 'linear-gradient(180deg, rgba(24, 144, 255, 0.05) 0%, transparent 100%)',
          zIndex: 0
        }} />
      )}
      
      <Content style={innerStyle}>
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          ...(centered && {
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100%'
          })
        }}>
          {children}
        </div>
      </Content>

      {/* Footer
      <Footer style={{
        textAlign: 'center',
        padding: '24px 50px',
        background: variant === 'gradient' ? 'rgba(255, 255, 255, 0.1)' : 
                   variant === 'glass' ? 'rgba(255, 255, 255, 0.05)' :
                   'rgba(240, 242, 245, 0.8)',
        backdropFilter: variant === 'glass' || variant === 'gradient' ? 'blur(10px)' : 'none',
        borderTop: variant === 'minimal' ? '1px solid #f0f0f0' : 'none',
        marginTop: 'auto',
        zIndex: 2,
        position: 'relative'
      }}>
        <Space direction="vertical" size="small">
          <Space size="large" wrap>
            <Space>
              <RocketOutlined style={{ color: '#1890ff' }} />
              <Text style={{ 
                color: variant === 'gradient' ? 'rgba(255, 255, 255, 0.9)' : '#666',
                fontWeight: '500'
              }}>
                KLTN04 - AI Project Management
              </Text>
            </Space>
            
            <Divider type="vertical" style={{
              borderColor: variant === 'gradient' ? 'rgba(255, 255, 255, 0.3)' : '#d9d9d9'
            }} />
            
            <Space>
              <Text style={{ 
                color: variant === 'gradient' ? 'rgba(255, 255, 255, 0.8)' : '#999' 
              }}>
                Made with
              </Text>
              <HeartFilled style={{ color: '#ff4d4f' }} />
              <Text style={{ 
                color: variant === 'gradient' ? 'rgba(255, 255, 255, 0.8)' : '#999' 
              }}>
                by SangVu, NghiaLe
              </Text>
            </Space>
            
            <Divider type="vertical" style={{
              borderColor: variant === 'gradient' ? 'rgba(255, 255, 255, 0.3)' : '#d9d9d9'
            }} />
            
            <Link 
              href="https://github.com/Waito3007/KLTN04" 
              target="_blank"
              style={{ 
                color: variant === 'gradient' ? 'rgba(255, 255, 255, 0.9)' : '#1890ff',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <GithubOutlined />
              GitHub Repository
            </Link>
          </Space>
          
          <Text style={{ 
            color: variant === 'gradient' ? 'rgba(255, 255, 255, 0.7)' : '#ccc',
            fontSize: '12px'
          }}>
            © 2025 KLTN04 Project. Powered by React + FastAPI + AI
          </Text>
        </Space>
      </Footer> */}
    </Layout>
  );
};

export default MainLayout;
